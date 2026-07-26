"""Бесконечный поток пар «красивый кадр -> каноническая карта».

Зачем такой датасет
-------------------
Разметку для сегментации обычно рисуют руками, и её всегда мало. Здесь
симулятор сам знает истину: `render_semantic` строит эталонную карту, а
`Renderer` — картинку в произвольном оформлении из той же камеры. Значит,
данных можно сделать бесконечно много и с любым разбросом дизайна. Датасет
ничего не хранит на диске: он генерирует пары на лету прямо в
worker-процессах DataLoader'а.

Честная валидация — только на отложенных темах
----------------------------------------------
Главный вопрос к зрению: «а сработает ли оно на дизайне, которого не было в
обучении?» Проверять это на тех же темах бессмысленно — сеть могла просто
запомнить их палитры. Поэтому `BUILTIN_THEMES` жёстко разбит на два
непересекающихся списка (`TRAIN_THEME_NAMES` / `HELD_OUT_THEME_NAMES`), и
валидационный поток видит ТОЛЬКО отложенные. Разбиение фиксировано именами, а
не случайной перестановкой: иначе «held-out» менялся бы от запуска к запуску и
метрики двух прогонов было бы нельзя сравнивать.

Сверх отложенных встроенных тем валидация включает и полностью случайные темы
(`random_theme`) — но из ОТДЕЛЬНОГО генератора случайных чисел, чей поток
никогда не используется при обучении. Это вторая, более строгая проверка: тема
собирается из непрерывного пространства палитр и стилей, и совпасть с
обучающей она не может.

Производительность
------------------
Обучение идёт на CPU, поэтому данные не имеют права быть узким местом:

* уровни ДОРОГИЕ (генерация с проверкой проходимости — сотни миллисекунд),
  поэтому они генерируются один раз в пул и переиспользуются, а пул медленно
  обновляется по ходу обучения;
* `Renderer` создаётся ОДИН на worker и живёт весь прогон: он держит кэши фона,
  параллакса и спрайтов;
* смена темы (~2 мс) и рендер кадра (~2 мс) идут параллельно в нескольких
  worker-процессах, полностью прячась за шагом оптимизатора.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from gdai.config import PerceptionConfig
from gdai.constants import GROUND_Y, HAZARD, OBS_H, OBS_W, SOLID
from gdai.env.generator import generate_level
from gdai.env.level import Level, LevelObject
from gdai.env.physics import PlayerState, make_initial_state, step_physics
from gdai.env.render import Renderer
from gdai.env.semantic import render_semantic
from gdai.env.themes import (
    BUILTIN_THEMES,
    Theme,
    theme_by_name,
    to_inverted,
    to_monochrome,
)
from gdai.perception.augment import AugmentConfig, DEFAULT_AUGMENT, augment_frame
from gdai.utils.logging import get_logger

_LOG = get_logger("perception.dataset")

# --- разбиение тем ----------------------------------------------------------
# Отложенные темы выбраны так, чтобы покрыть РАЗНЫЕ оси дизайна, а не четыре
# похожие картинки: тёмный неон с обводкой (cyberpunk), светлая мягкая палитра
# (pastel), агрессивный шум и полосы (glitch), светлая «стеклянная» со скруглениями
# (ice). Если зрение справляется с ними, не видев ни одной, — оно опирается на
# форму, а не на цвет.
HELD_OUT_THEME_NAMES: tuple[str, ...] = ("cyberpunk", "pastel", "glitch", "ice")
TRAIN_THEME_NAMES: tuple[str, ...] = tuple(
    t.name for t in BUILTIN_THEMES if t.name not in HELD_OUT_THEME_NAMES
)

# Сколько пар отдаёт валидационный поток за проход. 256 кадров при 9216
# пикселях каждый — это ~2.4 млн размеченных пикселей: достаточно, чтобы IoU по
# редким классам не прыгал от одного удачного кадра, и достаточно мало, чтобы
# валидация занимала секунды.
VAL_SAMPLES: int = 256

# Смещения зерна для независимых потоков случайности. Разные константы гарантируют,
# что «случайные темы валидации» никогда не повторят поток обучения.
_SEED_LEVELS: int = 1_000_003
_SEED_SAMPLES: int = 2_000_003
_SEED_THEMES: int = 3_000_003
_SEED_VAL_THEMES: int = 7_654_321


def train_themes() -> tuple[Theme, ...]:
    """Встроенные темы, разрешённые в обучении (без отложенных)."""
    return tuple(theme_by_name(name) for name in TRAIN_THEME_NAMES)


def held_out_themes() -> tuple[Theme, ...]:
    """Отложенные встроенные темы — их видит только валидация."""
    return tuple(theme_by_name(name) for name in HELD_OUT_THEME_NAMES)


def check_theme_split() -> None:
    """Проверить, что train и val темы не пересекаются и покрывают все встроенные.

    Зачем отдельная функция: это ключевое свойство честности эксперимента, и
    оно должно проверяться и тестом, и самим датасетом при создании — опечатка
    в одном имени тихо превратила бы «обобщение» в «запоминание».
    """
    train = set(TRAIN_THEME_NAMES)
    held = set(HELD_OUT_THEME_NAMES)
    all_names = {t.name for t in BUILTIN_THEMES}
    if train & held:
        raise ValueError(f"Темы попали и в train, и в val: {sorted(train & held)}")
    if not held:
        raise ValueError("Список отложенных тем пуст — валидация станет нечестной")
    missing = held - all_names
    if missing:
        raise ValueError(f"Отложенных тем нет среди встроенных: {sorted(missing)}")
    if train | held != all_names:
        raise ValueError("Разбиение тем не покрывает BUILTIN_THEMES целиком")


@dataclass
class DatasetConfig:
    """Параметры генерации пар (то, чего нет в `PerceptionConfig`).

    Отдельная структура нужна, потому что `PerceptionConfig` — контракт из
    SPEC §3 и расширять его нельзя, а ручки генерации данных всё равно нужно
    уметь передавать в тестах и в отладке.
    """

    level_pool: int = 8               # сколько разных уровней держит один worker
    eager_levels: int = 3             # сколько сгенерировать сразу (остальные — на ходу)
    difficulty: tuple[float, float] = (0.0, 1.0)
    random_theme_prob: float = 0.65   # доля полностью случайных тем
    decoration_level: float = 1.0     # верхняя граница плотности декора
    bare_prob: float = 0.12           # доля кадров с почти голым уровнем
    # Диапазон плотности декора для «обычных» (не голых) кадров, доля от
    # `decoration_level`. Вынесен в конфиг, чтобы проверку «сеть не ломается при
    # decoration_level=1.0» можно было провести честно: с (1.0, 1.0) КАЖДЫЙ кадр
    # рисуется с максимумом декораций, а не в среднем на 0.75.
    deco_range: tuple[float, float] = (0.5, 1.0)
    # «Злые» варианты встроенной темы в ОБУЧЕНИИ: обесцвеченная и инвертированная
    # палитра. Валидации они не достаются намеренно — её смысл в том, чтобы
    # мерить обобщение на фиксированный, ни разу не виденный набор дизайнов, и
    # состав этого набора не должен плыть.
    theme_mono_prob: float = 0.12
    theme_invert_prob: float = 0.12
    # Перетасовывать ли зерно узора у встроенной темы. Зачем: палитра темы
    # задаёт цвета, а `Theme.seed` — РАСКЛАДКУ звёзд, облаков и полос фона. Без
    # перетасовки десять встроенных тем давали ровно десять фоновых картинок, и
    # «звезда вот в этом месте» становилась признаком темы.
    theme_seed_jitter: bool = True
    grow_prob: float = 0.08           # шанс дорастить пул уровней на очередном сэмпле
    refresh_prob: float = 0.004       # шанс заменить уровень в пуле на новый
    rollout_steps: int = 24           # длина случайного «доигрывания» состояния
    ship_prob: float = 0.22           # доля кадров в режиме корабля
    wave_prob: float = 0.13           # доля кадров в режиме волны
    flip_gravity_prob: float = 0.18   # доля кадров с перевёрнутой гравитацией
    # Доля кадров, НАВЕДЁННЫХ на редкий объект (шип, пила, кольцо, пад, портал,
    # финиш). Без этого камера почти всегда смотрит в пустоту: измерения по
    # честно-случайным состояниям дают около 5 пикселей HAZARD на кадр из 9216,
    # то есть 0.05% — при такой плотности сеть тысячи шагов не видит ни одного
    # положительного примера и уверенно предсказывает «пусто и пол».
    focus_prob: float = 0.75
    # Какая доля наведений приходится на ОПАСНОСТИ (остальное — пады, кольца,
    # порталы, финиш). Шип — самый мелкий объект игры и единственный, чей
    # пропуск стоит агенту жизни, поэтому ему отдан перевес.
    focus_hazard_prob: float = 0.6
    # Насколько далеко перед игроком может оказаться выбранный объект, в тайлах.
    # Камера показывает [x-4, x+12], поэтому диапазон подобран так, чтобы объект
    # гарантированно попал в кадр и при этом бывал и близко, и на горизонте.
    focus_ahead: tuple[float, float] = (-2.5, 10.5)


DEFAULT_DATASET: DatasetConfig = DatasetConfig()


def _mix_seed(*parts: int) -> int:
    """Смешать несколько целых в одно зерно (детерминированно и без коллизий).

    Зачем не сложение: `seed + worker_id` у двух потоков с соседними базовыми
    зёрнами даёт пересекающиеся последовательности, и часть данных оказывается
    дубликатами.
    """
    seq = np.random.SeedSequence([int(p) & 0xFFFF_FFFF_FFFF for p in parts])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _load_level_files(levels_dir: str | os.PathLike[str] | None) -> list[Level]:
    """Загрузить уровни из каталога `levels/`, если он есть.

    Зачем: рукотворные уровни содержат сочетания объектов, которых генератор
    может не построить (например, «портал прямо над пилой»), и зрению полезно
    их увидеть. Отсутствие каталога — не ошибка, а норма.
    """
    if levels_dir is None:
        return []
    path = Path(levels_dir)
    if not path.is_dir():
        return []
    levels: list[Level] = []
    for file in sorted(path.glob("*.json")):
        try:
            levels.append(Level.load(file))
        except Exception as exc:  # уровень мог быть сохранён другой версией
            _LOG.warning("уровень %s не загружен: %s", file, exc)
    return levels


def _focus_objects(level: Level) -> tuple[tuple[LevelObject, ...], tuple[LevelObject, ...]]:
    """Объекты уровня, ради которых стоит навести камеру: (опасности, прочие).

    Это всё, что НЕ пол и не блок: шипы и пилы отдельно, пады/кольца/порталы/
    финиш отдельно. Разделение нужно, потому что HAZARD — самый мелкий и самый
    важный класс: шип занимает около десяти пикселей кадра, и если наводиться
    равномерно по всем объектам, он всё равно останется исчезающе редким.
    """
    hazards = tuple(obj for obj in level.objects if obj.semantic_class() == HAZARD)
    others = tuple(
        obj for obj in level.objects
        if obj.semantic_class() > SOLID and obj.semantic_class() != HAZARD
    )
    return hazards, others


class SyntheticSegDataset(IterableDataset):
    """Бесконечный (или конечный для валидации) поток пар (кадр, карта).

    Каждый сэмпл собирается из трёх независимых случайностей:

    1. **уровень** — из пула процедурных уровней случайной сложности;
    2. **состояние игрока** — случайные x, высота, скорость, режим и гравитация,
       после чего состояние «доигрывается» несколькими кадрами физики, чтобы
       поза была достижимой, а не нарисованной наугад;
    3. **оформление** — случайная тема (или встроенная из разрешённого списка),
       случайная плотность декора и случайная фаза анимации.

    Возвращает `(frame, label)`: `frame` — float32 (3, H, W) в [0,1],
    `label` — int64 (H, W) с классами 0..9.
    """

    def __init__(
        self,
        split: str = "train",
        seed: int = 0,
        samples: int | None = None,
        augment: bool | None = None,
        width: int = OBS_W,
        height: int = OBS_H,
        data_cfg: DatasetConfig | None = None,
        augment_cfg: AugmentConfig | None = None,
        levels_dir: str | os.PathLike[str] | None = "levels",
    ) -> None:
        super().__init__()
        split = str(split).lower()
        if split not in ("train", "val"):
            raise ValueError(f"split должен быть 'train' или 'val', получено {split!r}")
        check_theme_split()

        self.split = split
        self.seed = int(seed)
        self.samples = None if samples is None else int(samples)
        self.width = int(width)
        self.height = int(height)
        self.cfg = data_cfg if data_cfg is not None else DEFAULT_DATASET
        self.augment = (split == "train") if augment is None else bool(augment)
        self.augment_cfg = augment_cfg if augment_cfg is not None else DEFAULT_AUGMENT
        # Темы: обучение видит только разрешённые, валидация — только отложенные.
        self.themes: tuple[Theme, ...] = (
            train_themes() if split == "train" else held_out_themes()
        )

        # Пул уровней. Валидация и обучение строят его из РАЗНЫХ зёрен: даже
        # набор препятствий на валидации не должен совпадать с обучающим.
        self._level_seed = _mix_seed(self.seed, _SEED_LEVELS, 0 if split == "train" else 991)
        self._levels: list[Level] = _load_level_files(levels_dir)
        eager = max(1, min(int(self.cfg.eager_levels), int(self.cfg.level_pool)))
        pool_rng = np.random.default_rng(self._level_seed)
        for i in range(eager):
            self._levels.append(self._make_level(pool_rng, i))
        # Списки «интересных» объектов, параллельные пулу уровней: считаются один
        # раз на уровень, потому что нужны на КАЖДОМ сэмпле (см. focus_prob).
        self._focus: list[tuple[tuple[LevelObject, ...], tuple[LevelObject, ...]]] = [
            _focus_objects(level) for level in self._levels
        ]
        # Ленивое состояние worker'а: создаётся в __iter__, не переживает pickle.
        self._renderer: Renderer | None = None

    # --- построение частей сэмпла -------------------------------------------
    def _make_level(self, rng: np.random.Generator, index: int) -> Level:
        """Новый процедурный уровень случайной сложности из заданного диапазона."""
        lo, hi = self.cfg.difficulty
        difficulty = float(rng.uniform(float(lo), float(hi)))
        return generate_level(difficulty, rng, name=f"{self.split}_{index}")

    def _pick_level(
        self, rng: np.random.Generator, level_rng: np.random.Generator
    ) -> tuple[Level, tuple[tuple[LevelObject, ...], tuple[LevelObject, ...]]]:
        """Уровень из пула и его «интересные» объекты, попутно обновляя пул.

        Пул растёт и обновляется постепенно, а не строится целиком на старте:
        генерация уровня стоит сотни миллисекунд, и десяток уровней сразу
        задержал бы начало обучения на несколько секунд в каждом worker'е.
        """
        pool = self._levels
        if len(pool) < self.cfg.level_pool and rng.random() < self.cfg.grow_prob:
            level = self._make_level(level_rng, len(pool))
            pool.append(level)
            self._focus.append(_focus_objects(level))
        elif len(pool) > 1 and rng.random() < self.cfg.refresh_prob:
            slot = int(rng.integers(0, len(pool)))
            level = self._make_level(level_rng, len(pool))
            pool[slot] = level
            self._focus[slot] = _focus_objects(level)
        index = int(rng.integers(0, len(pool)))
        return pool[index], self._focus[index]

    def _random_state(
        self,
        level: Level,
        focus: tuple[tuple[LevelObject, ...], tuple[LevelObject, ...]],
        rng: np.random.Generator,
    ) -> PlayerState:
        """Случайное, но достижимое состояние игрока на уровне.

        Зачем «достижимое»: если ставить игрока в произвольную точку, половина
        кадров окажется внутри блоков и в позах, которых в игре не бывает —
        сеть потратит ёмкость на бессмысленный домен. Поэтому состояние сначала
        сэмплится грубо, а потом несколько кадров физики приводят его в
        согласие с миром (приземление, падение, полёт корабля).

        Зачем наведение на объект (`focus_prob`): при равномерном x камера почти
        всегда смотрит в пустой коридор. Замер по готовому датасету: около 5
        пикселей HAZARD на кадр (0.05%) — сеть просто не получает положительных
        примеров и застревает на ответе «пусто + пол». Наведение поднимает
        плотность редких классов на порядок, не меняя ни разметку, ни рендер:
        меняется только то, КУДА смотрит камера.
        """
        length = max(2.0, float(level.length))
        hazards, others = focus
        target: LevelObject | None = None
        if (hazards or others) and rng.random() < self.cfg.focus_prob:
            take_hazard = bool(hazards) and (
                not others or rng.random() < self.cfg.focus_hazard_prob
            )
            pool = hazards if take_hazard else others
            target = pool[int(rng.integers(0, len(pool)))]
            lo, hi = self.cfg.focus_ahead
            x = float(target.x) - float(rng.uniform(lo, hi))
            x = float(np.clip(x, 0.5, length))
        else:
            x = float(rng.uniform(0.5, length))
        state = make_initial_state(level, x)

        roll = float(rng.random())
        if roll < self.cfg.wave_prob:
            state.mode = "wave"
        elif roll < self.cfg.wave_prob + self.cfg.ship_prob:
            state.mode = "ship"
        else:
            state.mode = "cube"
        if rng.random() < self.cfg.flip_gravity_prob:
            state.gravity = -int(state.gravity)
        state.speed_index = int(rng.integers(0, 5))

        ceiling = float(level.ceiling_y)
        if target is not None:
            # Камера по вертикали следует за игроком, поэтому «навести» её на
            # объект можно только через высоту самого игрока: слишком высоко —
            # и наземный шип уедет за нижний край кадра.
            state.y = float(target.y) + float(rng.uniform(-1.5, 3.5))
        elif rng.random() < 0.5:
            # У «пола» своей гравитации: самый частый режим игры.
            floor = GROUND_Y if state.gravity > 0 else ceiling
            offset = float(abs(rng.normal(0.6, 1.2))) + 0.5
            state.y = floor + offset * state.gravity
        else:
            state.y = float(rng.uniform(GROUND_Y + 0.5, ceiling - 0.5))
        state.y = float(np.clip(state.y, GROUND_Y + 0.3, ceiling - 0.3))
        state.vy = float(rng.normal(0.0, 7.0))
        state.on_ground = False

        # После наведения «доигрывание» делаем короче: за 24 кадра игрок уезжает
        # на 4 тайла вперёд и может утащить объект за край кадра.
        max_roll = int(self.cfg.rollout_steps) // (2 if target is not None else 1)
        steps = int(rng.integers(0, max_roll + 1))
        hold_p = float(rng.uniform(0.1, 0.7))
        for _ in range(steps):
            nxt, events = step_physics(state, level, bool(rng.random() < hold_p))
            if events["died"] or events["finished"]:
                break
            state = nxt
        return state

    def _pick_theme(
        self, rng: np.random.Generator, theme_rng: np.random.Generator
    ) -> Theme | None:
        """Встроенная тема для кадра или None, если кадр рисуется случайной.

        None означает «оставить ту случайную тему, которую поставил
        `Renderer.randomize`»: случайные темы всегда берутся из ОТДЕЛЬНОГО
        генератора (`theme_rng`), и у валидации это гарантирует поток
        оформлений, не пересекающийся с обучающим ни при каком совпадении
        базового зерна.

        Встроенная тема отдаётся не «как есть», а с перетасованным зерном узора
        (и в обучении — иногда обесцвеченная или инвертированная). Иначе на
        десять встроенных тем приходилось бы ровно десять фонов, и сеть могла бы
        опознавать тему по расположению звёзд, а не разбирать форму объектов.
        """
        if rng.random() < self.cfg.random_theme_prob:
            return None
        theme = self.themes[int(rng.integers(0, len(self.themes)))]
        if self.cfg.theme_seed_jitter:
            theme = theme.replace(seed=int(theme_rng.integers(0, 2**32)))
        if self.split == "train":
            if theme_rng.random() < self.cfg.theme_mono_prob:
                theme = to_monochrome(theme)
            if theme_rng.random() < self.cfg.theme_invert_prob:
                theme = to_inverted(theme)
        return theme

    def _make_sample(
        self,
        renderer: Renderer,
        rng: np.random.Generator,
        theme_rng: np.random.Generator,
        level_rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Одна пара (кадр uint8 (H,W,3), карта uint8 (H,W)).

        Порядок важен: сначала полностью определяется состояние мира, потом по
        нему СТРОЯТСЯ ОБА выхода. Кадр и карта берут камеру из одного и того же
        `state`, поэтому соответствуют друг другу пиксель в пиксель.
        """
        level, focus = self._pick_level(rng, level_rng)
        state = self._random_state(level, focus, rng)

        theme = self._pick_theme(rng, theme_rng)
        # `randomize` ставит случайную тему и заодно двигает раскладку
        # декораций и фазу партиклов; для встроенной темы после этого просто
        # кладём её палитру поверх — декор при этом всё равно остаётся новым.
        renderer.randomize(theme_rng)
        if theme is not None:
            renderer.set_theme(theme)
        if rng.random() < self.cfg.bare_prob:
            deco = float(rng.uniform(0.0, 0.4))
        else:
            lo_d, hi_d = self.cfg.deco_range
            deco = float(rng.uniform(float(lo_d), float(hi_d)))
        renderer.set_decoration_level(deco * float(self.cfg.decoration_level))

        frame = renderer.render(level, state, int(rng.integers(0, 100_000)))
        label = render_semantic(level, state, self.width, self.height)
        if self.augment:
            frame = augment_frame(frame, rng, self.augment_cfg)
        return frame, label

    # --- поток ---------------------------------------------------------------
    def _worker_split(self) -> tuple[int, int]:
        """(id worker'а, их общее число) — работает и в одиночном процессе."""
        info = get_worker_info()
        if info is None:
            return 0, 1
        return int(info.id), int(info.num_workers)

    def _ensure_renderer(self, worker_id: int) -> Renderer:
        """Один рендерер на worker: его кэши — самая дорогая часть генерации."""
        if self._renderer is None:
            self._renderer = Renderer(
                width=self.width,
                height=self.height,
                decoration_level=self.cfg.decoration_level,
                seed=_mix_seed(self.seed, worker_id, 17),
            )
        return self._renderer

    def raw_samples(
        self, count: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`count` пар в виде numpy-массивов (N,H,W,3) uint8 и (N,H,W) uint8.

        Зачем: для визуализации датасета, для тестов инвариантности и для
        быстрой оценки уже обученного зрения без DataLoader'а.
        """
        n = int(count)
        base = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(_mix_seed(base, _SEED_SAMPLES, 0))
        theme_rng = np.random.default_rng(
            _mix_seed(base, _SEED_THEMES if self.split == "train" else _SEED_VAL_THEMES, 0)
        )
        level_rng = np.random.default_rng(_mix_seed(base, _SEED_LEVELS, 5))
        renderer = self._ensure_renderer(0)
        frames = np.empty((n, self.height, self.width, 3), dtype=np.uint8)
        labels = np.empty((n, self.height, self.width), dtype=np.uint8)
        for i in range(n):
            frames[i], labels[i] = self._make_sample(renderer, rng, theme_rng, level_rng)
        return frames, labels

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Поток пар для DataLoader'а; при `samples=None` — бесконечный.

        Каждый worker получает СВОИ зёрна (через `_mix_seed`), поэтому потоки
        не дублируют друг друга; конечная выборка (валидация) делится между
        worker'ами поровну, чтобы суммарно получилось ровно `samples` пар.
        """
        worker_id, num_workers = self._worker_split()
        rng = np.random.default_rng(_mix_seed(self.seed, _SEED_SAMPLES, worker_id))
        theme_seed = _SEED_THEMES if self.split == "train" else _SEED_VAL_THEMES
        theme_rng = np.random.default_rng(_mix_seed(self.seed, theme_seed, worker_id))
        level_rng = np.random.default_rng(_mix_seed(self._level_seed, worker_id, 3))
        renderer = self._ensure_renderer(worker_id)

        if self.samples is None:
            total = -1
        else:
            # Остаток раздаём первым worker'ам, иначе выборка окажется короче.
            total = self.samples // num_workers + (
                1 if worker_id < self.samples % num_workers else 0
            )

        produced = 0
        while total < 0 or produced < total:
            frame, label = self._make_sample(renderer, rng, theme_rng, level_rng)
            yield frame_to_tensor(frame), torch.from_numpy(label.astype(np.int64))
            produced += 1

    def __len__(self) -> int:
        """Длина есть только у конечного (валидационного) потока."""
        if self.samples is None:
            raise TypeError("Бесконечный поток не имеет длины (samples=None)")
        return int(self.samples)


def frame_to_tensor(frame_uint8: np.ndarray) -> torch.Tensor:
    """Кадр (H,W,3) uint8 -> тензор (3,H,W) float32 в [0,1].

    Единая точка преобразования: обучение, оценка и `pipeline` обязаны кормить
    сеть одинаково нормированными данными, иначе обученные веса «не узнают»
    свой вход.
    """
    arr = np.asarray(frame_uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Ожидался кадр (H,W,3), получено {arr.shape}")
    data = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return torch.from_numpy(data).to(torch.float32).div_(255.0)


def frames_to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    """Пачка кадров (N,H,W,3) uint8 -> тензор (N,3,H,W) float32 в [0,1]."""
    arr = np.asarray(frames_uint8)
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise ValueError(f"Ожидались кадры (N,H,W,3), получено {arr.shape}")
    data = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    return torch.from_numpy(data).to(torch.float32).div_(255.0)


def _worker_init(worker_id: int) -> None:
    """Инициализация worker-процесса: по одному потоку BLAS на процесс.

    Зачем: и обучение, и генерация данных живут на одном CPU. Если каждый из
    четырёх worker'ов запустит ещё по четыре потока, они будут драться за ядра
    с шагом оптимизатора, и обучение станет МЕДЛЕННЕЕ, чем без worker'ов.
    """
    torch.set_num_threads(1)


def _default_workers() -> int:
    """Сколько worker-процессов брать по умолчанию.

    Переменная окружения `GDAI_NUM_WORKERS` перекрывает эвристику: на слабой
    машине (или в CI, где ядер два) лишние процессы только вредят.
    """
    env = os.environ.get("GDAI_NUM_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            _LOG.warning("GDAI_NUM_WORKERS=%r — не число, беру эвристику", env)
    cpus = os.cpu_count() or 1
    return int(min(4, max(0, cpus - 1)))


def make_loaders(
    cfg: PerceptionConfig,
    *,
    val_samples: int = VAL_SAMPLES,
    num_workers: int | None = None,
    seed: int = 0,
    data_cfg: DatasetConfig | None = None,
    levels_dir: str | os.PathLike[str] | None = "levels",
) -> tuple[DataLoader, DataLoader]:
    """Пара загрузчиков: бесконечный обучающий и конечный валидационный.

    Гарантии, ради которых функция существует (SPEC §10):

    * валидация идёт ТОЛЬКО на отложенных темах плюс случайные темы из
      отдельного потока — обучающий поток этих оформлений не видит никогда;
    * валидация не аугментируется: измеряем обобщение на новый ДИЗАЙН, а не
      устойчивость к шуму (её меряет отдельный тест);
    * валидационная выборка детерминирована (фиксированные зёрна), поэтому
      метрики двух прогонов сравнимы между собой.
    """
    workers = _default_workers() if num_workers is None else max(0, int(num_workers))
    val_workers = min(workers, 2)

    train_ds = SyntheticSegDataset(
        split="train",
        seed=seed,
        samples=None,
        augment=bool(cfg.augment),
        data_cfg=data_cfg,
        levels_dir=levels_dir,
    )
    val_ds = SyntheticSegDataset(
        split="val",
        seed=seed + 1,
        samples=int(val_samples),
        augment=False,
        data_cfg=data_cfg,
        levels_dir=levels_dir,
    )

    common = {
        "batch_size": int(cfg.batch_size),
        "pin_memory": False,
        "worker_init_fn": _worker_init,
    }
    train_loader = DataLoader(
        train_ds,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        num_workers=val_workers,
        persistent_workers=val_workers > 0,
        prefetch_factor=2 if val_workers > 0 else None,
        drop_last=False,
        **common,
    )
    _LOG.info(
        "загрузчики готовы: train-темы %s | held-out %s | worker'ов %d",
        ",".join(TRAIN_THEME_NAMES), ",".join(HELD_OUT_THEME_NAMES), workers,
    )
    return train_loader, val_loader


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Бесконечный итератор батчей: перезапускает загрузчик, если тот кончился.

    Зачем: обучение считает ШАГИ, а не эпохи (эпохи у бесконечного датасета
    нет). При `num_workers=0` и конечном потоке итератор всё равно однажды
    закончится — эта обёртка делает такой случай безболезненным.
    """
    while True:
        for batch in loader:
            yield batch


def class_pixel_counts(labels: Sequence[np.ndarray] | np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Сколько пикселей каждого класса в наборе карт.

    Зачем публично: веса классов в лоссе имеет смысл сверять с реальной
    статистикой датасета, а не подбирать вслепую.
    """
    arr = np.asarray(labels)
    return np.bincount(arr.ravel(), minlength=int(num_classes)).astype(np.int64)


__all__ = [
    "SyntheticSegDataset",
    "DatasetConfig",
    "DEFAULT_DATASET",
    "make_loaders",
    "infinite_batches",
    "frame_to_tensor",
    "frames_to_tensor",
    "train_themes",
    "held_out_themes",
    "check_theme_split",
    "class_pixel_counts",
    "TRAIN_THEME_NAMES",
    "HELD_OUT_THEME_NAMES",
    "VAL_SAMPLES",
]
