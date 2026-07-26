"""Среда Geometry Dash: gym-подобная обёртка над физикой и канонической картой.

Зачем этот модуль
-----------------
Политика обучается на **семантической карте**, а не на картинке, поэтому среда
обязана уметь работать в двух режимах и платить только за то, что реально
запрошено:

* `obs_mode="semantic"` — быстрый путь обучения RL. Здесь нет и не может быть
  pygame: одна среда из восьми делает миллионы шагов, и любой импорт графики
  превратился бы в постоянный налог на скорость. Цель — десятки тысяч шагов в
  секунду на одном ядре.
* `obs_mode="pixels"/"both"` — «честный» путь: тот же кадр, что увидит человек,
  для датасета зрения и для демонстраций. Рендерер (`gdai.env.render`)
  импортируется ЛЕНИВО, в момент первого обращения, и только тогда тянет pygame.

Practice-чекпойнты
------------------
Уровень длиной в сотню тайлов — это разреженная награда: агент, умирающий на
70-м тайле, тратит 95% опыта на уже выученное начало. Поэтому среда ведёт себя
как режим практики в оригинальной игре: пока эпизод идёт, она запоминает
**снимок состояния** игрока в момент прохождения каждого чекпойнта, а после
смерти с вероятностью `checkpoint_prob` начинает следующий эпизод прямо с
самого дальнего достигнутого снимка. Снимок хранит всё — режим, гравитацию,
скорость, скорость по вертикали, — поэтому восстановление честное, без
пересимуляции уровня с нуля.

Откуда берутся уровни
---------------------
`level_path` — фиксированный уровень из файла; иначе уровни генерируются
процедурно. Генерация одного уровня стоит сотни миллисекунд (внутри работает
проверка проходимости), а эпизод — единицы миллисекунд, поэтому генерировать
уровень на каждый reset нельзя: обучение уткнулось бы в генератор. Среда держит
небольшой **пул** уровней (`LEVEL_POOL_SIZE`): пул наполняется по одному уровню
за эпизод, дальше эпизоды случайно выбирают из пула. `set_difficulty` очищает
пул — новые уровни появятся на следующем `reset`.

Случайность
-----------
Три независимых потока (`np.random.Generator`), выведенных из одного seed:
уровни/чекпойнты, шум карты, темы рендера. Разделение нужно, чтобы включение
`semantic_noise` или смена темы не сдвигали последовательность генерации
уровней — иначе два прогона с одним seed были бы несравнимы.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from typing import Any

import numpy as np

from gdai.config import EnvConfig
from gdai.constants import (
    ACTION_HOLD,
    ACTION_NONE,
    GROUND_Y,
    MAX_FALL_V,
    NUM_ACTIONS,
    NUM_CLASSES,
    OBS_H,
    OBS_W,
    SPEEDS,
)
from gdai.env.generator import generate_level, make_checkpoints
from gdai.env.level import Level
from gdai.env.physics import (
    WAVE_START_HEIGHT,
    PlayerState,
    make_initial_state,
    player_half_extent,
    step_physics,
)
from gdai.env.semantic import render_semantic
from gdai.utils.logging import get_logger
from gdai.utils.seeding import make_rng, seed_from

_log = get_logger("env.gd_env")

# Размерность вектора признаков (SPEC §8): [vy_norm, on_ground, gravity,
# mode_is_cube, mode_is_ship, mode_is_wave, speed_norm, progress].
FEATURE_DIM: int = 8

OBS_MODES: tuple[str, ...] = ("semantic", "pixels", "both")

# Сколько процедурных уровней среда держит наготове. Зачем пул: генерация с
# проверкой проходимости стоит ~0.5 с, а эпизод — миллисекунды; без пула 99%
# времени обучения ушло бы в генератор. Восьми уровней хватает, чтобы агент не
# заучил один конкретный, и мало, чтобы прогрев был незаметен.
LEVEL_POOL_SIZE: int = 8

# Награда за прогресс делится на это число (SPEC §8): dx за кадр — это доли
# тайла, и без масштаба сумма за уровень получилась бы в сотнях, полностью
# заглушив штраф за смерть.
REWARD_PROGRESS_SCALE: float = 10.0

# Насколько далеко от запрошенного `start_x` снимок ещё считается «тем самым».
# Дальше этого расстояния из снимка берутся только режим/гравитация/скорость,
# а игрок ставится на пол в нужной точке.
SNAPSHOT_X_TOLERANCE: float = 1.0


def _clamp01(value: float) -> float:
    """Загнать число в [0, 1] — сложность приходит из учебного плана и извне."""
    v = float(value)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class GeometryDashEnv:
    """Среда одного игрока: `reset`/`step` в стиле Gymnasium, но без зависимости от него.

    Зачем свой интерфейс, а не gym: единственное, что нужно от гима, — это
    протокол `(obs, reward, terminated, truncated, info)`, а тащить ради него
    зависимость (и её версии пространств) в проект, который считает миллионы
    кадров, невыгодно. Совместимость по сигнатурам сохранена, обернуть в
    `gymnasium.Env` при желании можно тремя строками.

    Наблюдение — словарь:
    * `"semantic"` — uint8 (OBS_H, OBS_W), классы 0..9 (при obs_mode
      "semantic"/"both");
    * `"pixels"` — uint8 (OBS_H, OBS_W, 3), «красивый» кадр (при "pixels"/"both");
    * `"features"` — float32 (FEATURE_DIM,), всегда: то, что по одной картинке
      не восстановить (вертикальная скорость, режим, гравитация, прогресс).
    """

    metadata: dict[str, list[str]] = {"render_modes": ["rgb_array", "human"]}
    action_space_n: int = NUM_ACTIONS

    def __init__(self, config: EnvConfig | None = None, renderer: Any = None) -> None:
        """Создать среду. `renderer` — готовый `gdai.env.render.Renderer` (или совместимый).

        Зачем внешний рендерер: в датасете зрения и в окне визуализации один
        рендерер обслуживает несколько сред, а его создание тянет pygame и
        поверхности — платить за это в каждой среде незачем.
        """
        self.config: EnvConfig = config if config is not None else EnvConfig()
        if self.config.obs_mode not in OBS_MODES:
            raise ValueError(
                f"obs_mode={self.config.obs_mode!r} неизвестен, допустимы {OBS_MODES}"
            )

        self._needs_semantic: bool = self.config.obs_mode in ("semantic", "both")
        self._needs_pixels: bool = self.config.obs_mode in ("pixels", "both")

        # Пул уровней настраивается на экземпляре: сигнатура __init__ зафиксирована
        # контрактом, а разным сценариям (обучение, датасет, тест) нужен разный
        # компромисс «разнообразие против прогрева».
        self.level_pool_size: int = LEVEL_POOL_SIZE

        self._difficulty: float = _clamp01(self.config.difficulty)
        self._renderer: Any = renderer
        self._closed: bool = False

        self._seed: int | None = self.config.seed
        self._init_rngs(self.config.seed)

        # Источники уровня, в порядке приоритета: переданный в reset ->
        # загруженный из файла -> процедурный пул.
        self._fixed_level: Level | None = None
        if self.config.level_path:
            self._fixed_level = Level.load(self.config.level_path)
        self._pinned_level: Level | None = None
        self._pool: list[Level] = []
        self._pool_dirty: bool = False
        self._generated: int = 0

        self._level: Level | None = None
        self._state: PlayerState | None = None
        self._checkpoints: list[float] = []

        # Практика: снимки состояния на чекпойнтах текущего уровня.
        self._snapshots: dict[int, PlayerState] = {}
        self._next_cp: int = 0            # курсор «следующий непройденный чекпойнт»
        self._episode_cp: int | None = None   # самый дальний, пройденный в этом эпизоде
        self._pending_cp: int | None = None   # откуда практиковаться на следующем reset

        self._steps: int = 0
        self._t: int = 0                  # номер кадра для анимаций рендера
        self._episode_return: float = 0.0
        self._started_from_checkpoint: bool = False
        self._done: bool = True           # до первого reset шагать нельзя

    # -- внутреннее: случайность ------------------------------------------
    def _init_rngs(self, seed: int | None) -> None:
        """Развести три независимых потока случайности от одного seed.

        Зачем врозь: иначе включение `semantic_noise` меняло бы уровни, а смена
        темы — траекторию обучения, и воспроизводимость превратилась бы в
        фикцию.
        """
        self._rng: np.random.Generator = make_rng(seed)
        self._noise_rng: np.random.Generator = make_rng(
            seed_from("gd_env.noise", seed)
        )
        self._theme_rng: np.random.Generator = make_rng(
            seed_from("gd_env.theme", seed)
        )

    # -- публичные свойства -------------------------------------------------
    @property
    def level(self) -> Level:
        """Текущий уровень (после `reset`)."""
        if self._level is None:
            raise RuntimeError("Уровня ещё нет — сначала вызовите reset()")
        return self._level

    @property
    def state(self) -> PlayerState:
        """Текущее состояние игрока (после `reset`)."""
        if self._state is None:
            raise RuntimeError("Состояния ещё нет — сначала вызовите reset()")
        return self._state

    @property
    def difficulty(self) -> float:
        """Текущая сложность генерации (0..1)."""
        return self._difficulty

    @property
    def steps(self) -> int:
        """Сколько шагов сделано в текущем эпизоде."""
        return self._steps

    @property
    def checkpoints(self) -> list[float]:
        """x-координаты practice-чекпойнтов текущего уровня."""
        return list(self._checkpoints)

    @property
    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        """Формы всех ключей наблюдения — чтобы сети не угадывали размеры."""
        shapes: dict[str, tuple[int, ...]] = {}
        if self._needs_semantic:
            shapes["semantic"] = (OBS_H, OBS_W)
        if self._needs_pixels:
            shapes["pixels"] = (OBS_H, OBS_W, 3)
        shapes["features"] = (FEATURE_DIM,)
        return shapes

    # -- жизненный цикл эпизода --------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        level: Level | None = None,
        start_x: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Начать новый эпизод и вернуть `(obs, info)`.

        * `seed` — полный сброс случайности: пул уровней и накопленная практика
          выбрасываются, чтобы прогон стал чистой функцией seed (иначе «тот же
          seed» давал бы разное в зависимости от предыстории).
        * `level` — играть на конкретном уровне. Уровень **закрепляется** за
          средой до следующего явного `level` или `set_difficulty`: практика по
          чекпойнтам возможна только тогда, когда уровень между эпизодами не
          меняется.
        * `start_x` — стартовать не с начала. Если для этой точки есть снимок
          состояния (пройденный чекпойнт), он восстанавливается целиком; если
          снимок есть, но далеко позади, из него берутся режим/гравитация/
          скорость, а игрок ставится на пол в запрошенной точке; если снимков
          нет — обычный старт, перенесённый в `start_x`.
        """
        if self._closed:
            raise RuntimeError("Среда закрыта (close()) — reset невозможен")

        if seed is not None:
            self._seed = int(seed)
            self._init_rngs(self._seed)
            self._pool.clear()
            self._pool_dirty = False
            self._generated = 0
            self._forget_practice()

        # Решение о практике принимается до выбора уровня: рестарт с чекпойнта
        # обязан идти на ТОМ ЖЕ уровне, иначе снимок бессмыслен.
        use_checkpoint = (
            level is None
            and start_x is None
            and self.config.practice_checkpoints
            and self._level is not None
            and self._pending_cp is not None
            and self._pending_cp in self._snapshots
            and float(self._rng.random()) < float(self.config.checkpoint_prob)
        )

        if use_checkpoint:
            assert self._level is not None and self._pending_cp is not None
            snapshot = self._snapshots[self._pending_cp]
            # hold_prev сбрасываем: новый эпизод начинается с отпущенной кнопки,
            # иначе кольцо под чекпойнтом не сработало бы по фронту нажатия.
            self._state = replace(
                snapshot, alive=True, finished=False, hold_prev=False
            )
            self._next_cp = self._pending_cp + 1
            self._episode_cp = self._pending_cp
            self._started_from_checkpoint = True
        else:
            chosen = self._select_level(level)
            if chosen is not self._level:
                self._install_level(chosen)
            self._started_from_checkpoint = False
            if start_x is None:
                self._state = make_initial_state(self._level)
                self._next_cp = 0
                self._episode_cp = None
            else:
                x = min(max(float(start_x), 0.0), float(self.level.length))
                self._state = self._state_at(self.level, x)
                self._sync_checkpoint_cursor(self._state.x)

        self._steps = 0
        self._t = 0
        self._episode_return = 0.0
        self._done = False

        if self._needs_pixels and self.config.randomize_theme:
            # Новая тема на эпизод — та самая доменная рандомизация, ради
            # которой зрение вообще обучается отдельно от политики.
            self._randomize_theme()

        obs = self._make_obs()
        info = self._make_info(died=False, finished=False)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Один кадр игры: `(obs, reward, terminated, truncated, info)`.

        Награда (SPEC §8): `reward_progress * dx / 10` за пройденные за кадр
        тайлы, плюс `reward_alive`, плюс `reward_death` при смерти и
        `reward_finish` на финише. Прогресс — плотный сигнал, без него агент
        не находит первую награду за десятки миллионов кадров.
        """
        if self._closed:
            raise RuntimeError("Среда закрыта (close()) — step невозможен")
        if self._state is None or self._level is None:
            raise RuntimeError("Сначала вызовите reset()")
        if self._done:
            raise RuntimeError(
                "Эпизод уже завершён (terminated/truncated) — нужен reset()"
            )

        act = int(action)
        if act not in (ACTION_NONE, ACTION_HOLD):
            raise ValueError(
                f"Действие {act} вне диапазона 0..{NUM_ACTIONS - 1} "
                f"({ACTION_NONE} — ничего, {ACTION_HOLD} — держать)"
            )

        level = self._level
        prev_x = self._state.x
        state, events = step_physics(self._state, level, act == ACTION_HOLD)
        self._state = state
        self._steps += 1
        self._t += 1

        died = bool(events["died"])
        finished = bool(events["finished"])

        cfg = self.config
        reward = cfg.reward_progress * (state.x - prev_x) / REWARD_PROGRESS_SCALE
        reward += cfg.reward_alive
        if died:
            reward += cfg.reward_death
        if finished:
            reward += cfg.reward_finish
        self._episode_return += reward

        if cfg.practice_checkpoints and not died:
            self._capture_checkpoints(state)

        terminated = died or finished
        truncated = (not terminated) and self._steps >= int(cfg.max_steps)
        self._done = terminated or truncated

        if died and cfg.practice_checkpoints and self._episode_cp is not None:
            # Практикуемся с самого дальнего места, до которого агент вообще
            # добирался на этом уровне: смысл режима — не переигрывать начало.
            best = self._episode_cp
            if self._pending_cp is None or best > self._pending_cp:
                self._pending_cp = best
        if finished:
            # Уровень пройден целиком — практиковать больше нечего.
            self._pending_cp = None

        obs = self._make_obs()
        info = self._make_info(died=died, finished=finished)
        return obs, float(reward), terminated, truncated, info

    # -- рендер и настройки -------------------------------------------------
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Всегда «красивый» кадр (H, W, 3) uint8 — то, что увидел бы человек.

        Оба режима возвращают массив: настоящее окно — задача `gdai.viz.viewer`,
        среда не должна открывать окна во время обучения.
        """
        if mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"mode={mode!r} неизвестен, допустимы {self.metadata['render_modes']}"
            )
        if self._state is None or self._level is None:
            raise RuntimeError("Нечего рисовать — сначала вызовите reset()")
        return self._get_renderer().render(self._level, self._state, self._t)

    def set_difficulty(self, d: float) -> None:
        """Сменить сложность процедурной генерации (учебный план зовёт это отсюда).

        Уровни не перегенерируются немедленно: текущий эпизод доигрывается на
        старом уровне, а пул очищается — новые уровни появятся на следующем
        `reset`. Если уровень задан файлом или закреплён явно, сложность влияет
        только на отчётность в `info`.
        """
        new_d = _clamp01(d)
        if new_d == self._difficulty:
            return
        self._difficulty = new_d
        self._pool_dirty = True

    def close(self) -> None:
        """Освободить рендерер (pygame-поверхности) — повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        self._done = True
        closer = getattr(self._renderer, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # pragma: no cover - зависит от рендерера
                _log.debug("рендерер не закрылся штатно: %s", exc)
        self._renderer = None

    def __enter__(self) -> "GeometryDashEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- выбор уровня -------------------------------------------------------
    def _select_level(self, level: Level | None) -> Level:
        """Какой уровень играть в этом эпизоде.

        Приоритет: явно переданный -> закреплённый ранее -> из файла ->
        процедурный пул (пока не наполнен, каждый эпизод добавляет один уровень).
        """
        if level is not None:
            self._pinned_level = level
            return level
        if self._pinned_level is not None:
            return self._pinned_level
        if self._fixed_level is not None:
            return self._fixed_level

        if self._pool_dirty:
            self._pool.clear()
            self._pool_dirty = False
        if len(self._pool) < max(1, int(self.level_pool_size)):
            name = f"proc-d{self._difficulty:.2f}-{self._generated:03d}"
            self._generated += 1
            new_level = generate_level(self._difficulty, self._rng, name=name)
            self._pool.append(new_level)
            return new_level
        idx = int(self._rng.integers(len(self._pool)))
        return self._pool[idx]

    def _install_level(self, level: Level) -> None:
        """Сделать уровень текущим: пересчитать чекпойнты и забыть чужую практику."""
        self._level = level
        checkpoints = [float(c) for c in level.checkpoints]
        if not checkpoints and self.config.practice_checkpoints:
            # Уровень из файла может не иметь чекпойнтов — считаем их сами и
            # держим у себя, чтобы не мутировать чужой объект уровня.
            checkpoints = make_checkpoints(level)
        checkpoints.sort()
        self._checkpoints = checkpoints
        self._forget_practice()

    def _forget_practice(self) -> None:
        """Сбросить снимки и курсоры практики (уровень сменился или новый seed)."""
        self._snapshots.clear()
        self._next_cp = 0
        self._episode_cp = None
        self._pending_cp = None

    # -- практика -----------------------------------------------------------
    def _capture_checkpoints(self, state: PlayerState) -> None:
        """Снять состояние на каждом только что пройденном чекпойнте.

        Зачем снимок, а не пересимуляция: восстановить режим/гравитацию/скорость
        для произвольного x иначе можно только прокруткой уровня с начала —
        это десятки тысяч кадров физики на каждый reset.
        """
        cps = self._checkpoints
        x = state.x
        n = len(cps)
        while self._next_cp < n and x >= cps[self._next_cp]:
            self._snapshots[self._next_cp] = replace(state)
            self._episode_cp = self._next_cp
            self._next_cp += 1

    def _sync_checkpoint_cursor(self, x: float) -> None:
        """Поставить курсор чекпойнтов в позицию, соответствующую координате x."""
        idx = bisect_right(self._checkpoints, float(x))
        self._next_cp = idx
        self._episode_cp = idx - 1 if idx > 0 else None

    def _state_at(self, level: Level, start_x: float) -> PlayerState:
        """Состояние игрока для старта с произвольного x (см. `reset`)."""
        best_idx: int | None = None
        best_x = -1.0
        for idx, snap in self._snapshots.items():
            if snap.x <= start_x + SNAPSHOT_X_TOLERANCE and snap.x > best_x:
                best_idx, best_x = idx, snap.x
        if best_idx is None:
            return make_initial_state(level, start_x)
        snap = self._snapshots[best_idx]
        if abs(snap.x - start_x) <= SNAPSHOT_X_TOLERANCE:
            return replace(snap, alive=True, finished=False, hold_prev=False)
        return self._ground_state(
            level, start_x, snap.mode, int(snap.gravity), int(snap.speed_index)
        )

    @staticmethod
    def _ground_state(
        level: Level, x: float, mode: str, gravity: int, speed_index: int
    ) -> PlayerState:
        """Игрок стоит на «своём» полу в точке x с заданными режимом и гравитацией.

        Повторяет логику `make_initial_state`, но без привязки к стартовым
        настройкам уровня: после порталов режим и гравитация уже другие, и
        ставить игрока «как на старте» значило бы уронить его в потолок.
        """
        _, half_y = player_half_extent(mode)
        floor_y = GROUND_Y if gravity > 0 else float(level.ceiling_y)
        offset = WAVE_START_HEIGHT if mode == "wave" else half_y
        return PlayerState(
            x=float(x),
            y=floor_y + offset * gravity,
            vy=0.0,
            mode=mode,
            gravity=gravity,
            speed_index=speed_index,
            on_ground=(mode != "wave"),
            alive=True,
            finished=False,
            hold_prev=False,
        )

    # -- наблюдения ---------------------------------------------------------
    def _make_obs(self) -> dict[str, np.ndarray]:
        """Собрать словарь наблюдения ровно из того, что требует obs_mode."""
        assert self._level is not None and self._state is not None
        obs: dict[str, np.ndarray] = {}
        if self._needs_semantic:
            sem = render_semantic(self._level, self._state, OBS_W, OBS_H)
            noise = float(self.config.semantic_noise)
            if noise > 0.0:
                sem = self._corrupt(sem, noise)
            obs["semantic"] = sem
        if self._needs_pixels:
            frame = self._get_renderer().render(self._level, self._state, self._t)
            obs["pixels"] = frame
        obs["features"] = self._features()
        return obs

    def _corrupt(self, sem: np.ndarray, prob: float) -> np.ndarray:
        """Испортить долю пикселей карты случайными классами.

        Зачем: в бою карту рисует не растеризатор, а U-Net, и она ошибается.
        Политика, обученная на идеальной разметке, разваливается от первого же
        ложного «шипа»; шум при обучении делает её терпимой к таким ошибкам.
        """
        mask = self._noise_rng.random(sem.shape) < prob
        count = int(np.count_nonzero(mask))
        if count:
            sem[mask] = self._noise_rng.integers(
                0, NUM_CLASSES, size=count, dtype=np.uint8
            )
        return sem

    def _features(self) -> np.ndarray:
        """Вектор `FEATURE_DIM` — то, чего не видно на одной карте.

        Вертикальная скорость, режим, направление гравитации и прогресс не
        читаются из статичного кадра, а решают всё: одна и та же картинка на
        взлёте и на падении требует противоположных действий.
        """
        state = self._state
        assert state is not None
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        vy_norm = state.vy / MAX_FALL_V
        feat[0] = min(max(vy_norm, -1.0), 1.0)
        feat[1] = 1.0 if state.on_ground else 0.0
        feat[2] = float(state.gravity)
        feat[3] = 1.0 if state.mode == "cube" else 0.0
        feat[4] = 1.0 if state.mode == "ship" else 0.0
        feat[5] = 1.0 if state.mode == "wave" else 0.0
        feat[6] = float(state.speed_index) / float(len(SPEEDS) - 1)
        feat[7] = self._progress()
        return feat

    def _progress(self) -> float:
        """Доля уровня, пройденная игроком (0..1)."""
        assert self._level is not None and self._state is not None
        length = float(self._level.length)
        if length <= 0.0:
            return 1.0
        return min(max(self._state.x / length, 0.0), 1.0)

    def _make_info(self, died: bool, finished: bool) -> dict[str, Any]:
        """Служебная информация шага (SPEC §8) плюс данные для логов обучения."""
        assert self._level is not None and self._state is not None
        return {
            "x": float(self._state.x),
            "progress": self._progress(),
            "died": bool(died),
            "finished": bool(finished),
            "level_name": self._level.name,
            "difficulty": float(self._difficulty),
            "steps": int(self._steps),
            "from_checkpoint": bool(self._started_from_checkpoint),
            "episode_return": float(self._episode_return),
        }

    # -- рендерер -----------------------------------------------------------
    def _get_renderer(self) -> Any:
        """Достать (и при необходимости создать) «красивый» рендерер.

        Импорт ленивый: путь `obs_mode="semantic"` обязан работать вообще без
        pygame — в обучении RL графика не нужна, а её инициализация стоит
        дороже тысяч шагов физики.
        """
        if self._closed:
            raise RuntimeError("Среда закрыта (close()) — рендер невозможен")
        if self._renderer is not None:
            return self._renderer
        try:
            from gdai.env.render import Renderer
        except ImportError as exc:
            raise ImportError(
                "Красивый кадр требует модуль gdai.env.render (и pygame): "
                f"{exc}. Для обучения политики он не нужен — используйте "
                "obs_mode='semantic', иначе установите pygame и убедитесь, что "
                "gdai/env/render.py доступен."
            ) from exc
        self._renderer = Renderer(
            width=OBS_W,
            height=OBS_H,
            decoration_level=float(self.config.decoration_level),
            seed=seed_from("gd_env.renderer", self._seed),
        )
        if self.config.randomize_theme:
            self._randomize_theme()
        return self._renderer

    def _randomize_theme(self) -> None:
        """Выдать рендереру новую тему и новые декорации на эпизод."""
        renderer = self._renderer
        if renderer is None:
            # Рендерера ещё нет: тема выберется при его создании — тянуть сюда
            # pygame ради одной случайной темы незачем.
            return
        randomize = getattr(renderer, "randomize", None)
        if callable(randomize):
            randomize(self._theme_rng)


__all__ = [
    "GeometryDashEnv",
    "FEATURE_DIM",
    "OBS_MODES",
    "LEVEL_POOL_SIZE",
    "REWARD_PROGRESS_SCALE",
    "SNAPSHOT_X_TOLERANCE",
]
