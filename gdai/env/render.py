"""«Красивый» рендер уровня: то, что видит человек (и глаз нейросети).

Зачем этот модуль
-----------------
Это вторая половина сердца проекта. `semantic.py` отвечает на вопрос «что тут
на самом деле», а `render.py` — на вопрос «как это может выглядеть». Чем шире
разброс ответов на второй вопрос при неизменном первом, тем более устойчивым
получается зрение: агент физически не может выучить палитру, если один и тот же
уровень показывают ему то неоновым, то монохромным, то залитым декорациями.

Три жёстких инварианта
----------------------
1. **Общая камера.** Кадр строится от `semantic.camera_origin(state)` — той же
   функции, что и разметка, — и по тем же формулам перевода мира в пиксели.
   Пиксель кадра и пиксель карты соответствуют друг другу; ни одна декорация не
   имеет права сдвинуть игровой объект даже на полпикселя.
2. **Декорации не игровые.** Всё, что рисуется в слое декора (полосы, трубы,
   шестерёнки, глифы, плавающие блоки, свечения), не существует ни для физики,
   ни для карты. Сеть обязана научиться их игнорировать — это и есть главный
   источник сложности задачи сегментации.
3. **Тряска по умолчанию выключена.** Сдвинуть кадр, не сдвинув разметку,
   значит систематически врать зрению. `Theme.shake` существует, но встроенные
   и случайные темы держат его в нуле; фактический сдвиг доступен в
   `Renderer.last_shake`, чтобы вызывающий мог применить его и к карте.

Порядок слоёв (SPEC §9)
-----------------------
1) фоновый градиент + узор + пульсация;
2) 0..3 слоя параллакса (движутся медленнее камеры);
3) НЕИГРОВЫЕ декорации, привязанные к мировым координатам;
4) игровые объекты (пол, потолок, блоки, шипы, кольца, пады, порталы, финиш);
5) игрок, след и партиклы;
6) пост-эффекты: bloom, виньетка, шум, гамма/контраст, хроматическая аберрация.

Производительность
------------------
Рендер вызывается на каждом сэмпле датасета зрения, поэтому:
* всё, что зависит только от темы (фон, параллакс, спрайты объектов, LUT,
  маска виньетки), считается один раз в `set_theme` и потом только блитится;
* декорации кэшируются по «ячейкам мира» и переживают смену темы;
* пост-эффекты — чистый numpy на массиве 72x128x3 без питоновских циклов.
Ориентир — заметно больше 200 кадров/с на одном ядре CPU при 128x72.
"""

from __future__ import annotations

import math
import os
import sys
import zlib
from typing import Any

import numpy as np

from gdai.constants import (
    GOAL,
    GROUND_Y,
    HAZARD,
    OBS_H,
    OBS_W,
    ORB,
    PAD,
    PORTAL_GRAVITY,
    PORTAL_MODE,
    PORTAL_SPEED,
    PX_PER_TILE,
    SOLID,
)
from gdai.env.level import Level, OBJECT_TYPES, type_semantic_class
from gdai.env.physics import PlayerState, player_half_extent
from gdai.env.semantic import (
    RING_INNER_RATIO,
    SHAPE_BY_TYPE,
    camera_origin,
    class_priority,
)
from gdai.env.themes import (
    BUILTIN_THEMES,
    RGB,
    Theme,
    luminance,
    mix_rgb,
    random_theme,
    theme_by_name,
)
from gdai.utils.logging import get_logger

_LOG = get_logger("env.render")


# ---------------------------------------------------------------------------
# headless-инициализация SDL до импорта pygame
# ---------------------------------------------------------------------------
def _prepare_sdl() -> None:
    """Включить dummy-драйвер SDL, если дисплея нет.

    Зачем строго до `import pygame`: SDL читает переменные окружения в момент
    инициализации, и выставленный позже `SDL_VIDEODRIVER` уже ничего не меняет.
    Без этого генерация датасета на сервере без X падает на попытке открыть окно.
    """
    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform in ("win32", "darwin")
    )
    if not has_display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # Приветственный баннер pygame — это print в stdout библиотекой, что
    # запрещено правилами проекта; глушим его тем же способом, что и сам pygame.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


_prepare_sdl()

try:
    import pygame
except ImportError as exc:  # pragma: no cover - зависит от окружения
    raise ImportError(
        "Для рендера нужен pygame. Установите его: pip install pygame>=2.1"
    ) from exc


# ---------------------------------------------------------------------------
# константы модуля
# ---------------------------------------------------------------------------
_P: float = float(PX_PER_TILE)

# Вертикальный запас у фоновых слоёв (px). Нужен, чтобы параллакс мог уехать
# вверх-вниз вслед за камерой, не обнажив край текстуры.
_VPAD: int = 32

# Ширина ячейки мира, в которой генерируются декорации (тайлы). Ячейка — способ
# сделать декор ДЕТЕРМИНИРОВАННОЙ функцией мировых координат: объект, однажды
# появившийся на x=42, будет там всегда, а не «плыть» вслед за камерой.
_DECO_CELL: float = 6.0
# Сколько кандидатов на ячейку хранит кэш. Реально рисуется подмножество: у
# каждого элемента есть «ранг», и порог отбора зависит от `decoration_level` и
# от плотности темы. Запас нужен, чтобы плотная тема могла завалить кадр мусором.
_DECO_MAX_PER_CELL: int = 11

# Виды декораций (все — неигровые). Числа, а не строки: диспетчер вызывается
# десятки раз на кадр.
(
    _D_BAR, _D_PIPE, _D_GEAR, _D_GLYPH, _D_FLOATER, _D_GLOW,
    _D_CHEVRON, _D_PATCH, _D_LATTICE, _D_WAVE, _D_DIAMOND, _D_DASH,
) = range(12)

_D_ALL: tuple[int, ...] = tuple(range(12))

# Потолок непрозрачности декора. Полностью непрозрачная декорация — это уже не
# декорация: кольцо-шестерёнка нужного размера стала бы неотличима от игрового
# кольца, и задача сегментации перестала бы иметь решение. Остаточная
# прозрачность — тот самый признак «это мусор», который сеть обязана выучить.
_DECOR_ALPHA_CAP: int = 205

# Какие виды декора разрешает каждый стиль темы. Ключ — Theme.decor_style.
_DECOR_KINDS: dict[str, tuple[int, ...]] = {
    "bars": (_D_BAR, _D_PATCH, _D_DASH),
    "pipes": (_D_PIPE, _D_BAR, _D_LATTICE),
    "gears": (_D_GEAR, _D_GLOW, _D_LATTICE),
    "glyphs": (_D_GLYPH, _D_CHEVRON, _D_PATCH, _D_DASH),
    "floaters": (_D_FLOATER, _D_GLOW, _D_DIAMOND),
    "lattice": (_D_LATTICE, _D_PATCH, _D_BAR, _D_DASH),
    "waves": (_D_WAVE, _D_GLOW, _D_DASH, _D_CHEVRON),
    "crystals": (_D_DIAMOND, _D_FLOATER, _D_GLOW, _D_WAVE),
    "mixed": _D_ALL,
}

# Период «биения» яркости в кадрах (примерно 0.8 с — темп типичного трека GD).
_PULSE_PERIOD: int = 48
# Ниже этой яркости пиксель не участвует в bloom.
_BLOOM_THRESHOLD: float = 168.0
# Максимум, который bloom имеет право добавить к пикселю. Без потолка светлая
# тема с сильным свечением превращает весь кадр в белое пятно — такой сэмпл
# бесполезен для обучения зрения, потому что в нём не осталось информации.
_BLOOM_CAP: float = 96.0
# Размер таблицы предрасчитанных случайных чисел для партиклов и следа
# (степень двойки: индекс берётся побитовой маской).
_RAND_TAB: int = 4096

# Поля шума для пост-эффекта, общие для всех рендереров одного размера кадра.
# Зачем кэш: генерация 72x256x3 гауссиан стоит около миллисекунды, а зависит
# только от размера — пересоздавать её на каждую смену темы расточительно.
_NOISE_CACHE: dict[tuple[int, int], np.ndarray] = {}
# Базовая радиальная маска виньетки — тоже зависит только от размера кадра.
_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}

# Класс объекта по типу — считаем один раз, чтобы не звать метод на каждый
# объект каждого кадра.
_CLASS_BY_TYPE: dict[str, int] = {t: type_semantic_class(t) for t in OBJECT_TYPES}
# Порядок отрисовки берём из семантики: кадр обязан согласовываться с разметкой
# и в том, ЧТО поверх чего лежит, иначе зрение получит противоречивую цель.
_PRIORITY_BY_TYPE: dict[str, int] = {
    t: class_priority(_CLASS_BY_TYPE[t]) for t in OBJECT_TYPES
}
_PORTAL_KIND: dict[int, str] = {
    PORTAL_GRAVITY: "gravity",
    PORTAL_MODE: "mode",
    PORTAL_SPEED: "speed",
}


# ---------------------------------------------------------------------------
# вспомогательная арифметика
# ---------------------------------------------------------------------------
def _stable_seed(*parts: Any) -> int:
    """Детерминированный 32-битный seed из произвольных частей.

    Зачем не `hash()`: встроенный хэш строк рандомизируется между запусками
    (PYTHONHASHSEED), и узоры темы менялись бы от запуска к запуску — прощай
    воспроизводимость датасета. CRC32 стабилен всегда и везде.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(payload)) & 0xFFFFFFFF


def _span_x(x0: float, x1: float, cam_x: float) -> tuple[int, int]:
    """Столбцы [c0, c1) для мирового отрезка [x0, x1].

    Формула ДОСЛОВНО повторяет `semantic._col_span`: пиксель принадлежит фигуре,
    если внутрь попал его центр. Любое расхождение здесь — систематический сдвиг
    кадра относительно разметки, то есть тихая порча всего обучения зрения.
    """
    c0 = math.ceil((x0 - cam_x) * _P - 0.5)
    c1 = math.floor((x1 - cam_x) * _P - 0.5) + 1
    return c0, c1


def _span_y(y0: float, y1: float, cam_y: float, view_h: int) -> tuple[int, int]:
    """Строки [r0, r1) для мирового отрезка [y0, y1] (ось Y перевёрнута)."""
    top = float(view_h) - 0.5
    r0 = math.ceil(top - (y1 - cam_y) * _P)
    r1 = math.floor(top - (y0 - cam_y) * _P) + 1
    return r0, r1


def _sx(wx: float, cam_x: float) -> float:
    """Мировой x -> дробный индекс столбца (для декораций и партиклов)."""
    return (wx - cam_x) * _P - 0.5


def _sy(wy: float, cam_y: float, view_h: int) -> float:
    """Мировой y -> дробный индекс строки."""
    return (float(view_h) - 0.5) - (wy - cam_y) * _P


def _rgba(color: RGB, alpha: int) -> tuple[int, int, int, int]:
    """Цвет с альфой — декорации всегда полупрозрачны, это их опознавательный знак."""
    a = 0 if alpha < 0 else (255 if alpha > 255 else int(alpha))
    return (color[0], color[1], color[2], a)


def _blur3(a: np.ndarray) -> np.ndarray:
    """Разделимое размытие ядром 1-2-1 по обеим осям (края повторяются).

    Зачем своё, а не scipy/cv2: и то и другое — опциональные зависимости, а тут
    нужны буквально шесть арифметических операций над массивом 36x64x3.
    """
    out = a * 2.0
    out[:, 1:] += a[:, :-1]
    out[:, :-1] += a[:, 1:]
    out *= 0.25
    res = out * 2.0
    res[1:, :] += out[:-1, :]
    res[:-1, :] += out[1:, :]
    res *= 0.25
    return res


def _bloom_blur(bright: np.ndarray) -> np.ndarray:
    """Широкое размытие для свечения: понижаем разрешение вдвое, мажем, поднимаем.

    Зачем через downsample: радиус свечения нужен порядка 6-8 px, а прямое
    размытие такого радиуса стоило бы в разы дороже; на половинном разрешении
    двух проходов ядра 1-2-1 хватает, а качество для 128x72 избыточно.
    """
    h, w, c = bright.shape
    hh, ww = h // 2, w // 2
    if hh < 2 or ww < 2:
        return _blur3(bright)
    # Понижение разрешения сложением четырёх срезов: втрое дешевле, чем
    # reshape(...).mean(axis=(1, 3)) — а результат тот же.
    a = bright[: hh * 2, : ww * 2]
    small = a[0::2, 0::2]
    small = small + a[1::2, 0::2]
    small += a[0::2, 1::2]
    small += a[1::2, 1::2]
    small *= 0.25
    small = _blur3(_blur3(small))
    up = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    if up.shape[0] == h and up.shape[1] == w:
        return up
    out = np.zeros_like(bright)
    out[: up.shape[0], : up.shape[1]] = up
    return out


def _tri_points(shape: str, w: int, h: int) -> list[tuple[int, int]]:
    """Вершины треугольника шипа в спрайте (w, h) по его ориентации.

    Ориентация обязана совпадать с растеризацией карты: `spike` смотрит вверх,
    `spike_down` — вниз и так далее (`semantic.SHAPE_BY_TYPE`). Если картинка и
    разметка разойдутся по форме, сеть будет учить противоречие.
    """
    xm, ym = w - 1, h - 1
    if shape == "tri_down":
        return [(0, 0), (xm, 0), (w // 2, ym)]
    if shape == "tri_left":
        return [(0, h // 2), (xm, 0), (xm, ym)]
    if shape == "tri_right":
        return [(xm, h // 2), (0, 0), (0, ym)]
    return [(w // 2, 0), (xm, ym), (0, ym)]


def _shrink(points: list[tuple[int, int]], factor: float) -> list[tuple[float, float]]:
    """Сжать многоугольник к его центру тяжести (для стиля «double»)."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    """Рисует кадр уровня в выбранной теме. Один экземпляр — одна среда.

    Экземпляр держит тяжёлые кэши (фон, параллакс, спрайты), поэтому его
    создают один раз на среду и переиспользуют; `randomize` меняет тему и
    раскладку декора на новый эпизод.
    """

    def __init__(
        self,
        width: int = OBS_W,
        height: int = OBS_H,
        theme: Theme | None = None,
        decoration_level: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Размер кадра должен быть положительным, получено {self.width}x{self.height}"
            )
        self.decoration_level = float(min(max(float(decoration_level), 0.0), 1.0))
        self._seed = 0 if seed is None else int(seed) & 0xFFFFFFFF
        self._rng = np.random.default_rng(self._seed if seed is not None else None)
        # Зерно раскладки декораций: отдельно от темы, потому что декор привязан
        # к миру и не обязан меняться при смене палитры.
        self._deco_seed = int(self._rng.integers(0, 2**32))
        # Последняя применённая тряска: вызывающий может применить её к карте.
        self.last_shake: tuple[int, int] = (0, 0)

        self._surface = pygame.Surface((self.width, self.height))
        self._alpha = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._sprites: dict[tuple[Any, ...], pygame.Surface] = {}
        self._deco_cache: dict[int, tuple[tuple[Any, ...], ...]] = {}
        self._bg: pygame.Surface | None = None
        self._parallax: list[tuple[pygame.Surface, float]] = []
        self._vignette: np.ndarray | None = None
        self._lut: np.ndarray | None = None
        self._noise_field: np.ndarray | None = None
        self._rand: list[float] = []
        self._rand_off: int = 0

        self._theme: Theme = theme if theme is not None else BUILTIN_THEMES[0]
        self._build_rand_table()
        self.set_theme(self._theme)
        _LOG.debug(
            "Renderer %dx%d, тема %s, декор %.2f, SDL=%s",
            self.width, self.height, self._theme.name, self.decoration_level,
            os.environ.get("SDL_VIDEODRIVER", "default"),
        )

    # --- свойства -----------------------------------------------------------
    @property
    def theme(self) -> Theme:
        """Текущая тема (менять только через `set_theme`, иначе кэши устареют)."""
        return self._theme

    @property
    def size(self) -> tuple[int, int]:
        """Размер кадра (width, height) — тот же, что у семантической карты."""
        return (self.width, self.height)

    # --- публичное API ------------------------------------------------------
    def set_theme(self, theme: Theme) -> None:
        """Сменить тему и пересобрать всё, что от неё зависит.

        Дорогая операция (градиент, слои параллакса, спрайты) — её место в
        начале эпизода, а не внутри кадра. Раскладка декораций переживает смену
        темы: декор привязан к миру, меняется только его цвет.
        """
        if not isinstance(theme, Theme):
            raise TypeError(f"Ожидался Theme, получено {type(theme).__name__}")
        self._theme = theme
        self._sprites.clear()
        self._build_background()
        self._build_parallax()
        self._build_post()

    def set_decoration_level(self, level: float) -> None:
        """Плотность декора 0..1 без пересборки тяжёлых кэшей.

        Зачем отдельный метод: уровень декора крутят в интерактивном просмотре
        (клавиша D) и в учебном плане, а перестраивать ради этого фон незачем —
        декорации отбираются по «рангу», а не генерируются заново.

        Пост-эффекты всё же пересчитываются: сила виньетки зависит и от темы, и
        от уровня декора. Пока она считалась только в `set_theme`, кадр зависел
        от ПОРЯДКА вызовов (`set_theme` до или после `set_decoration_level`) и
        отличался до 51/255 на пиксель при одних и тех же параметрах — то есть
        `render` переставал быть функцией своих аргументов. Пересчёт стоит
        меньше десятка микросекунд (LUT на 256 значений и умножение маски).
        """
        self.decoration_level = float(min(max(float(level), 0.0), 1.0))
        self._build_post()

    def randomize(self, rng: np.random.Generator) -> None:
        """Новая случайная тема + новая раскладка декораций.

        Это точка входа доменной рандомизации: вызывается на каждом сэмпле
        датасета зрения и на каждом эпизоде среды с пиксельными наблюдениями.
        """
        self._deco_seed = int(rng.integers(0, 2**32))
        self._deco_cache.clear()
        # Таблицу случайных чисел не пересоздаём (это дорого) — достаточно
        # сдвинуть окно чтения, чтобы партиклы и след легли по-новому.
        self._rand_off = int(rng.integers(0, _RAND_TAB))
        self.set_theme(random_theme(rng))

    def render(self, level: Level, state: PlayerState, t: int) -> np.ndarray:
        """Кадр (H, W, 3) uint8 для данного состояния и номера кадра `t`.

        `t` управляет только анимацией (пульс, партиклы, вращение шестерёнок):
        геометрия игровых объектов от него не зависит, поэтому кадр остаётся
        согласованным с `render_semantic(level, state)` при любом `t`.
        """
        frame = int(t)
        cam = camera_origin(state)
        surf = self._surface

        self._draw_background(surf, cam, frame)
        self._draw_parallax(surf, cam)
        self._draw_decor(surf, cam, frame)
        self._draw_world(surf, level, cam, frame)
        self._draw_player(surf, state, cam, frame)

        arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
        return self._post(arr, frame)

    def close(self) -> None:
        """Освободить поверхности. pygame сам всё соберёт, но явное лучше неявного."""
        self._sprites.clear()
        self._deco_cache.clear()
        self._bg = None
        self._parallax = []

    # ------------------------------------------------------------------
    # подготовка кэшей
    # ------------------------------------------------------------------
    def _build_rand_table(self) -> None:
        """Таблица псевдослучайных чисел для партиклов и следа.

        Зачем таблица, а не Generator на кадр: создание PCG64 стоит десятки
        микросекунд, а нам нужно лишь «шумовое, но воспроизводимое» число по
        индексу (кадр, номер частицы). Таблица даёт то же самое за наносекунды.
        """
        rng = np.random.default_rng(_stable_seed(self._seed, "rand"))
        self._rand = rng.random(_RAND_TAB).tolist()

    def _r(self, index: int) -> float:
        """Псевдослучайное число в [0, 1) по индексу — детерминированно."""
        return self._rand[(index + self._rand_off) & (_RAND_TAB - 1)]

    def _build_background(self) -> None:
        """Собрать фон: вертикальный градиент + узор, тайлящийся по X.

        Поверхность делается двойной ширины и периодичной с периодом `width`:
        тогда скроллинг фона — это ОДИН блит куска `[sx, sx + width)`, без
        арифметики обрезки и без второго вызова.
        """
        th = self._theme
        W, H = self.width, self.height
        ph = H + 2 * _VPAD
        rng = np.random.default_rng(_stable_seed(th.name, th.seed, "bg"))

        # Градиент: сверху bg_top, снизу bg_bottom.
        ramp = np.linspace(0.0, 1.0, ph, dtype=np.float32)[:, None]
        top = np.asarray(th.bg_top, dtype=np.float32)[None, :]
        bottom = np.asarray(th.bg_bottom, dtype=np.float32)[None, :]
        col = top + (bottom - top) * ramp             # (ph, 3)
        arr = np.repeat(col[:, None, :], W, axis=1)   # (ph, W, 3)

        if th.bg_style == "noise":
            amp = 8.0 + 22.0 * th.pattern_scale
            arr += rng.normal(0.0, amp, size=(ph, W, 1)).astype(np.float32)

        np.clip(arr, 0.0, 255.0, out=arr)
        tile = pygame.surfarray.make_surface(
            np.ascontiguousarray(arr.transpose(1, 0, 2).astype(np.uint8))
        )
        self._draw_bg_pattern(tile, W, ph, rng)

        full = pygame.Surface((2 * W, ph))
        full.blit(tile, (0, 0))
        full.blit(tile, (W, 0))
        self._bg = full

    def _draw_bg_pattern(
        self, tile: pygame.Surface, W: int, ph: int, rng: np.random.Generator
    ) -> None:
        """Нарисовать фоновый узор так, чтобы он бесшовно тайлился по X.

        Все узоры строятся с периодом, делящим ширину кадра, а фигуры у краёв
        дублируются со сдвигом ±W. Иначе на стыке появилась бы вертикальная
        полоса, и сеть выучила бы её как ориентир.
        """
        th = self._theme
        style = th.bg_style
        if style in ("plain", "noise"):
            return

        # Цвет узора: чуть контрастнее фона, но не «в лицо».
        base = th.bg_top
        goal = th.block_edge if luminance(base) < 128 else th.ground_fill
        pat = mix_rgb(base, goal, 0.22 + 0.3 * float(rng.random()))
        pat2 = mix_rgb(base, goal, 0.45 + 0.3 * float(rng.random()))

        # Шаг узора обязан делить ширину кадра — иначе тайл не сойдётся по шву.
        divisors = [d for d in (4, 8, 16, 32, 64) if W % d == 0] or [max(2, W // 8)]
        idx = int(round(2.0 + math.log2(max(0.25, th.pattern_scale))))
        step = divisors[min(len(divisors) - 1, max(0, idx))]

        if style == "grid":
            for x in range(0, W, step):
                pygame.draw.line(tile, pat, (x, 0), (x, ph - 1))
            for y in range(0, ph, step):
                pygame.draw.line(tile, pat, (0, y), (W - 1, y))
            return

        if style == "stripes":
            slope = 1 if rng.random() < 0.5 else -1
            width = max(1, step // 3)
            for x in range(-ph, W + ph, step):
                pygame.draw.line(tile, pat, (x, 0), (x + slope * ph, ph - 1), width)
                # Дубли со сдвигом ±W: полоса пересекает границу тайла.
                pygame.draw.line(tile, pat, (x - W, 0), (x - W + slope * ph, ph - 1), width)
            return

        if style == "stars":
            count = int(30 + 190 * min(2.0, th.pattern_scale))
            for i in range(count):
                x = int(rng.integers(0, W))
                y = int(rng.integers(0, ph))
                r = 1 if rng.random() < 0.82 else 2
                c = mix_rgb(pat, pat2, float(rng.random()))
                pygame.draw.circle(tile, c, (x, y), r)
            return

        if style in ("circles", "clouds"):
            count = int(6 + 26 * min(2.0, th.pattern_scale))
            for i in range(count):
                x = float(rng.integers(0, W))
                y = float(rng.integers(0, ph))
                r = int(rng.integers(3, max(5, ph // 4)))
                c = mix_rgb(pat, pat2, float(rng.random()))
                for dx in (-W, 0, W):
                    if style == "circles":
                        wdt = 0 if rng.random() < 0.5 else max(1, r // 4)
                        pygame.draw.circle(tile, c, (int(x + dx), int(y)), r, wdt)
                    else:
                        # «Облако» — три перекрывающихся круга: мягче и живее.
                        pygame.draw.circle(tile, c, (int(x + dx), int(y)), r)
                        pygame.draw.circle(tile, c, (int(x + dx - r * 0.7), int(y + r * 0.3)), int(r * 0.7))
                        pygame.draw.circle(tile, c, (int(x + dx + r * 0.8), int(y + r * 0.25)), int(r * 0.6))
            return

    def _build_parallax(self) -> None:
        """Собрать слои дальнего плана: абстрактные фигуры, тайлящиеся по X.

        Слои движутся медленнее камеры (коэффициенты 0.18/0.34/0.52), поэтому
        по кадру создаётся ощущение глубины — и одновременно ещё один класс
        нуисанс-сигналов, который зрение обязано игнорировать.

        Уровень декора здесь НЕ учитывается намеренно: раньше слои не строились,
        если в момент `set_theme` декор был выключен, и после включения декора
        параллакс молча не появлялся до следующей смены темы. Отбор «рисовать
        или нет» живёт в `_draw_parallax`, где ему и место.
        """
        self._parallax = []
        th = self._theme
        layers = th.parallax_layers
        if layers <= 0:
            return

        W, H = self.width, self.height
        ph = H + 2 * _VPAD
        for i in range(layers):
            rng = np.random.default_rng(_stable_seed(th.name, th.seed, "px", i))
            factor = 0.18 + 0.17 * i
            depth = (i + 1) / (layers + 1)
            color = mix_rgb(th.bg_bottom, th.block_fill, 0.10 + 0.28 * (1.0 - depth))
            alpha = int(70 + 90 * (1.0 - depth))
            surf = pygame.Surface((2 * W, ph), pygame.SRCALPHA)
            self._draw_parallax_shapes(surf, W, ph, color, alpha, rng, i)
            self._parallax.append((surf, factor))

    def _draw_parallax_shapes(
        self,
        surf: pygame.Surface,
        W: int,
        ph: int,
        color: RGB,
        alpha: int,
        rng: np.random.Generator,
        index: int,
    ) -> None:
        """Абстрактные фигуры одного слоя параллакса (тайлятся по X)."""
        shape = self._theme.parallax_shape
        col = _rgba(color, alpha)
        count = int(4 + 10 * float(rng.random()))

        if shape == "mountains":
            # Ломаная линия горизонта, замкнутая по краям тайла.
            k = 8
            heights = [float(rng.uniform(0.15, 0.6)) * ph for _ in range(k)]
            heights.append(heights[0])
            base = ph - 1
            pts: list[tuple[float, float]] = []
            for j in range(k + 1):
                pts.append((j * W / k, base - heights[j]))
            for dx in (0, W):
                poly = [(x + dx, y) for x, y in pts]
                poly.append((W + dx, base))
                poly.append((0 + dx, base))
                pygame.draw.polygon(surf, col, poly)
            return

        for _ in range(count):
            x = float(rng.integers(0, W))
            y = float(rng.integers(0, ph))
            size = float(rng.uniform(4.0, 4.0 + ph * 0.35))
            for dx in (-W, 0, W):
                cx = x + dx
                if shape == "blocks":
                    rect = pygame.Rect(int(cx), int(y), int(size), int(size * rng.uniform(0.5, 1.6)))
                    pygame.draw.rect(surf, col, rect, border_radius=int(size // 6))
                elif shape == "triangles":
                    pygame.draw.polygon(
                        surf, col,
                        [(cx, y - size), (cx + size, y + size), (cx - size, y + size)],
                    )
                elif shape == "circles":
                    pygame.draw.circle(surf, col, (int(cx), int(y)), int(max(2, size / 2)))
                else:  # bars
                    rect = pygame.Rect(int(cx), 0, max(2, int(size / 3)), ph)
                    pygame.draw.rect(surf, col, rect)

    def _build_post(self) -> None:
        """Предрасчёт всего, что нужно пост-эффектам: виньетка, LUT, поле шума."""
        th = self._theme
        W, H = self.width, self.height
        key = (H, W)

        # Виньетка: 1 в центре, темнее к углам. Мягкая степень 1.5 — иначе
        # получается «дырка от объектива», а не затемнение краёв.
        rad = _VIGNETTE_CACHE.get(key)
        if rad is None:
            yy = (np.arange(H, dtype=np.float32) - (H - 1) / 2.0) / max(1.0, (H - 1) / 2.0)
            xx = (np.arange(W, dtype=np.float32) - (W - 1) / 2.0) / max(1.0, (W - 1) / 2.0)
            rad = np.sqrt(yy[:, None] ** 2 * 0.9 + xx[None, :] ** 2)
            rad = (np.clip(rad / 1.35, 0.0, 1.0) ** 1.5).astype(np.float32)[:, :, None]
            _VIGNETTE_CACHE[key] = rad
        strength = th.vignette * (0.35 + 0.65 * self.decoration_level)
        self._vignette = (1.0 - strength * rad).astype(np.float32)

        # Гамма и контраст — одной таблицей на 256 значений.
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.clip((x - 0.5) * th.contrast + 0.5, 0.0, 1.0) ** th.gamma
        self._lut = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)

        # Поле шума двойной ширины: сдвигом окна получаем «живое» зерно без
        # генерации случайных чисел на каждом кадре.
        field = _NOISE_CACHE.get(key)
        if field is None:
            field = np.random.default_rng(_stable_seed(H, W, "noise")).standard_normal(
                (H, 2 * W, 3)
            ).astype(np.float32)
            _NOISE_CACHE[key] = field
        self._noise_field = field

    # ------------------------------------------------------------------
    # слой 1: фон
    # ------------------------------------------------------------------
    def _draw_background(self, surf: pygame.Surface, cam: tuple[float, float], t: int) -> None:
        """Фоновый градиент с узором, медленно ползущий вслед за камерой.

        Здесь же живёт «биение» фона: полупрозрачная заливка поверх градиента,
        пульсирующая с периодом трека. В отличие от общей пульсации в
        пост-обработке она затрагивает ТОЛЬКО фон, поэтому фон и объекты
        «дышат» вразнобой — ещё один признак, на который сети опираться нельзя.
        """
        th = self._theme
        bg = self._bg
        if bg is None:  # pragma: no cover - set_theme всегда его создаёт
            surf.fill(th.bg_bottom)
            return
        W, H = self.width, self.height
        cam_x, cam_y = cam
        # Фон — самый дальний план: коэффициент минимальный.
        factor = 0.08
        sx = int(cam_x * _P * factor) % W
        dy = int(round(cam_y * _P * factor))
        dy = -_VPAD if dy < -_VPAD else (_VPAD if dy > _VPAD else dy)
        surf.blit(bg, (0, 0), area=pygame.Rect(sx, _VPAD - dy, W, H))

        if th.pulse > 0.0 and self.decoration_level > 0.0:
            k = 0.5 + 0.5 * math.sin(2.0 * math.pi * (t % _PULSE_PERIOD) / _PULSE_PERIOD)
            alpha = int(46 * th.pulse * self.decoration_level * k)
            if alpha > 2:
                tint = th.ground_line if th.is_dark() else th.bg_bottom
                self._alpha.fill(_rgba(tint, alpha))
                surf.blit(self._alpha, (0, 0))

    # ------------------------------------------------------------------
    # слой 2: параллакс
    # ------------------------------------------------------------------
    def _draw_parallax(self, surf: pygame.Surface, cam: tuple[float, float]) -> None:
        """Слои дальнего плана — движутся медленнее камеры, создавая глубину."""
        if not self._parallax or self.decoration_level <= 0.0:
            return
        W, H = self.width, self.height
        cam_x, cam_y = cam
        for layer, factor in self._parallax:
            sx = int(cam_x * _P * factor) % W
            dy = int(round(cam_y * _P * factor))
            dy = -_VPAD if dy < -_VPAD else (_VPAD if dy > _VPAD else dy)
            surf.blit(layer, (0, 0), area=pygame.Rect(sx, _VPAD - dy, W, H))

    # ------------------------------------------------------------------
    # слой 3: неигровые декорации, привязанные к миру
    # ------------------------------------------------------------------
    def _cell_items(self, cell: int) -> tuple[tuple[Any, ...], ...]:
        """Декорации одной ячейки мира — детерминированно по её индексу.

        Каждый элемент несёт «ранг» в [0, 1): на кадр попадают только элементы
        с рангом ниже `decoration_level`. Так плотность декора крутится одной
        ручкой, а сам набор не приходится пересоздавать (и он не «мигает» при
        изменении уровня декора).
        """
        cached = self._deco_cache.get(cell)
        if cached is not None:
            return cached
        rng = np.random.default_rng(_stable_seed(self._deco_seed, "cell", cell))
        base_x = cell * _DECO_CELL
        # Один вызов на всю ячейку вместо шестидесяти скалярных: генерация
        # декора идёт при каждой смене темы, и скалярный rng заметен в профиле.
        vals = rng.random((_DECO_MAX_PER_CELL, 9)).tolist()
        items: list[tuple[Any, ...]] = []
        for row in vals:
            items.append((
                row[0],                                 # 0 ранг
                int(row[1] * 4093),                     # 1 вид (маппится стилем темы)
                base_x + row[2] * _DECO_CELL,           # 2 мировой x
                -0.5 + row[3] * 11.0,                   # 3 мировой y
                0.25 + row[4] * 2.35,                   # 4 ширина в тайлах
                0.25 + row[5] * 2.95,                   # 5 высота в тайлах
                int(row[6] * 4093),                     # 6 индекс цвета
                70 + int(row[7] * 170),                 # 7 базовая альфа
                row[8],                                 # 8 свободный параметр
            ))
        result = tuple(items)
        self._deco_cache[cell] = result
        return result

    def _draw_decor(self, surf: pygame.Surface, cam: tuple[float, float], t: int) -> None:
        """Слой неигровых декораций: их нет ни в физике, ни в карте.

        Именно здесь создаётся главная трудность для зрения: полосы, трубы,
        шестерёнки, глифы и ПЛАВАЮЩИЕ БЛОКИ без хитбокса выглядят как часть
        уровня, но не являются ею. Отличить их можно только по тому, что они
        полупрозрачны и написаны палитрой декора, — сеть обязана это выучить.
        """
        level_decor = self.decoration_level
        if level_decor <= 0.0:
            return
        th = self._theme
        kinds = _DECOR_KINDS[th.decor_style]
        layer = self._alpha
        layer.fill((0, 0, 0, 0))

        # Порог отбора: ручка кадра (`decoration_level`) умножается на плотность
        # ТЕМЫ. Так «пустой минимализм» и «свалка труб» — разные темы, а не
        # разные настройки среды, и обе попадают в датасет при одном и том же
        # decoration_level=1.
        threshold = level_decor * (0.45 + 0.55 * th.decor_density)

        W = self.width
        cam_x, _cam_y = cam
        x0 = cam_x - 2.0
        x1 = cam_x + W / _P + 2.0
        c_start = int(math.floor(x0 / _DECO_CELL))
        c_end = int(math.floor(x1 / _DECO_CELL))
        drawn = 0
        for cell in range(c_start, c_end + 1):
            for item in self._cell_items(cell):
                if item[0] > threshold:
                    continue
                self._draw_decor_item(layer, item, kinds, cam, t)
                drawn += 1
        if drawn:
            surf.blit(layer, (0, 0))

    def decor_count(self, cam: tuple[float, float]) -> int:
        """Сколько неигровых декораций попадает в кадр при текущих настройках.

        Зачем публично: «декораций достаточно, чтобы задача была нетривиальной» —
        проверяемое утверждение, и мерить его надо тем же кодом, что и рисует,
        а не повторяя формулу порога в тесте.
        """
        th = self._theme
        threshold = self.decoration_level * (0.45 + 0.55 * th.decor_density)
        if self.decoration_level <= 0.0:
            return 0
        cam_x, _ = cam
        c_start = int(math.floor((cam_x - 2.0) / _DECO_CELL))
        c_end = int(math.floor((cam_x + self.width / _P + 2.0) / _DECO_CELL))
        return sum(
            1
            for cell in range(c_start, c_end + 1)
            for item in self._cell_items(cell)
            if item[0] <= threshold
        )

    def _draw_decor_item(
        self,
        layer: pygame.Surface,
        item: tuple[Any, ...],
        kinds: tuple[int, ...],
        cam: tuple[float, float],
        t: int,
    ) -> None:
        """Нарисовать одну декорацию в экранных координатах её мировой точки."""
        th = self._theme
        H = self.height
        cam_x, cam_y = cam
        _, raw_kind, wx, wy, tw, thh, ci, alpha, param = item
        kind = kinds[raw_kind % len(kinds)]
        color = th.decor_color(ci)
        # Заметность декора — тоже параметр темы. Раньше альфа была жёстко
        # 40..190, а палитра декора всегда тянулась к фону: «декорация = бледное
        # пятно» превращалось в надёжный признак, по которому сеть отличала бы
        # мусор от объекта, не глядя на форму. Теперь бывает и еле видимый туман,
        # и совершенно непрозрачная конструкция.
        visibility = (0.4 + 0.6 * self.decoration_level) * (0.55 + 0.9 * th.decor_contrast)
        col = _rgba(color, min(_DECOR_ALPHA_CAP, int(alpha * visibility)))
        x = _sx(wx, cam_x)
        y = _sy(wy, cam_y, H)
        w = max(1, int(tw * _P))
        h = max(1, int(thh * _P))

        if kind == _D_BAR:
            if param < 0.5:
                rect = pygame.Rect(int(x), int(y - h / 2), max(1, w // 3), h)
            else:
                rect = pygame.Rect(int(x - w / 2), int(y), w, max(1, h // 3))
            pygame.draw.rect(layer, col, rect)
            return

        if kind == _D_PIPE:
            rect = pygame.Rect(int(x), int(y - h / 2), max(2, w // 2), h)
            pygame.draw.rect(layer, col, rect, border_radius=max(1, w // 6))
            flange = _rgba(mix_rgb(color, th.block_edge, 0.4), col[3])
            pygame.draw.rect(layer, flange, pygame.Rect(rect.x - 1, rect.y, rect.w + 2, 2))
            pygame.draw.rect(layer, flange, pygame.Rect(rect.x - 1, rect.bottom - 2, rect.w + 2, 2))
            return

        if kind == _D_GEAR:
            r = max(2, min(w, h) // 2)
            cx, cy = int(x), int(y)
            pygame.draw.circle(layer, col, (cx, cy), r, max(1, r // 3))
            # Зубцы вращаются со временем — анимация без влияния на геометрию.
            spokes = 6
            phase = (t * (0.03 + 0.05 * param)) % (2 * math.pi)
            for k in range(spokes):
                a = phase + k * (2 * math.pi / spokes)
                pygame.draw.line(
                    layer, col,
                    (cx + math.cos(a) * r * 0.6, cy + math.sin(a) * r * 0.6),
                    (cx + math.cos(a) * (r + 2), cy + math.sin(a) * (r + 2)),
                    1,
                )
            return

        if kind == _D_GLYPH:
            cell = max(1, min(w, h) // 3)
            bits = int(param * 511) | 1
            for k in range(9):
                if not (bits >> k) & 1:
                    continue
                gx = int(x) + (k % 3) * cell
                gy = int(y) + (k // 3) * cell
                pygame.draw.rect(layer, col, pygame.Rect(gx, gy, cell, cell))
            return

        if kind == _D_FLOATER:
            # Плавающий «блок» без хитбокса: самый коварный вид декора.
            bob = math.sin((t * 0.05) + param * 6.283) * 2.0
            rect = pygame.Rect(int(x - w / 2), int(y - h / 2 + bob), w, h)
            pygame.draw.rect(layer, col, rect, border_radius=th.corner_radius)
            edge = _rgba(mix_rgb(color, th.block_edge, 0.5), min(255, col[3] + 40))
            pygame.draw.rect(layer, edge, rect, width=1, border_radius=th.corner_radius)
            return

        if kind == _D_GLOW:
            r = max(2, min(w, h))
            steps = 3
            for k in range(steps, 0, -1):
                a = int(col[3] * 0.5 * k / steps)
                pygame.draw.circle(layer, _rgba(color, a), (int(x), int(y)), int(r * k / steps))
            return

        if kind == _D_CHEVRON:
            size = max(2, min(w, h))
            sign = 1 if param < 0.5 else -1
            for k in range(2):
                off = k * max(2, size // 2)
                pygame.draw.lines(
                    layer, col, False,
                    [(x + off, y - size), (x + off + sign * size, y), (x + off, y + size)],
                    1,
                )
            return

        if kind == _D_LATTICE:
            # Решётка/ферма: прямоугольник с крестовинами. Даёт много тонких
            # линий рядом с игровыми объектами — самый неприятный фон для
            # сегментации кромок блока.
            rect = pygame.Rect(int(x - w / 2), int(y - h / 2), w, h)
            pygame.draw.rect(layer, col, rect, width=1)
            cells = max(1, int(1 + param * 3))
            step_x = max(2, w // cells)
            for gx in range(rect.x, rect.right, step_x):
                pygame.draw.line(layer, col, (gx, rect.y), (min(gx + step_x, rect.right), rect.bottom - 1), 1)
                pygame.draw.line(layer, col, (gx, rect.bottom - 1), (min(gx + step_x, rect.right), rect.y), 1)
            return

        if kind == _D_WAVE:
            # Синусоидальная лента: единственная декорация с криволинейным
            # контуром во всю ширину — ломает «всё прямоугольное» у сети.
            amp = max(1.0, h * 0.4)
            period = max(4.0, w * (0.6 + param))
            phase = t * (0.02 + 0.04 * param)
            pts = [
                (int(x - w / 2 + k), int(y + amp * math.sin(2 * math.pi * (x - w / 2 + k) / period + phase)))
                for k in range(0, w + 1, 2)
            ]
            if len(pts) >= 2:
                pygame.draw.lines(layer, col, False, pts, max(1, min(3, h // 4)))
            return

        if kind == _D_DIAMOND:
            # Ромб/кристалл: форма, которой нет ни у одного игрового объекта,
            # но по «весу» в кадре сравнимая с блоком.
            hw = max(1, w // 2)
            hh = max(1, h // 2)
            bob = math.sin((t * 0.04) + param * 6.283) * 1.5
            cx, cy = int(x), int(y + bob)
            pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
            pygame.draw.polygon(layer, col, pts)
            if hw > 2 and hh > 2:
                pygame.draw.polygon(layer, _rgba(mix_rgb(color, th.bg_top, 0.5), col[3]),
                                    _shrink([(int(px), int(py)) for px, py in pts], 0.5))
            return

        if kind == _D_DASH:
            # Пунктир/«бегущая строка»: короткие штрихи с зазором. Даёт кадру
            # высокочастотный рисунок, похожий на кромку блока, но без объекта.
            seg = max(2, int(2 + param * 4))
            gap = max(1, seg // 2)
            thick = max(1, h // 6)
            gx = int(x - w / 2)
            end = gx + w
            while gx < end:
                pygame.draw.rect(layer, col, pygame.Rect(gx, int(y), min(seg, end - gx), thick))
                gx += seg + gap
            return

        # _D_PATCH: пучок параллельных линий («технический» узор на стене)
        step = max(2, h // 4)
        for gy in range(int(y), int(y) + h, step):
            pygame.draw.line(layer, col, (int(x), gy), (int(x) + w, gy), 1)

    # ------------------------------------------------------------------
    # слой 4: игровые объекты
    # ------------------------------------------------------------------
    def _draw_world(
        self, surf: pygame.Surface, level: Level, cam: tuple[float, float], t: int
    ) -> None:
        """Пол, потолок и все игровые объекты в кадре — в стиле темы.

        Прямоугольники считаются теми же формулами, что и в `semantic.py`,
        поэтому картинка и разметка совпадают пиксель-в-пиксель.
        """
        W, H = self.width, self.height
        cam_x, cam_y = cam
        self._draw_ground(surf, level, cam)

        objs = level.objects_in_range(cam_x - 1.0, cam_x + W / _P + 1.0)
        if not objs:
            return
        objs.sort(key=lambda o: _PRIORITY_BY_TYPE[o.type])
        for obj in objs:
            hx, hy = obj.half_extent()
            c0, c1 = _span_x(obj.x - hx, obj.x + hx, cam_x)
            r0, r1 = _span_y(obj.y - hy, obj.y + hy, cam_y, H)
            w = c1 - c0
            h = r1 - r0
            if w <= 0 or h <= 0 or c1 <= 0 or c0 >= W or r1 <= 0 or r0 >= H:
                continue
            sprite = self._object_sprite(obj.type, w, h)
            if sprite is not None:
                surf.blit(sprite, (c0, r0))

    def _draw_ground(
        self, surf: pygame.Surface, level: Level, cam: tuple[float, float]
    ) -> None:
        """Пол ниже GROUND_Y и потолок выше ceiling_y — сплошные, как в карте.

        В семантике это SOLID до самого края кадра, значит и рисовать их надо
        сплошной заливкой: «линия пола» вместо заливки означала бы, что зрение
        учится видеть SOLID там, где на картинке фон.
        """
        W, H = self.width, self.height
        cam_x, cam_y = cam
        top = H - 0.5

        gr0 = math.ceil(top - (GROUND_Y - cam_y) * _P)
        if gr0 < H:
            y0 = max(0, gr0)
            self._fill_ground_band(surf, pygame.Rect(0, y0, W, H - y0), cam_x, top_edge=(gr0 >= 0))

        cr1 = math.floor(top - (float(level.ceiling_y) - cam_y) * _P) + 1
        if cr1 > 0:
            y1 = min(H, cr1)
            self._fill_ground_band(surf, pygame.Rect(0, 0, W, y1), cam_x, top_edge=False,
                                   bottom_edge=(cr1 <= H))

    def _fill_ground_band(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        cam_x: float,
        top_edge: bool = True,
        bottom_edge: bool = False,
    ) -> None:
        """Залить полосу пола/потолка с узором и светящейся кромкой."""
        th = self._theme
        if rect.h <= 0:
            return
        pygame.draw.rect(surf, th.ground_fill, rect)
        style = th.ground_style
        line = th.ground_line
        pat = mix_rgb(th.ground_fill, line, 0.35)

        if style == "grid":
            step = 8
            off = int(cam_x * _P) % step
            for x in range(-off, rect.w, step):
                pygame.draw.line(surf, pat, (x, rect.y), (x, rect.bottom - 1))
            for y in range(rect.y + step // 2, rect.bottom, step):
                pygame.draw.line(surf, pat, (0, y), (rect.w - 1, y))
        elif style == "stripes":
            step = 6
            off = int(cam_x * _P) % step
            for x in range(-off - rect.h, rect.w, step):
                pygame.draw.line(surf, pat, (x, rect.bottom - 1), (x + rect.h, rect.y), 2)
        elif style == "gradient":
            bands = min(6, max(1, rect.h // 2))
            for i in range(bands):
                c = mix_rgb(line, th.ground_fill, 0.25 + 0.75 * (i / bands))
                y = rect.y + i * rect.h // bands
                pygame.draw.rect(surf, c, pygame.Rect(0, y, rect.w, max(1, rect.h // bands)))
        elif style == "dotted":
            step = 5
            off = int(cam_x * _P) % step
            for y in range(rect.y + 2, rect.bottom, step):
                for x in range(-off + 2, rect.w, step):
                    surf.fill(pat, pygame.Rect(x, y, 1, 1))

        if top_edge:
            pygame.draw.rect(surf, line, pygame.Rect(0, rect.y, rect.w, max(1, th.outline_width)))
        if bottom_edge:
            wdt = max(1, th.outline_width)
            pygame.draw.rect(surf, line, pygame.Rect(0, rect.bottom - wdt, rect.w, wdt))

    # ------------------------------------------------------------------
    # спрайты игровых объектов
    # ------------------------------------------------------------------
    def _object_sprite(self, obj_type: str, w: int, h: int) -> pygame.Surface | None:
        """Спрайт объекта нужного размера (кэшируется по типу и размеру).

        Зачем кэш: в кадре до полусотни блоков одинакового размера, и рисовать
        каждый заново значило бы тратить на них больше времени, чем на всё
        остальное вместе. Здесь блок — это один блит уже готовой картинки.
        """
        cls = _CLASS_BY_TYPE[obj_type]
        shape = SHAPE_BY_TYPE[obj_type]
        key = (cls, shape, w, h)
        sprite = self._sprites.get(key)
        if sprite is not None:
            return sprite

        if cls == SOLID:
            sprite = self._make_block(w, h)
        elif cls == HAZARD:
            sprite = self._make_hazard(shape, w, h)
        elif cls == PAD:
            sprite = self._make_pad(w, h)
        elif cls == ORB:
            sprite = self._make_orb(w, h)
        elif cls in _PORTAL_KIND:
            sprite = self._make_portal(_PORTAL_KIND[cls], w, h)
        elif cls == GOAL:
            sprite = self._make_goal(w, h)
        else:  # pragma: no cover - других классов у объектов уровня нет
            return None
        self._sprites[key] = sprite
        return sprite

    def _make_block(self, w: int, h: int) -> pygame.Surface:
        """Спрайт блока/платформы в стиле темы (`Theme.block_style`).

        Внутри каждого стиля есть ещё и вариации, выбираемые зерном темы:
        направление полос и градиента, сторона фаски, шаг и размер точек,
        глубина «пустой» заливки у обводки. Зачем: имён стилей всего семь
        (контракт SPEC §9), и без вариаций две случайные темы с одним стилем
        давали блок, отличающийся ровно палитрой — то есть структура блока была
        почти константой, и сеть могла выучить именно её.
        """
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        fill = th.block_fill
        edge = th.block_edge
        rad = min(th.corner_radius, max(0, min(w, h) // 2 - 1))
        rect = pygame.Rect(0, 0, w, h)
        style = th.block_style
        # Биты вариации: детерминированы темой, поэтому спрайт остаётся кэшируемым.
        var = _stable_seed(th.name, th.seed, "blockvar")

        if style == "gradient":
            # Градиент и скругление не совмещаем намеренно: вырезать углы из
            # построчной заливки дороже, чем стоит эффект, а «квадратный
            # градиентный блок» — вполне самостоятельный стиль.
            rad = 0
            top = mix_rgb(fill, (255, 255, 255), 0.30)
            bot = mix_rgb(fill, (0, 0, 0), 0.35)
            if var & 1:
                top, bot = bot, top
            if (var >> 1) & 1:
                # Горизонтальный градиент: тот же стиль, другой рисунок.
                for i in range(w):
                    k = i / max(1, w - 1)
                    pygame.draw.line(s, mix_rgb(top, bot, k), (i, 0), (i, h - 1))
            else:
                for i in range(h):
                    k = i / max(1, h - 1)
                    pygame.draw.line(s, mix_rgb(top, bot, k), (0, i), (w - 1, i))
        elif style == "outline":
            # Насколько «пустой» блок внутри: от почти залитого до полой рамки.
            depth = 0.45 + 0.2 * ((var >> 2) & 3)
            pygame.draw.rect(s, mix_rgb(fill, th.bg_bottom, depth), rect, border_radius=rad)
        else:
            pygame.draw.rect(s, fill, rect, border_radius=rad)

        if style == "bevel":
            light = mix_rgb(fill, (255, 255, 255), 0.45)
            dark = mix_rgb(fill, (0, 0, 0), 0.45)
            if (var >> 4) & 1:
                light, dark = dark, light
            b = max(1, min(w, h) // (3 + ((var >> 5) & 1)))
            pygame.draw.polygon(
                s, light,
                [(0, 0), (w, 0), (w - b, b), (b, b), (b, h - b), (0, h)],
            )
            pygame.draw.polygon(
                s, dark,
                [(w, 0), (w, h), (0, h), (b, h - b), (w - b, h - b), (w - b, b)],
            )
        elif style == "striped":
            stripe = mix_rgb(fill, edge, 0.4 + 0.1 * ((var >> 6) & 3))
            step = max(2, min(w, h) // 2 + ((var >> 8) & 1))
            thick = 1 + ((var >> 9) & 1)
            if (var >> 10) & 1:
                # Полосы «в другую сторону»: диагональ меняет знак.
                for k in range(-h, w + h, step):
                    pygame.draw.line(s, stripe, (k, 0), (k + h, h), thick)
            elif (var >> 11) & 1:
                # Горизонтальные полосы.
                for k in range(0, h, step):
                    pygame.draw.line(s, stripe, (0, k), (w - 1, k), thick)
            else:
                for k in range(-h, w + h, step):
                    pygame.draw.line(s, stripe, (k, h), (k + h, 0), thick)
        elif style == "dotted":
            dot = mix_rgb(fill, edge, 0.45 + 0.1 * ((var >> 12) & 3))
            step = max(2, min(w, h) // (2 + ((var >> 14) & 1)))
            size = 1 + ((var >> 15) & 1)
            off = (var >> 16) & 1
            for yy in range(step // 2, h, step):
                for xx in range(step // 2 + off * (step // 2), w, step):
                    s.fill(dot, pygame.Rect(xx, yy, size, size))
        elif style == "noise":
            rng = np.random.default_rng(_stable_seed(th.name, th.seed, "blk", w, h))
            speck = mix_rgb(fill, edge, 0.5)
            speck2 = mix_rgb(fill, (0, 0, 0), 0.35)
            for _ in range(max(3, (w * h) // 6)):
                xx = int(rng.integers(0, w))
                yy = int(rng.integers(0, h))
                s.fill(speck if rng.random() < 0.5 else speck2, pygame.Rect(xx, yy, 1, 1))

        if th.outline_width > 0:
            wdt = min(th.outline_width, max(1, min(w, h) // 2))
            pygame.draw.rect(s, edge, rect, width=wdt, border_radius=rad)
        return s

    def _make_hazard(self, shape: str, w: int, h: int) -> pygame.Surface:
        """Спрайт шипа (треугольник по ориентации) или пилы (круг с зубцами)."""
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        fill = th.hazard_fill
        edge = th.hazard_edge
        style = th.hazard_style
        wdt = max(1, min(th.outline_width if th.outline_width else 1, max(1, min(w, h) // 3)))

        if shape == "circle":
            # Круг рисуется ЭЛЛИПСОМ, вписанным в весь прямоугольник спрайта, а
            # не `circle` в его середине: при чётных w/h центр `(w//2, h//2)`
            # уезжает на полпикселя от настоящего центра хитбокса, и пила
            # оказывалась сдвинутой вниз-вправо относительно карты (IoU 0.66).
            # Карта строит ту же фигуру вписанной в тот же bbox — совпадение
            # становится точным.
            rect_full = pygame.Rect(0, 0, w, h)
            r = max(1, min(w, h) // 2)
            c = ((w - 1) / 2.0, (h - 1) / 2.0)
            if style == "outline":
                pygame.draw.ellipse(s, mix_rgb(fill, th.bg_bottom, 0.6), rect_full)
                pygame.draw.ellipse(s, edge, rect_full, wdt)
            elif style == "gradient":
                steps = max(2, r)
                for i in range(steps, 0, -1):
                    col = mix_rgb(edge, fill, i / steps)
                    inset = int(round((steps - i) * 0.5 * min(w, h) / steps))
                    rr = rect_full.inflate(-2 * inset, -2 * inset)
                    if rr.w > 0 and rr.h > 0:
                        pygame.draw.ellipse(s, col, rr)
            else:
                pygame.draw.ellipse(s, fill, rect_full)
                if style == "double":
                    inner = rect_full.inflate(-(w // 2), -(h // 2))
                    if inner.w > 0 and inner.h > 0:
                        pygame.draw.ellipse(s, edge, inner)
            # Зубцы пилы: короткие лучи по кругу. Наружу за радиус они НЕ
            # выходят: пила и так занимает всего 4-5 px, и лишний пиксель зубца
            # раздувал нарисованную опасность в полтора раза относительно
            # хитбокса, которым размечена карта.
            teeth = 8
            for k in range(teeth):
                a = k * (2 * math.pi / teeth)
                pygame.draw.line(
                    s, edge,
                    (c[0] + math.cos(a) * r * 0.45, c[1] + math.sin(a) * r * 0.45),
                    (c[0] + math.cos(a) * r, c[1] + math.sin(a) * r),
                    1,
                )
            return s

        pts = _tri_points(shape, w, h)
        if style == "outline":
            pygame.draw.polygon(s, mix_rgb(fill, th.bg_bottom, 0.6), pts)
            pygame.draw.polygon(s, edge, pts, wdt)
        elif style == "gradient":
            pygame.draw.polygon(s, fill, pts)
            pygame.draw.polygon(s, mix_rgb(fill, edge, 0.6), _shrink(pts, 0.55))
        elif style == "double":
            pygame.draw.polygon(s, fill, pts)
            pygame.draw.polygon(s, edge, _shrink(pts, 0.5))
        else:
            pygame.draw.polygon(s, fill, pts)
            if th.outline_width > 0:
                pygame.draw.polygon(s, edge, pts, wdt)
        return s

    def _make_pad(self, w: int, h: int) -> pygame.Surface:
        """Спрайт трамплина: широкая низкая «плита» с пружиной.

        Спрайт заливает ВЕСЬ хитбокс. Раньше середина плиты оставалась
        прозрачной, и сквозь неё просвечивал фон с декорациями: примерно 8%
        пикселей, размеченных как PAD, на картинке были обычным фоном — то есть
        сеть учили угадывать класс там, где объекта буквально не видно.
        """
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        fill = th.pad_fill
        base = mix_rgb(fill, th.block_edge, 0.35)
        pygame.draw.rect(s, base, pygame.Rect(0, 0, w, h),
                         border_radius=max(0, min(2, h // 2)))
        cap = mix_rgb(fill, (255, 255, 255), 0.45)
        pygame.draw.rect(s, cap, pygame.Rect(0, 0, w, max(1, h // 3)))
        # Две «ножки» пружины — читаемый признак пада даже на 8x4 пикселях.
        leg = mix_rgb(fill, th.block_edge, 0.4)
        pygame.draw.line(s, leg, (w // 4, h - 1), (w // 4, max(0, h // 3)), 1)
        pygame.draw.line(s, leg, (3 * w // 4, h - 1), (3 * w // 4, max(0, h // 3)), 1)
        return s

    def _make_orb(self, w: int, h: int) -> pygame.Surface:
        """Спрайт кольца: именно КОЛЬЦО с дыркой — его нельзя путать с пилой.

        Толщина кольца берётся из `RING_INNER_RATIO` семантики, а не «на глаз»:
        карта рисует кольцо с дыркой ровно этого радиуса, и любое расхождение
        превращается в пиксели, размеченные ORB, но показанные как фон.
        """
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, w, h)
        r = max(2, min(w, h) // 2)
        ring = max(1, r - int(round(r * RING_INNER_RATIO)))
        pygame.draw.ellipse(s, th.orb_fill, rect, ring)
        halo = mix_rgb(th.orb_fill, (255, 255, 255), 0.5)
        pygame.draw.ellipse(s, halo, rect, 1)
        return s

    def _make_portal(self, kind: str, w: int, h: int) -> pygame.Surface:
        """Спрайт портала: вертикальный овал + метка рода внутри.

        Три рода порталов — три разных семантических класса при одинаковой
        форме, поэтому кроме цвета внутрь кладётся ФОРМЕННАЯ метка (стрелка,
        круг, двойной шеврон). Это единственный признак, который переживает
        монохромную тему.
        """
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        color = th.portal_color(kind)
        rect = pygame.Rect(0, 0, w, h)
        pygame.draw.ellipse(s, _rgba(mix_rgb(color, th.bg_bottom, 0.55), 210), rect)
        pygame.draw.ellipse(s, color, rect, max(1, min(2, w // 3)))
        cx, cy = w // 2, h // 2
        mark = mix_rgb(color, (255, 255, 255), 0.5)
        size = max(2, min(w, h) // 4)
        if kind == "gravity":
            pygame.draw.lines(s, mark, False,
                              [(cx - size, cy + size // 2), (cx, cy - size), (cx + size, cy + size // 2)], 1)
        elif kind == "mode":
            pygame.draw.circle(s, mark, (cx, cy), size, 1)
        else:
            for k in (-1, 1):
                off = k * size
                pygame.draw.lines(s, mark, False,
                                  [(cx - size // 2, cy + off - size // 2),
                                   (cx + size // 2, cy + off),
                                   (cx - size // 2, cy + off + size // 2)], 1)
        return s

    def _make_goal(self, w: int, h: int) -> pygame.Surface:
        """Спрайт финиша: клетчатая вертикальная полоса во всю высоту."""
        th = self._theme
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        a = th.goal_fill
        b = mix_rgb(a, (0, 0, 0), 0.65)
        cell = max(1, w // 2)
        for j, y in enumerate(range(0, h, cell)):
            for i, x in enumerate(range(0, w, cell)):
                pygame.draw.rect(s, a if (i + j) % 2 == 0 else b, pygame.Rect(x, y, cell, cell))
        return s

    # ------------------------------------------------------------------
    # слой 5: игрок, след, партиклы
    # ------------------------------------------------------------------
    def _make_player(self, mode: str, w: int, h: int) -> pygame.Surface:
        """Спрайт игрока: внешний контур = хитбокс, внутри — метка режима.

        Внешний прямоугольник совпадает с тем, что рисует карта: зрение должно
        находить именно коробку столкновений. Внутренняя метка (квадрат/треугольник/
        ромб) — чистая декорация, помогающая человеку понять режим.
        """
        th = self._theme
        key = ("player", mode, w, h)
        cached = self._sprites.get(key)
        if cached is not None:
            return cached
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, w, h)
        rad = min(th.corner_radius, max(0, min(w, h) // 3))
        pygame.draw.rect(s, th.player_fill, rect, border_radius=rad)
        inner = mix_rgb(th.player_fill, th.player_edge, 0.55)
        # Метка режима занимает СЕРЕДИНУ коробки, а не почти всю её. Раньше она
        # отступала от края на min(w,h)//4 (для куба 7x7 это 1 px), и оставшуюся
        # рамку добивала обводка — цвет игрока не было видно вообще, кубик
        # выглядел тёмным квадратом любой темы. Игрок — самый важный объект в
        # кадре, и его палитра обязана читаться.
        side = max(2, min(w, h) // 2)
        mx = (w - side) // 2
        my = (h - side) // 2
        if mode == "ship":
            pygame.draw.polygon(s, inner, [(mx, my + side), (mx + side, my + side // 2), (mx, my)])
        elif mode == "wave":
            pygame.draw.polygon(
                s, inner,
                [(mx + side // 2, my), (mx + side, my + side // 2),
                 (mx + side // 2, my + side), (mx, my + side // 2)],
            )
        else:
            pygame.draw.rect(s, inner, pygame.Rect(mx, my, side, side))
        if th.outline_width > 0:
            pygame.draw.rect(s, th.player_edge, rect,
                             width=min(th.outline_width, max(1, min(w, h) // 3)), border_radius=rad)
        self._sprites[key] = s
        return s

    def _draw_player(
        self, surf: pygame.Surface, state: PlayerState, cam: tuple[float, float], t: int
    ) -> None:
        """Игрок, его след и партиклы вокруг него."""
        th = self._theme
        H = self.height
        cam_x, cam_y = cam
        hx, hy = player_half_extent(state.mode)
        c0, c1 = _span_x(state.x - hx, state.x + hx, cam_x)
        r0, r1 = _span_y(state.y - hy, state.y + hy, cam_y, H)
        w = max(1, c1 - c0)
        h = max(1, r1 - r0)

        deco = self.decoration_level
        if deco > 0.0 and (th.trail > 0.0 or th.particles > 0.0):
            layer = self._alpha
            layer.fill((0, 0, 0, 0))
            self._draw_trail(layer, state, cam, w, h, c0, r0)
            self._draw_particles(layer, state, cam, t)
            surf.blit(layer, (0, 0))

        surf.blit(self._make_player(state.mode, w, h), (c0, r0))

    def _draw_trail(
        self,
        layer: pygame.Surface,
        state: PlayerState,
        cam: tuple[float, float],
        w: int,
        h: int,
        c0: int,
        r0: int,
    ) -> None:
        """След за игроком: затухающие копии позади него.

        След синтезируется из текущего состояния (истории у рендера нет — он
        обязан оставаться чистой функцией, иначе кадры датасета зависели бы от
        порядка вызовов). Для глаза этого достаточно, а для зрения это ещё один
        нуисанс прямо у самого важного объекта.
        """
        th = self._theme
        if th.trail <= 0.0:
            return
        steps = int(2 + 5 * th.trail * self.decoration_level)
        color = th.player_fill
        for k in range(1, steps + 1):
            frac = k / (steps + 1)
            # Потолок альфы намеренно невысокий: след обязан читаться как след,
            # а не как продолжение игрока — иначе разметка «коробка игрока»
            # перестанет соответствовать тому, что видно на картинке.
            alpha = int(105 * th.trail * (1.0 - frac))
            if alpha <= 3:
                continue
            dx = int(k * max(2, w // 2))
            size_w = max(1, int(w * (1.0 - 0.5 * frac)))
            size_h = max(1, int(h * (1.0 - 0.5 * frac)))
            rect = pygame.Rect(
                c0 - dx + (w - size_w) // 2,
                r0 + (h - size_h) // 2,
                size_w,
                size_h,
            )
            pygame.draw.rect(layer, _rgba(color, alpha), rect, border_radius=th.corner_radius)

    def _draw_particles(
        self, layer: pygame.Surface, state: PlayerState, cam: tuple[float, float], t: int
    ) -> None:
        """Партиклы: искры у игрока и «пыль», летящая навстречу движению."""
        th = self._theme
        density = th.particles * self.decoration_level
        if density <= 0.0:
            return
        W, H = self.width, self.height
        cam_x, cam_y = cam
        count = int(4 + 26 * density)
        base = (t * 31) & (_RAND_TAB - 1)
        spark = mix_rgb(th.player_fill, th.ground_line, 0.4)
        dust = mix_rgb(th.bg_top, th.block_edge, 0.5)

        for i in range(count):
            r1 = self._r(base + i * 3)
            r2 = self._r(base + i * 3 + 1)
            r3 = self._r(base + i * 3 + 2)
            if i % 3 == 0:
                # Искры рядом с игроком.
                x = _sx(state.x + (r1 - 0.7) * 1.2, cam_x)
                y = _sy(state.y + (r2 - 0.5) * 1.2, cam_y, H)
                col = _rgba(spark, int(90 + 140 * r3))
                size = 1 if r3 < 0.7 else 2
            else:
                # Свободная пыль по всему кадру, плывущая влево.
                x = (r1 * W - t * (0.6 + 1.8 * r3)) % W
                y = r2 * H
                col = _rgba(dust, int(30 + 110 * r3 * density))
                size = 1
            layer.fill(col, pygame.Rect(int(x), int(y), size, size))

    # ------------------------------------------------------------------
    # слой 6: пост-эффекты
    # ------------------------------------------------------------------
    def _post(self, arr: np.ndarray, t: int) -> np.ndarray:
        """Пост-обработка кадра: пульс, bloom, аберрация, виньетка, шум, гамма.

        Зачем всё это в датасете: реальные уровни Geometry Dash залиты
        свечением, а реальный захват экрана добавляет сжатие и шум. Сеть,
        обученная на «чистых» кадрах, разваливается от первого же блика.
        """
        th = self._theme
        deco = self.decoration_level
        f = arr.astype(np.float32)

        # Пульсация яркости «под музыку».
        if th.pulse > 0.0 and deco > 0.0:
            amp = (0.05 + 0.25 * th.pulse) * deco
            f *= 1.0 + amp * math.sin(2.0 * math.pi * (t % _PULSE_PERIOD) / _PULSE_PERIOD)

        # Bloom: светлое расплывается и подсвечивает соседей.
        strength = th.bloom * (0.35 + 0.65 * th.glow) * (0.3 + 0.7 * deco)
        if strength > 0.02:
            bright = f - _BLOOM_THRESHOLD
            np.maximum(bright, 0.0, out=bright)
            add = _bloom_blur(bright)
            add *= 1.1 * strength
            np.minimum(add, _BLOOM_CAP, out=add)
            f += add

        # Хроматическая аберрация: каналы R и B расходятся на пиксель.
        if th.chromatic > 0.02 and deco > 0.0:
            a = th.chromatic * deco * 0.8
            red = f[:, :, 0]
            blue = f[:, :, 2]
            red_sh = np.empty_like(red)
            red_sh[:, 1:] = red[:, :-1]
            red_sh[:, 0] = red[:, 0]
            blue_sh = np.empty_like(blue)
            blue_sh[:, :-1] = blue[:, 1:]
            blue_sh[:, -1] = blue[:, -1]
            f[:, :, 0] = red * (1.0 - a) + red_sh * a
            f[:, :, 2] = blue * (1.0 - a) + blue_sh * a

        if self._vignette is not None:
            f *= self._vignette

        if th.noise > 0.0 and self._noise_field is not None:
            off = (t * 13) % self.width
            f += self._noise_field[:, off:off + self.width] * (th.noise * 42.0 * (0.3 + 0.7 * deco))

        np.clip(f, 0.0, 255.0, out=f)
        out = f.astype(np.uint8)
        if self._lut is not None:
            out = self._lut[out]

        # Тряска: сдвиг ВСЕГО кадра на целое число пикселей. Запоминаем его,
        # чтобы вызывающий мог применить тот же сдвиг к семантической карте.
        if th.shake > 0.0 and deco > 0.0:
            amp = th.shake * deco
            dx = int(round(amp * math.sin(t * 0.9)))
            dy = int(round(amp * math.cos(t * 1.3)))
            if dx or dy:
                out = np.roll(np.roll(out, dx, axis=1), dy, axis=0)
            self.last_shake = (dx, dy)
        else:
            self.last_shake = (0, 0)

        return np.ascontiguousarray(out)


def make_renderer(
    theme: Theme | str | None = None,
    decoration_level: float = 1.0,
    seed: int | None = None,
    width: int = OBS_W,
    height: int = OBS_H,
) -> Renderer:
    """Собрать рендерер, принимая тему объектом, именем или None.

    Зачем: у среды тема приходит из `Level.theme_hint` (строка) или из конфига,
    и раскладывать эту развилку по всем вызывающим местам — лишний повод для
    ошибки.
    """
    if isinstance(theme, str):
        resolved: Theme | None = theme_by_name(theme)
    else:
        resolved = theme
    return Renderer(
        width=width, height=height, theme=resolved,
        decoration_level=decoration_level, seed=seed,
    )


__all__ = ["Renderer", "make_renderer"]
