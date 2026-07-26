"""Темы оформления: палитры, стили, «злые» случайные комбинации.

Зачем этот модуль
-----------------
Политика в GDAI никогда не видит картинку — она работает с канонической картой.
Но зрение (U-Net) обязано переводить в эту карту ЛЮБОЙ дизайн: неоновый уровень,
пиксель-арт, монохром, «лаву», блюпринт. Единственный способ этого добиться —
показать сети столько разных оформлений, чтобы «выучить палитру» стало
невозможно, и остался только один надёжный признак: ФОРМА и ПОЛОЖЕНИЕ объекта.

Поэтому `Theme` описывает исключительно оформление и не содержит ни одного поля,
способного повлиять на геометрию: ни смещений, ни размеров объектов, ни камеры.
Тема может перекрасить шип в цвет блока, утопить передний план в фоне и
перевернуть всю палитру — семантическая карта от этого не изменится ни на байт.

Про «злые» темы
---------------
`random_theme` намеренно сэмплит вредные случаи:

* **похожий шип** — цвет шипа почти совпадает с цветом блока (сеть не сможет
  разделить их по цвету и вынуждена смотреть на треугольную форму);
* **слабый контраст** — передний план почти сливается с фоном;
* **монохром** — цветовой канал вообще не несёт информации;
* **инверсия** — светлое становится тёмным, «тёмный фон» перестаёт быть якорем.

Такие темы делают обучение медленнее, но именно они дают устойчивость: если
сеть выжила на них, обычный красивый уровень для неё тривиален.

Все цвета — кортежи `(r, g, b)` в диапазоне 0..255. Вся случайность идёт через
`np.random.Generator`, переданный явно, — прогон воспроизводим по seed.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

# Тип цвета. Отдельный алиас нужен, чтобы сигнатуры читались как в SPEC §9.
RGB = tuple[int, int, int]

# --- допустимые значения стилей ---------------------------------------------
# Зачем валидировать: опечатка в стиле («striped» -> «stripped») не уронила бы
# рендер, а тихо превратила бы половину датасета в одинаковые кадры — то есть
# незаметно убила бы доменную рандомизацию.
BLOCK_STYLES: tuple[str, ...] = (
    "flat", "outline", "bevel", "striped", "dotted", "gradient", "noise",
)
HAZARD_STYLES: tuple[str, ...] = ("solid", "outline", "gradient", "double")
BG_STYLES: tuple[str, ...] = (
    "plain", "grid", "stars", "stripes", "circles", "clouds", "noise",
)
GROUND_STYLES: tuple[str, ...] = ("flat", "grid", "stripes", "gradient", "dotted")
PARALLAX_SHAPES: tuple[str, ...] = (
    "blocks", "triangles", "circles", "bars", "mountains",
)
DECOR_STYLES: tuple[str, ...] = (
    "bars", "pipes", "gears", "glyphs", "floaters", "mixed",
    "lattice", "waves", "crystals",
)

# Названия полей-цветов: используются в __post_init__ и в монохром/инверсии.
_COLOR_FIELDS: tuple[str, ...] = (
    "bg_top", "bg_bottom",
    "block_fill", "block_edge",
    "hazard_fill", "hazard_edge",
    "player_fill", "player_edge",
    "ground_fill", "ground_line",
    "pad_fill", "orb_fill", "portal_fill", "goal_fill",
)


# ---------------------------------------------------------------------------
# мелкая цветовая арифметика
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    """Загнать число в [0, 1] — защита от случайных значений вне диапазона."""
    v = float(value)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clamp(value: float, lo: float, hi: float) -> float:
    """Загнать число в произвольный отрезок."""
    v = float(value)
    return lo if v < lo else (hi if v > hi else v)


def _rgb(color: Any) -> RGB:
    """Привести что угодно похожее на цвет к строгому `(r, g, b)` из int 0..255.

    Зачем: цвета приходят из numpy (np.uint8), из списков после JSON и из
    ручных литералов; pygame принимает не всё, а тихое приведение типов ловит
    ошибки на этапе создания темы, а не в середине рендера.
    """
    try:
        r, g, b = (int(round(float(c))) for c in tuple(color)[:3])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ожидался цвет (r, g, b), получено {color!r}") from exc
    return (
        0 if r < 0 else (255 if r > 255 else r),
        0 if g < 0 else (255 if g > 255 else g),
        0 if b < 0 else (255 if b > 255 else b),
    )


def mix_rgb(a: RGB, b: RGB, t: float) -> RGB:
    """Линейно смешать два цвета: `t=0` -> a, `t=1` -> b.

    Зачем публично: рендер постоянно выводит производные оттенки (тень блока,
    подсветка грани, цвет декора между фоном и передним планом), и делать это
    надо ровно одной формулой, иначе темы «расползаются» по яркости.
    """
    s = _clamp01(t)
    return (
        int(round(a[0] + (b[0] - a[0]) * s)),
        int(round(a[1] + (b[1] - a[1]) * s)),
        int(round(a[2] + (b[2] - a[2]) * s)),
    )


def shade_rgb(color: RGB, factor: float) -> RGB:
    """Умножить яркость цвета (factor<1 — темнее, >1 — светлее, с обрезкой)."""
    f = max(0.0, float(factor))
    return _rgb((color[0] * f, color[1] * f, color[2] * f))


def luminance(color: RGB) -> float:
    """Воспринимаемая яркость 0..255 (ITU-R BT.601).

    Зачем: по ней проверяют контраст пары «объект/фон» и строят монохромные
    темы — в них цвет вообще перестаёт нести информацию.
    """
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def contrast_ratio(a: RGB, b: RGB) -> float:
    """Грубая мера различимости двух цветов: |ΔL| / 255 в [0, 1].

    Зачем не строгий WCAG: нужна не полиграфическая точность, а быстрый способ
    сказать «эти два цвета сеть по яркости не разделит» при генерации злых тем.
    """
    return abs(luminance(a) - luminance(b)) / 255.0


def hsv_rgb(h: float, s: float, v: float) -> RGB:
    """Цвет из HSV (h — по кругу, дробная часть; s, v в [0, 1])."""
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, _clamp01(s), _clamp01(v))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _to_gray(color: RGB) -> RGB:
    """Обесцветить цвет по его яркости (для монохромных тем)."""
    g = int(round(luminance(color)))
    g = 0 if g < 0 else (255 if g > 255 else g)
    return (g, g, g)


def _invert(color: RGB) -> RGB:
    """Инвертировать цвет — светлая тема мгновенно становится тёмной."""
    return (255 - color[0], 255 - color[1], 255 - color[2])


# ---------------------------------------------------------------------------
# сама тема
# ---------------------------------------------------------------------------
@dataclass
class Theme:
    """Полное описание оформления кадра.

    Все поля имеют разумные значения по умолчанию: тему можно собрать частично
    (`Theme(name="x", glow=1.0)`) и не бояться, что рендер упадёт на None.
    Порядок первых семнадцати полей зафиксирован контрактом (SPEC §9), поэтому
    позиционное создание темы остаётся совместимым; всё, что идёт дальше, —
    дополнительные ручки рандомизации, добавленные этим модулем.

    Ни одно поле не влияет на геометрию мира. Это инвариант: как только тема
    получит право сдвинуть объект, кадр и разметка разъедутся, и обучение
    зрения станет обучением на шуме.
    """

    name: str = "custom"

    # --- палитра (контракт SPEC §9) ---
    bg_top: RGB = (16, 18, 32)          # верх фонового градиента
    bg_bottom: RGB = (8, 9, 16)         # низ фонового градиента
    block_fill: RGB = (220, 225, 235)   # заливка блока
    block_edge: RGB = (255, 255, 255)   # обводка/грань блока
    hazard_fill: RGB = (230, 60, 60)    # заливка шипа/пилы
    hazard_edge: RGB = (255, 200, 200)  # обводка шипа
    player_fill: RGB = (255, 210, 60)   # заливка игрока
    player_edge: RGB = (40, 30, 10)     # обводка игрока
    ground_fill: RGB = (40, 44, 60)     # заливка пола/потолка
    ground_line: RGB = (200, 210, 255)  # светящаяся кромка пола

    # --- стили и эффекты (контракт SPEC §9) ---
    glow: float = 0.4            # 0..1 сила свечения объектов
    block_style: str = "flat"    # см. BLOCK_STYLES
    hazard_style: str = "solid"  # см. HAZARD_STYLES
    bg_style: str = "plain"      # см. BG_STYLES
    parallax_layers: int = 1     # 0..3 слоёв дальнего плана
    particles: float = 0.4       # 0..1 плотность партиклов
    pulse: float = 0.2           # 0..1 «биение» яркости под музыку

    # --- дополнительные ручки рандомизации ---
    pad_fill: RGB = (255, 150, 220)     # трамплин
    orb_fill: RGB = (90, 220, 255)      # кольцо
    portal_fill: RGB = (120, 140, 255)  # базовый цвет порталов
    goal_fill: RGB = (255, 255, 255)    # финишная полоса
    # Палитра НЕИГРОВЫХ декораций. Отдельная от игровой намеренно: у сети
    # должен быть шанс отличить декор от объекта, иначе задача неразрешима.
    decor_colors: tuple[RGB, ...] = ((60, 70, 110), (90, 100, 150), (40, 45, 70))
    outline_width: int = 1       # 0..3 px толщина обводки
    corner_radius: int = 0       # 0..6 px скругление углов блоков
    ground_style: str = "flat"   # см. GROUND_STYLES
    parallax_shape: str = "blocks"  # см. PARALLAX_SHAPES
    decor_style: str = "mixed"      # см. DECOR_STYLES
    # Сколько неигрового мусора живёт в кадре и насколько он заметен. Две ручки,
    # а не одна: «много бледного тумана» и «три ярких трубы» — принципиально
    # разные задачи для зрения, и обе обязаны встречаться в датасете.
    decor_density: float = 0.5   # 0..1 плотность декораций
    decor_contrast: float = 0.5  # 0..1 насколько декор выделяется на фоне
    bloom: float = 0.35          # 0..1 сила пост-свечения
    vignette: float = 0.3        # 0..1 затемнение по краям
    noise: float = 0.05          # 0..1 зернистость кадра
    contrast: float = 1.0        # 0.3..2.5, 1.0 — без изменений
    gamma: float = 1.0           # 0.4..2.5, 1.0 — без изменений
    chromatic: float = 0.0       # 0..1 хроматическая аберрация
    trail: float = 0.5           # 0..1 длина следа за игроком
    # Тряска камеры в пикселях, 0..1 (SPEC §9: не более пикселя). По умолчанию
    # 0: кадр обязан совпадать с разметкой пиксель-в-пиксель, а сдвинуть можно
    # только оба массива сразу — это решение вызывающего, не темы.
    shake: float = 0.0
    pattern_scale: float = 1.0   # 0.25..4, масштаб фонового узора
    monochrome: bool = False     # палитра обесцвечена
    inverted: bool = False       # палитра инвертирована
    # Зерно процедурных узоров: две темы с одинаковой палитрой, но разным
    # seed дадут разное расположение звёзд, полос и декораций.
    seed: int = 0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        for field_name in _COLOR_FIELDS:
            setattr(self, field_name, _rgb(getattr(self, field_name)))
        colors = tuple(_rgb(c) for c in self.decor_colors)
        # Пустая палитра декора сломала бы выбор цвета по индексу.
        self.decor_colors = colors if colors else (mix_rgb(self.bg_top, self.block_fill, 0.3),)

        self.glow = _clamp01(self.glow)
        self.particles = _clamp01(self.particles)
        self.pulse = _clamp01(self.pulse)
        self.bloom = _clamp01(self.bloom)
        self.vignette = _clamp01(self.vignette)
        self.noise = _clamp01(self.noise)
        self.trail = _clamp01(self.trail)
        self.chromatic = _clamp01(self.chromatic)
        self.decor_density = _clamp01(self.decor_density)
        self.decor_contrast = _clamp01(self.decor_contrast)
        self.contrast = _clamp(self.contrast, 0.3, 2.5)
        self.gamma = _clamp(self.gamma, 0.4, 2.5)
        self.shake = _clamp(self.shake, 0.0, 1.0)
        self.pattern_scale = _clamp(self.pattern_scale, 0.25, 4.0)

        self.parallax_layers = int(_clamp(self.parallax_layers, 0, 3))
        self.outline_width = int(_clamp(self.outline_width, 0, 3))
        self.corner_radius = int(_clamp(self.corner_radius, 0, 6))
        self.seed = int(self.seed) & 0xFFFFFFFF
        self.monochrome = bool(self.monochrome)
        self.inverted = bool(self.inverted)

        _check_choice("block_style", self.block_style, BLOCK_STYLES)
        _check_choice("hazard_style", self.hazard_style, HAZARD_STYLES)
        _check_choice("bg_style", self.bg_style, BG_STYLES)
        _check_choice("ground_style", self.ground_style, GROUND_STYLES)
        _check_choice("parallax_shape", self.parallax_shape, PARALLAX_SHAPES)
        _check_choice("decor_style", self.decor_style, DECOR_STYLES)

    # --- производные цвета ---------------------------------------------------
    def portal_color(self, kind: str) -> RGB:
        """Цвет портала по его роду: "gravity" | "mode" | "speed".

        Зачем разные цвета: три вида порталов — это три разных семантических
        класса при одинаковой форме овала. Если покрасить их одинаково, зрение
        физически не сможет их различить. Оттенок разводится по кругу, а
        яркость — дополнительно, чтобы порталы оставались различимы даже в
        монохромной теме (там hue не несёт информации).
        """
        base = self.portal_fill
        if kind == "gravity":
            return base
        h, s, v = colorsys.rgb_to_hsv(base[0] / 255.0, base[1] / 255.0, base[2] / 255.0)
        if kind == "mode":
            return hsv_rgb(h + 1.0 / 3.0, s, _clamp(v * 1.18, 0.0, 1.0))
        if kind == "speed":
            return hsv_rgb(h + 2.0 / 3.0, s, _clamp(v * 0.72, 0.0, 1.0))
        raise ValueError(f"Неизвестный род портала {kind!r}: ожидалось gravity|mode|speed")

    def decor_color(self, index: int) -> RGB:
        """Цвет декорации по индексу (циклически) — декор не трогает игровую палитру."""
        palette = self.decor_colors
        return palette[int(index) % len(palette)]

    def is_dark(self) -> bool:
        """Тёмная ли тема — рендер по этому признаку выбирает сторону подсветки."""
        return luminance(self.bg_bottom) < 110.0

    # --- сервис --------------------------------------------------------------
    def replace(self, **changes: Any) -> "Theme":
        """Копия темы с изменёнными полями (тема неизменяема по смыслу)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимое представление (кортежи -> списки)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "decor_colors":
                out[f.name] = [list(c) for c in value]
            elif f.name in _COLOR_FIELDS:
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Theme":
        """Собрать тему из словаря, игнорируя незнакомые ключи.

        Зачем терпимость: чекпойнты и логи могут быть записаны более старой или
        более новой версией — падать из-за лишнего поля оформления глупо.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "decor_colors" in kwargs:
            kwargs["decor_colors"] = tuple(tuple(c) for c in kwargs["decor_colors"])
        return cls(**kwargs)


def _check_choice(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Проверить, что стиль из допустимого набора; иначе — понятная ошибка."""
    if value not in allowed:
        raise ValueError(
            f"Theme.{field_name}={value!r} недопустимо. Разрешены: {allowed}"
        )


# ---------------------------------------------------------------------------
# встроенные темы: стилистика реальных уровней Geometry Dash
# ---------------------------------------------------------------------------
BUILTIN_THEMES: tuple[Theme, ...] = (
    Theme(
        name="neon",
        bg_top=(12, 6, 38), bg_bottom=(3, 2, 12),
        block_fill=(26, 236, 220), block_edge=(206, 255, 255),
        hazard_fill=(255, 40, 132), hazard_edge=(255, 196, 224),
        player_fill=(255, 240, 96), player_edge=(18, 8, 40),
        ground_fill=(14, 10, 48), ground_line=(64, 244, 232),
        glow=0.95, block_style="outline", hazard_style="outline", bg_style="stars",
        parallax_layers=2, particles=0.85, pulse=0.7,
        pad_fill=(255, 118, 232), orb_fill=(122, 255, 198),
        portal_fill=(140, 128, 255), goal_fill=(250, 255, 255),
        decor_colors=((26, 20, 74), (62, 40, 134), (0, 168, 182)),
        outline_width=1, corner_radius=0, ground_style="grid",
        parallax_shape="bars", decor_style="mixed",
        decor_density=0.85, decor_contrast=0.8,
        bloom=0.9, vignette=0.45, noise=0.04, contrast=1.15, gamma=0.95,
        chromatic=0.25, trail=0.9, pattern_scale=1.0, seed=101,
    ),
    Theme(
        name="retro_pixel",
        bg_top=(74, 108, 148), bg_bottom=(44, 66, 96),
        block_fill=(206, 154, 88), block_edge=(112, 70, 34),
        hazard_fill=(196, 62, 46), hazard_edge=(96, 24, 18),
        player_fill=(238, 226, 190), player_edge=(60, 40, 24),
        ground_fill=(94, 62, 34), ground_line=(150, 106, 58),
        glow=0.05, block_style="dotted", hazard_style="solid", bg_style="grid",
        parallax_layers=1, particles=0.15, pulse=0.05,
        pad_fill=(232, 196, 72), orb_fill=(120, 196, 232),
        portal_fill=(150, 110, 210), goal_fill=(240, 240, 220),
        decor_colors=((60, 88, 124), (86, 118, 156), (120, 84, 44)),
        outline_width=1, corner_radius=0, ground_style="dotted",
        parallax_shape="mountains", decor_style="glyphs",
        decor_density=0.5, decor_contrast=0.45,
        bloom=0.05, vignette=0.15, noise=0.10, contrast=1.05, gamma=1.0,
        chromatic=0.0, trail=0.15, pattern_scale=2.0, seed=202,
    ),
    Theme(
        name="monochrome",
        bg_top=(232, 232, 232), bg_bottom=(196, 196, 196),
        block_fill=(46, 46, 46), block_edge=(12, 12, 12),
        hazard_fill=(80, 80, 80), hazard_edge=(6, 6, 6),
        player_fill=(255, 255, 255), player_edge=(0, 0, 0),
        ground_fill=(60, 60, 60), ground_line=(150, 150, 150),
        glow=0.0, block_style="flat", hazard_style="outline", bg_style="stripes",
        parallax_layers=1, particles=0.2, pulse=0.0,
        pad_fill=(120, 120, 120), orb_fill=(150, 150, 150),
        portal_fill=(96, 96, 96), goal_fill=(20, 20, 20),
        decor_colors=((170, 170, 170), (210, 210, 210), (140, 140, 140)),
        outline_width=2, corner_radius=0, ground_style="stripes",
        parallax_shape="triangles", decor_style="bars",
        decor_density=0.55, decor_contrast=0.7,
        bloom=0.0, vignette=0.25, noise=0.06, contrast=1.25, gamma=1.0,
        chromatic=0.0, trail=0.3, pattern_scale=1.5,
        monochrome=True, seed=303,
    ),
    Theme(
        name="pastel",
        bg_top=(255, 226, 236), bg_bottom=(214, 238, 250),
        block_fill=(168, 214, 200), block_edge=(108, 162, 152),
        hazard_fill=(246, 156, 170), hazard_edge=(196, 96, 118),
        player_fill=(255, 246, 176), player_edge=(180, 150, 100),
        ground_fill=(198, 216, 232), ground_line=(240, 250, 255),
        glow=0.2, block_style="gradient", hazard_style="gradient", bg_style="clouds",
        parallax_layers=2, particles=0.45, pulse=0.25,
        pad_fill=(250, 190, 220), orb_fill=(170, 220, 244),
        portal_fill=(198, 176, 240), goal_fill=(255, 255, 255),
        decor_colors=((240, 214, 234), (206, 232, 240), (250, 236, 208)),
        outline_width=1, corner_radius=3, ground_style="gradient",
        parallax_shape="circles", decor_style="floaters",
        decor_density=0.7, decor_contrast=0.35,
        bloom=0.3, vignette=0.12, noise=0.03, contrast=0.92, gamma=1.05,
        chromatic=0.05, trail=0.5, pattern_scale=1.2, seed=404,
    ),
    Theme(
        name="lava",
        bg_top=(52, 8, 4), bg_bottom=(12, 2, 2),
        block_fill=(48, 30, 26), block_edge=(255, 130, 30),
        hazard_fill=(255, 214, 60), hazard_edge=(255, 90, 0),
        player_fill=(255, 250, 240), player_edge=(90, 20, 0),
        ground_fill=(70, 16, 8), ground_line=(255, 108, 16),
        glow=0.85, block_style="bevel", hazard_style="gradient", bg_style="noise",
        parallax_layers=2, particles=0.9, pulse=0.6,
        pad_fill=(255, 150, 40), orb_fill=(255, 236, 140),
        portal_fill=(220, 90, 40), goal_fill=(255, 240, 200),
        decor_colors=((110, 26, 8), (176, 54, 12), (60, 12, 6)),
        outline_width=1, corner_radius=1, ground_style="gradient",
        parallax_shape="mountains", decor_style="pipes",
        decor_density=0.75, decor_contrast=0.65,
        bloom=0.8, vignette=0.5, noise=0.08, contrast=1.2, gamma=0.9,
        chromatic=0.15, trail=0.8, pattern_scale=0.8, seed=505,
    ),
    Theme(
        name="space",
        bg_top=(14, 8, 46), bg_bottom=(2, 1, 10),
        block_fill=(120, 132, 190), block_edge=(220, 228, 255),
        hazard_fill=(230, 96, 255), hazard_edge=(255, 214, 255),
        player_fill=(255, 255, 255), player_edge=(40, 30, 90),
        ground_fill=(22, 18, 60), ground_line=(140, 150, 255),
        glow=0.7, block_style="gradient", hazard_style="solid", bg_style="stars",
        parallax_layers=3, particles=0.6, pulse=0.35,
        pad_fill=(255, 170, 240), orb_fill=(150, 240, 255),
        portal_fill=(110, 160, 255), goal_fill=(255, 255, 240),
        decor_colors=((40, 30, 96), (70, 54, 150), (18, 14, 52)),
        outline_width=1, corner_radius=2, ground_style="flat",
        parallax_shape="circles", decor_style="crystals",
        decor_density=0.6, decor_contrast=0.5,
        bloom=0.7, vignette=0.55, noise=0.05, contrast=1.05, gamma=1.0,
        chromatic=0.1, trail=0.7, pattern_scale=1.0, seed=606,
    ),
    Theme(
        name="cyberpunk",
        bg_top=(24, 4, 44), bg_bottom=(4, 2, 16),
        block_fill=(20, 22, 40), block_edge=(255, 40, 170),
        hazard_fill=(0, 244, 226), hazard_edge=(200, 255, 252),
        player_fill=(255, 226, 40), player_edge=(40, 0, 60),
        ground_fill=(16, 6, 34), ground_line=(255, 40, 170),
        glow=1.0, block_style="outline", hazard_style="double", bg_style="grid",
        parallax_layers=3, particles=0.7, pulse=0.85,
        pad_fill=(255, 60, 130), orb_fill=(60, 255, 210),
        portal_fill=(180, 60, 255), goal_fill=(255, 255, 255),
        decor_colors=((60, 8, 92), (0, 130, 140), (120, 16, 90)),
        outline_width=2, corner_radius=0, ground_style="grid",
        parallax_shape="bars", decor_style="lattice",
        decor_density=0.95, decor_contrast=0.9,
        bloom=1.0, vignette=0.5, noise=0.07, contrast=1.3, gamma=0.88,
        chromatic=0.45, trail=0.95, pattern_scale=1.0, seed=707,
    ),
    Theme(
        name="blueprint",
        bg_top=(16, 46, 118), bg_bottom=(8, 26, 74),
        block_fill=(20, 56, 132), block_edge=(226, 240, 255),
        hazard_fill=(24, 62, 140), hazard_edge=(255, 255, 255),
        player_fill=(240, 248, 255), player_edge=(10, 30, 80),
        ground_fill=(12, 38, 100), ground_line=(210, 232, 255),
        glow=0.1, block_style="outline", hazard_style="outline", bg_style="grid",
        parallax_layers=0, particles=0.1, pulse=0.0,
        pad_fill=(200, 224, 255), orb_fill=(180, 214, 255),
        portal_fill=(160, 200, 255), goal_fill=(255, 255, 255),
        decor_colors=((30, 70, 150), (46, 92, 176), (18, 50, 120)),
        outline_width=1, corner_radius=0, ground_style="grid",
        parallax_shape="blocks", decor_style="lattice",
        decor_density=0.8, decor_contrast=0.6,
        bloom=0.1, vignette=0.2, noise=0.04, contrast=1.1, gamma=1.0,
        chromatic=0.0, trail=0.2, pattern_scale=0.5, seed=808,
    ),
    Theme(
        name="minimal",
        bg_top=(250, 250, 250), bg_bottom=(238, 238, 240),
        block_fill=(28, 28, 32), block_edge=(28, 28, 32),
        hazard_fill=(236, 64, 60), hazard_edge=(236, 64, 60),
        player_fill=(40, 120, 240), player_edge=(40, 120, 240),
        ground_fill=(28, 28, 32), ground_line=(28, 28, 32),
        glow=0.0, block_style="flat", hazard_style="solid", bg_style="plain",
        parallax_layers=0, particles=0.0, pulse=0.0,
        pad_fill=(250, 190, 40), orb_fill=(40, 190, 160),
        portal_fill=(150, 90, 230), goal_fill=(20, 20, 24),
        decor_colors=((226, 226, 230), (240, 240, 244), (214, 214, 220)),
        outline_width=0, corner_radius=2, ground_style="flat",
        parallax_shape="blocks", decor_style="bars",
        decor_density=0.0, decor_contrast=0.2,
        bloom=0.0, vignette=0.0, noise=0.0, contrast=1.0, gamma=1.0,
        chromatic=0.0, trail=0.0, pattern_scale=1.0, seed=909,
    ),
    Theme(
        name="jungle",
        bg_top=(18, 62, 38), bg_bottom=(6, 24, 16),
        block_fill=(104, 74, 40), block_edge=(56, 38, 18),
        hazard_fill=(180, 216, 70), hazard_edge=(72, 108, 24),
        player_fill=(250, 236, 150), player_edge=(40, 60, 20),
        ground_fill=(38, 28, 14), ground_line=(96, 160, 60),
        glow=0.15, block_style="noise", hazard_style="solid", bg_style="circles",
        parallax_layers=3, particles=0.5, pulse=0.15,
        pad_fill=(232, 160, 60), orb_fill=(120, 232, 180),
        portal_fill=(96, 200, 210), goal_fill=(240, 250, 220),
        decor_colors=((26, 78, 46), (44, 108, 62), (14, 48, 30)),
        outline_width=1, corner_radius=1, ground_style="dotted",
        parallax_shape="triangles", decor_style="floaters",
        decor_density=0.9, decor_contrast=0.4,
        bloom=0.2, vignette=0.4, noise=0.09, contrast=1.05, gamma=1.05,
        chromatic=0.05, trail=0.35, pattern_scale=1.6, seed=1010,
    ),
    Theme(
        name="glass",
        bg_top=(206, 230, 244), bg_bottom=(150, 186, 214),
        block_fill=(224, 244, 252), block_edge=(96, 148, 180),
        hazard_fill=(210, 234, 244), hazard_edge=(40, 90, 120),
        player_fill=(60, 92, 120), player_edge=(230, 246, 252),
        ground_fill=(170, 202, 224), ground_line=(250, 254, 255),
        glow=0.45, block_style="bevel", hazard_style="outline", bg_style="circles",
        parallax_layers=2, particles=0.35, pulse=0.2,
        pad_fill=(180, 240, 236), orb_fill=(120, 190, 230),
        portal_fill=(140, 170, 220), goal_fill=(255, 255, 255),
        decor_colors=((186, 216, 236), (216, 236, 248), (160, 194, 218)),
        outline_width=1, corner_radius=4, ground_style="gradient",
        parallax_shape="circles", decor_style="crystals",
        decor_density=0.65, decor_contrast=0.3,
        bloom=0.5, vignette=0.18, noise=0.02, contrast=0.95, gamma=1.1,
        chromatic=0.12, trail=0.45, pattern_scale=1.4, seed=1111,
    ),
    Theme(
        name="glitch",
        bg_top=(10, 10, 12), bg_bottom=(26, 4, 26),
        block_fill=(228, 232, 236), block_edge=(0, 255, 120),
        hazard_fill=(255, 0, 90), hazard_edge=(0, 220, 255),
        player_fill=(0, 255, 190), player_edge=(255, 0, 120),
        ground_fill=(20, 20, 24), ground_line=(0, 255, 140),
        glow=0.6, block_style="striped", hazard_style="double", bg_style="noise",
        parallax_layers=1, particles=0.75, pulse=0.95,
        pad_fill=(255, 240, 0), orb_fill=(0, 200, 255),
        portal_fill=(255, 90, 255), goal_fill=(255, 255, 255),
        decor_colors=((48, 0, 48), (0, 70, 60), (90, 90, 96)),
        outline_width=2, corner_radius=0, ground_style="stripes",
        parallax_shape="blocks", decor_style="glyphs",
        decor_density=1.0, decor_contrast=1.0,
        bloom=0.6, vignette=0.35, noise=0.35, contrast=1.45, gamma=0.85,
        chromatic=0.95, trail=0.85, pattern_scale=0.6, seed=1212,
    ),
    Theme(
        name="sunset",
        bg_top=(252, 168, 88), bg_bottom=(96, 40, 120),
        block_fill=(38, 18, 52), block_edge=(255, 190, 120),
        hazard_fill=(28, 10, 40), hazard_edge=(255, 128, 90),
        player_fill=(255, 244, 210), player_edge=(60, 20, 60),
        ground_fill=(30, 12, 44), ground_line=(255, 160, 90),
        glow=0.55, block_style="flat", hazard_style="outline", bg_style="stripes",
        parallax_layers=3, particles=0.3, pulse=0.3,
        pad_fill=(255, 200, 90), orb_fill=(255, 140, 170),
        portal_fill=(150, 110, 220), goal_fill=(255, 250, 230),
        decor_colors=((120, 60, 110), (196, 110, 110), (70, 30, 80)),
        outline_width=1, corner_radius=1, ground_style="flat",
        parallax_shape="mountains", decor_style="waves",
        decor_density=0.6, decor_contrast=0.55,
        bloom=0.5, vignette=0.42, noise=0.03, contrast=1.08, gamma=1.0,
        chromatic=0.08, trail=0.4, pattern_scale=1.0, seed=1313,
    ),
    Theme(
        name="ice",
        bg_top=(226, 246, 255), bg_bottom=(158, 206, 236),
        block_fill=(120, 186, 226), block_edge=(248, 254, 255),
        hazard_fill=(60, 110, 160), hazard_edge=(236, 250, 255),
        player_fill=(255, 255, 255), player_edge=(70, 130, 180),
        ground_fill=(178, 214, 238), ground_line=(255, 255, 255),
        glow=0.35, block_style="bevel", hazard_style="gradient", bg_style="clouds",
        parallax_layers=2, particles=0.55, pulse=0.18,
        pad_fill=(150, 236, 236), orb_fill=(90, 180, 230),
        portal_fill=(110, 150, 220), goal_fill=(255, 255, 255),
        decor_colors=((198, 228, 246), (226, 244, 254), (172, 208, 234)),
        outline_width=1, corner_radius=2, ground_style="grid",
        parallax_shape="triangles", decor_style="waves",
        decor_density=0.7, decor_contrast=0.45,
        bloom=0.45, vignette=0.2, noise=0.03, contrast=1.0, gamma=1.08,
        chromatic=0.06, trail=0.55, pattern_scale=1.3, seed=1414,
    ),
)

THEME_NAMES: tuple[str, ...] = tuple(t.name for t in BUILTIN_THEMES)
_BY_NAME: dict[str, Theme] = {t.name: t for t in BUILTIN_THEMES}


def theme_by_name(name: str) -> Theme:
    """Встроенная тема по имени (регистр не важен).

    Зачем не словарь наружу: у уровня есть `theme_hint`, у CLI — флаг `--theme`,
    и опечатка должна приводить к внятному сообщению со списком доступных имён,
    а не к KeyError где-то в середине рендера.
    """
    key = str(name).strip().lower()
    theme = _BY_NAME.get(key)
    if theme is None:
        raise ValueError(
            f"Неизвестная тема {name!r}. Доступные: {', '.join(THEME_NAMES)}"
        )
    return theme


def theme_names() -> tuple[str, ...]:
    """Имена всех встроенных тем (для CLI, тестов и held-out разбиения зрения)."""
    return THEME_NAMES


# ---------------------------------------------------------------------------
# случайная тема
# ---------------------------------------------------------------------------
# Схемы подбора оттенков. Зачем несколько: чисто равномерный рандом по HSV даёт
# «грязные» и однообразно-серые палитры, а гармонические схемы — узнаваемые
# стили, похожие на настоящие уровни; вместе они покрывают куда больший объём
# визуального пространства, чем каждая по отдельности.
_HUE_SCHEMES: tuple[str, ...] = ("mono", "analog", "complement", "triad", "chaos")


def _scheme_hue(base: float, scheme: str, rng: np.random.Generator) -> float:
    """Оттенок-компаньон к базовому по выбранной цветовой схеме."""
    if scheme == "mono":
        return base + float(rng.uniform(-0.04, 0.04))
    if scheme == "analog":
        return base + float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.06, 0.16))
    if scheme == "complement":
        return base + 0.5 + float(rng.uniform(-0.05, 0.05))
    if scheme == "triad":
        return base + float(rng.choice((1.0 / 3.0, 2.0 / 3.0))) + float(rng.uniform(-0.04, 0.04))
    return float(rng.random())


def random_theme(rng: np.random.Generator, name: str | None = None) -> Theme:
    """Полностью случайная тема — главный двигатель доменной рандомизации.

    Зачем такая структура, а не «каждое поле независимо равномерно»: полностью
    независимый рандом порождает в основном мутно-серые кадры, где всё плохо
    видно, и сеть учится на бесполезном шуме. Здесь палитра строится осмысленно
    (фон/передний план/акценты), а сверху накидываются НАМЕРЕННО ВРЕДНЫЕ
    модификаторы, каждый из которых убивает один «лёгкий» признак:

    * `similar_hazard` (~35%) — шип красится почти как блок: цвет перестаёт
      разделять SOLID и HAZARD, остаётся только треугольная форма;
    * `low_contrast` (~28%) — передний план подтягивается к фону: пропадает
      «граница по яркости»;
    * `monochrome` (~14%) — палитра обесцвечивается целиком: цветовой канал не
      несёт информации вообще;
    * `inverted` (~14%) — палитра инвертируется: «тёмный фон, светлые блоки»
      перестаёт быть надёжным правилом.

    Тряска камеры (`shake`) намеренно всегда 0: кадр обязан совпадать с
    канонической картой пиксель-в-пиксель, а сдвигать можно только оба массива
    сразу — это решение принимает вызывающий код, а не тема.
    """
    seed = int(rng.integers(0, 2**32))
    theme_name = str(name) if name is not None else f"random_{seed:08x}"

    scheme = str(rng.choice(_HUE_SCHEMES))
    h_bg = float(rng.random())
    h_block = _scheme_hue(h_bg, scheme, rng)
    h_hazard = _scheme_hue(h_block, str(rng.choice(_HUE_SCHEMES)), rng)
    h_player = _scheme_hue(h_block, "complement", rng)
    h_accent = float(rng.random())

    dark_bg = bool(rng.random() < 0.62)
    s_bg = float(rng.uniform(0.0, 0.85))
    if dark_bg:
        v_top = float(rng.uniform(0.06, 0.34))
        v_bottom = float(rng.uniform(0.01, max(0.02, v_top)))
    else:
        v_top = float(rng.uniform(0.66, 1.0))
        v_bottom = float(rng.uniform(min(0.62, v_top), 1.0))
    bg_top = hsv_rgb(h_bg, s_bg, v_top)
    bg_bottom = hsv_rgb(h_bg + float(rng.uniform(-0.06, 0.06)), s_bg * float(rng.uniform(0.6, 1.2)), v_bottom)

    # Передний план: по умолчанию уводим в противоположную сторону по яркости.
    s_block = float(rng.uniform(0.0, 0.95))
    if dark_bg:
        v_block = float(rng.uniform(0.45, 1.0))
    else:
        v_block = float(rng.uniform(0.02, 0.5))

    low_contrast = bool(rng.random() < 0.28)
    if low_contrast:
        # Подтягиваем блок к фону: остаётся тонкая разница, но не «чёрное на белом».
        pull = float(rng.uniform(0.55, 0.85))
        v_block = v_block + (v_top - v_block) * pull
        s_block = s_block + (s_bg - s_block) * pull * 0.5
    block_fill = hsv_rgb(h_block, s_block, v_block)

    # Обводка блока: либо светлее, либо темнее заливки — оба варианта встречаются
    # в реальных уровнях, и сеть не должна опираться ни на один из них.
    edge_up = bool(rng.random() < 0.5)
    block_edge = hsv_rgb(
        h_block + float(rng.uniform(-0.08, 0.08)),
        s_block * float(rng.uniform(0.3, 1.1)),
        _clamp(v_block + (0.32 if edge_up else -0.32) * float(rng.uniform(0.5, 1.4)), 0.0, 1.0),
    )

    similar_hazard = bool(rng.random() < 0.35)
    if similar_hazard:
        # Самый вредный случай: шип отличается от блока на грани различимости.
        jitter = float(rng.uniform(0.02, 0.13))
        hazard_fill = hsv_rgb(
            h_block + float(rng.uniform(-0.03, 0.03)),
            _clamp(s_block + float(rng.uniform(-0.1, 0.1)), 0.0, 1.0),
            _clamp(v_block + float(rng.choice((-1.0, 1.0))) * jitter, 0.0, 1.0),
        )
    else:
        hazard_fill = hsv_rgb(
            h_hazard,
            float(rng.uniform(0.3, 1.0)),
            float(rng.uniform(0.35, 1.0)) if dark_bg else float(rng.uniform(0.15, 0.85)),
        )
    hazard_edge = hsv_rgb(
        h_hazard + float(rng.uniform(-0.1, 0.1)),
        float(rng.uniform(0.0, 0.9)),
        _clamp(colorsys.rgb_to_hsv(*[c / 255.0 for c in hazard_fill])[2] + float(rng.choice((-0.3, 0.3))), 0.05, 1.0),
    )

    player_fill = hsv_rgb(h_player, float(rng.uniform(0.1, 1.0)), float(rng.uniform(0.55, 1.0)) if dark_bg else float(rng.uniform(0.1, 0.9)))
    player_edge = hsv_rgb(h_player + float(rng.uniform(-0.1, 0.1)), float(rng.uniform(0.0, 0.9)),
                          _clamp(colorsys.rgb_to_hsv(*[c / 255.0 for c in player_fill])[2] - 0.4, 0.0, 1.0))

    ground_fill = mix_rgb(bg_bottom, block_fill, float(rng.uniform(0.05, 0.55)))
    ground_line = mix_rgb(block_edge, hsv_rgb(h_accent, 0.8, 1.0), float(rng.uniform(0.0, 0.7)))

    pad_fill = hsv_rgb(h_accent, float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.5, 1.0)))
    orb_fill = hsv_rgb(h_accent + 0.5, float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.5, 1.0)))
    portal_fill = hsv_rgb(h_accent + float(rng.uniform(0.2, 0.4)), float(rng.uniform(0.35, 1.0)), float(rng.uniform(0.5, 1.0)))
    goal_fill = hsv_rgb(float(rng.random()), float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.7, 1.0)))

    # Палитра декора. Три стратегии вместо одной: раньше декор ВСЕГДА уводился к
    # цвету фона, и «декорация = бледное пятно» становилось надёжным признаком —
    # ровно тем, на который сеть не должна опираться. Теперь декор бывает и
    # бледным, и кричаще ярким, и покрашенным как настоящий блок.
    decor_contrast = float(rng.random())
    n_decor = int(rng.integers(3, 7))
    decor_list: list[RGB] = []
    for _ in range(n_decor):
        pick = float(rng.random())
        tint = hsv_rgb(
            h_bg + float(rng.uniform(-0.35, 0.35)),
            float(rng.uniform(0.0, 1.0)),
            float(rng.uniform(0.05, 1.0)),
        )
        if pick < 0.45:
            # «Туман»: чуть темнее или светлее фона, еле различим.
            t = float(rng.uniform(0.12, 0.45)) * (0.5 + decor_contrast)
            decor_list.append(mix_rgb(bg_bottom, tint, min(1.0, t)))
        elif pick < 0.78:
            # «Конструкции»: заметные, но своего цвета.
            t = float(rng.uniform(0.5, 1.0))
            decor_list.append(mix_rgb(bg_bottom, tint, t))
        else:
            # Самый вредный вариант: декор покрашен как игровой блок.
            decor_list.append(mix_rgb(block_fill, tint, float(rng.uniform(0.0, 0.45))))
    decor_colors = tuple(decor_list)

    theme = Theme(
        name=theme_name,
        bg_top=bg_top, bg_bottom=bg_bottom,
        block_fill=block_fill, block_edge=block_edge,
        hazard_fill=hazard_fill, hazard_edge=hazard_edge,
        player_fill=player_fill, player_edge=player_edge,
        ground_fill=ground_fill, ground_line=ground_line,
        glow=float(rng.random()),
        block_style=str(rng.choice(BLOCK_STYLES)),
        hazard_style=str(rng.choice(HAZARD_STYLES)),
        bg_style=str(rng.choice(BG_STYLES)),
        parallax_layers=int(rng.integers(0, 4)),
        particles=float(rng.random()),
        pulse=float(rng.random()),
        pad_fill=pad_fill, orb_fill=orb_fill,
        portal_fill=portal_fill, goal_fill=goal_fill,
        decor_colors=decor_colors,
        outline_width=int(rng.integers(0, 4)),
        corner_radius=int(rng.integers(0, 5)),
        ground_style=str(rng.choice(GROUND_STYLES)),
        parallax_shape=str(rng.choice(PARALLAX_SHAPES)),
        decor_style=str(rng.choice(DECOR_STYLES)),
        decor_density=float(rng.random()),
        decor_contrast=decor_contrast,
        bloom=float(rng.random()),
        vignette=float(rng.uniform(0.0, 0.7)),
        noise=float(rng.uniform(0.0, 0.35)),
        contrast=float(rng.uniform(0.75, 1.5)),
        gamma=float(rng.uniform(0.75, 1.5)),
        chromatic=float(rng.uniform(0.0, 0.9)),
        trail=float(rng.random()),
        shake=0.0,
        pattern_scale=float(rng.uniform(0.4, 2.6)),
        seed=seed,
    )

    if rng.random() < 0.14:
        theme = to_monochrome(theme)
    if rng.random() < 0.14:
        theme = to_inverted(theme)
    return theme


def to_monochrome(theme: Theme) -> Theme:
    """Обесцветить тему целиком.

    Зачем отдельной функцией: монохром применяется и к случайным темам, и к
    встроенным (для тестов на устойчивость), и делать это надо одинаково —
    иначе «монохромный вариант неона» окажется другим у рендера и у теста.
    """
    changes: dict[str, Any] = {f: _to_gray(getattr(theme, f)) for f in _COLOR_FIELDS}
    changes["decor_colors"] = tuple(_to_gray(c) for c in theme.decor_colors)
    changes["monochrome"] = True
    changes["name"] = f"{theme.name}_mono"
    return replace(theme, **changes)


def to_inverted(theme: Theme) -> Theme:
    """Инвертировать всю палитру темы (светлое <-> тёмное)."""
    changes: dict[str, Any] = {f: _invert(getattr(theme, f)) for f in _COLOR_FIELDS}
    changes["decor_colors"] = tuple(_invert(c) for c in theme.decor_colors)
    changes["inverted"] = not theme.inverted
    changes["name"] = f"{theme.name}_inv"
    return replace(theme, **changes)


def random_themes(n: int, rng: np.random.Generator) -> tuple[Theme, ...]:
    """`n` независимых случайных тем — удобная обёртка для датасетов и тестов."""
    count = int(n)
    if count < 0:
        raise ValueError(f"Количество тем не может быть отрицательным: {n}")
    return tuple(random_theme(rng) for _ in range(count))


def split_themes(
    holdout: int = 3, rng: np.random.Generator | None = None
) -> tuple[tuple[Theme, ...], tuple[Theme, ...]]:
    """Разбить встроенные темы на «обучающие» и «отложенные».

    Зачем: SPEC §10 требует валидировать зрение на темах, которых не было в
    обучении — только так измеряется обобщение на НОВЫЙ дизайн, а не на новый
    кадр знакомого дизайна. Разбиение живёт здесь, чтобы у датасета и у тестов
    оно было одно и то же.
    """
    k = int(holdout)
    if not 0 <= k < len(BUILTIN_THEMES):
        raise ValueError(f"holdout должен быть в 0..{len(BUILTIN_THEMES) - 1}, получено {holdout}")
    order = np.arange(len(BUILTIN_THEMES))
    if rng is not None:
        order = rng.permutation(order)
    held = tuple(BUILTIN_THEMES[i] for i in order[:k])
    train = tuple(BUILTIN_THEMES[i] for i in order[k:])
    return train, held


__all__ = [
    "RGB",
    "Theme",
    "BUILTIN_THEMES",
    "THEME_NAMES",
    "BLOCK_STYLES",
    "HAZARD_STYLES",
    "BG_STYLES",
    "GROUND_STYLES",
    "PARALLAX_SHAPES",
    "DECOR_STYLES",
    "random_theme",
    "random_themes",
    "theme_by_name",
    "theme_names",
    "to_monochrome",
    "to_inverted",
    "split_themes",
    "mix_rgb",
    "shade_rgb",
    "luminance",
    "contrast_ratio",
    "hsv_rgb",
]
