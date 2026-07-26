"""Канонический словарь проекта.

Зачем: все модули (физика, растеризация карты, рендер, сети, тесты) обязаны
говорить на одном языке — одинаковые номера семантических классов, одинаковые
размеры камеры и одинаковые физические константы. Любое расхождение здесь
мгновенно ломает связку «зрение -> политика», поэтому числа собраны в одном
месте и больше нигде не дублируются.

Единицы измерения: длина — тайлы (1 тайл == 1 блок Geometry Dash == 30 px
в оригинале), время — секунды, кадр — 1/60 с.
"""

from __future__ import annotations

# --- семантические классы (порядок фиксирован, менять нельзя) ---------------
# Зачем фиксирован: индекс класса — это индекс канала на выходе U-Net и
# значение пикселя в разметке; перестановка обесценит все сохранённые веса.
EMPTY: int = 0           # пустота / фон / любая декорация
SOLID: int = 1           # блок, платформа, пол — на него можно приземлиться
HAZARD: int = 2          # шип, пила, любой мгновенно убивающий объект
PLAYER: int = 3          # сам игрок
PAD: int = 4             # жёлтый/розовый/красный трамплин (срабатывает сам)
ORB: int = 5             # кольцо (срабатывает по нажатию)
PORTAL_GRAVITY: int = 6  # портал смены гравитации
PORTAL_MODE: int = 7     # портал смены режима (куб/корабль/волна)
PORTAL_SPEED: int = 8    # портал смены скорости
GOAL: int = 9            # финиш

NUM_CLASSES: int = 10

CLASS_NAMES: tuple[str, ...] = (
    "empty",
    "solid",
    "hazard",
    "player",
    "pad",
    "orb",
    "portal_gravity",
    "portal_mode",
    "portal_speed",
    "goal",
)

# Цвета нужны только человеку: ими раскрашивается каноническая карта в окне
# «что видит нейросеть». Подобраны максимально различимыми, чтобы на глаз
# ловить ошибки сегментации (например, шип, принятый за блок).
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (18, 18, 24),      # EMPTY
    (200, 205, 215),   # SOLID
    (230, 60, 60),     # HAZARD
    (255, 220, 60),    # PLAYER
    (255, 140, 220),   # PAD
    (80, 220, 255),    # ORB
    (80, 120, 255),    # PORTAL_GRAVITY
    (170, 90, 240),    # PORTAL_MODE
    (60, 220, 160),    # PORTAL_SPEED
    (255, 255, 255),   # GOAL
)

# --- геометрия мира ---------------------------------------------------------
TILE: float = 1.0            # 1 тайл мира == 1 блок GD (в оригинале 30 px)
VIEW_TILES_W: int = 16       # ширина камеры в тайлах
VIEW_TILES_H: int = 9        # высота камеры в тайлах
PX_PER_TILE: int = 8         # пикселей на тайл в наблюдении
OBS_W: int = VIEW_TILES_W * PX_PER_TILE   # 128
OBS_H: int = VIEW_TILES_H * PX_PER_TILE   # 72
PLAYER_X_IN_VIEW: float = 4.0   # игрок стоит на 4-м тайле от левого края камеры
GROUND_Y: float = 0.0           # уровень пола (низ игрока при y=0)

# --- физика (тайлы и секунды), dt = 1/60 ------------------------------------
DT: float = 1.0 / 60.0
GRAVITY: float = 76.8    # тайл/с^2, куб
JUMP_V: float = 19.2     # тайл/с, старт прыжка (высота ~2.4 тайла, полёт ~0.5 с)
MAX_FALL_V: float = 30.0
SPEEDS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)          # индексы 0..4
SPEED_TILES_PER_SEC: tuple[float, ...] = (8.36, 10.386, 12.914, 15.6, 19.2)
DEFAULT_SPEED_INDEX: int = 1

PAD_YELLOW_V: float = 24.0
PAD_PINK_V: float = 15.0
PAD_RED_V: float = 30.0
ORB_YELLOW_V: float = 19.2
ORB_PINK_V: float = 13.0
ORB_RED_V: float = 26.0

SHIP_THRUST: float = 52.0    # тайл/с^2 вверх при удержании
SHIP_GRAVITY: float = 52.0
SHIP_MAX_V: float = 16.0
WAVE_SPEED_RATIO: float = 1.0   # волна движется по диагонали 45°

# --- хитбоксы (полуразмеры относительно центра) -----------------------------
PLAYER_HALF: float = 0.45        # куб 0.9x0.9
PLAYER_HALF_SHIP: float = 0.45
PLAYER_HALF_WAVE: float = 0.25
HAZARD_HALF: float = 0.28        # хитбокс шипа сильно меньше картинки — как в GD
ORB_HALF: float = 0.60           # кольцо ловит щедро
PAD_HALF_X: float = 0.5
PAD_HALF_Y: float = 0.25
PORTAL_HALF_X: float = 0.35
PORTAL_HALF_Y: float = 1.25

# --- действия ---------------------------------------------------------------
ACTION_NONE: int = 0
ACTION_HOLD: int = 1
NUM_ACTIONS: int = 2

__all__ = [
    "EMPTY", "SOLID", "HAZARD", "PLAYER", "PAD", "ORB",
    "PORTAL_GRAVITY", "PORTAL_MODE", "PORTAL_SPEED", "GOAL",
    "NUM_CLASSES", "CLASS_NAMES", "CLASS_COLORS",
    "TILE", "VIEW_TILES_W", "VIEW_TILES_H", "PX_PER_TILE", "OBS_W", "OBS_H",
    "PLAYER_X_IN_VIEW", "GROUND_Y",
    "DT", "GRAVITY", "JUMP_V", "MAX_FALL_V", "SPEEDS", "SPEED_TILES_PER_SEC",
    "DEFAULT_SPEED_INDEX",
    "PAD_YELLOW_V", "PAD_PINK_V", "PAD_RED_V",
    "ORB_YELLOW_V", "ORB_PINK_V", "ORB_RED_V",
    "SHIP_THRUST", "SHIP_GRAVITY", "SHIP_MAX_V", "WAVE_SPEED_RATIO",
    "PLAYER_HALF", "PLAYER_HALF_SHIP", "PLAYER_HALF_WAVE", "HAZARD_HALF",
    "ORB_HALF", "PAD_HALF_X", "PAD_HALF_Y", "PORTAL_HALF_X", "PORTAL_HALF_Y",
    "ACTION_NONE", "ACTION_HOLD", "NUM_ACTIONS",
]
