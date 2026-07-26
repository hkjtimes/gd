"""Формат уровня: объекты, JSON-сериализация и быстрый поиск по x.

Зачем именно так: уровень — это единственный «источник правды», из которого
одновременно строятся (а) физика, (б) каноническая семантическая карта и
(в) красивый рендер с декорациями. Поэтому объект уровня хранит только смысл
(тип, координаты центра), а всё оформление живёт в themes/render и на игру
не влияет.

Система координат: тайлы, x растёт вправо, y растёт ВВЕРХ, пол на y = 0
(`GROUND_Y`). Координаты объекта — это его ЦЕНТР.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from gdai.constants import (
    GOAL,
    HAZARD,
    HAZARD_HALF,
    ORB,
    ORB_HALF,
    PAD,
    PAD_HALF_X,
    PAD_HALF_Y,
    PORTAL_GRAVITY,
    PORTAL_HALF_X,
    PORTAL_HALF_Y,
    PORTAL_MODE,
    PORTAL_SPEED,
    SOLID,
)

LEVEL_FORMAT_VERSION: int = 1

OBJECT_TYPES: tuple[str, ...] = (
    "block", "platform",                                          # SOLID
    "spike", "spike_down", "spike_left", "spike_right", "saw",    # HAZARD
    "pad_yellow", "pad_pink", "pad_red",                          # PAD
    "orb_yellow", "orb_pink", "orb_red",                          # ORB
    "portal_gravity_down", "portal_gravity_up",                   # PORTAL_GRAVITY
    "portal_cube", "portal_ship", "portal_wave",                  # PORTAL_MODE
    "portal_speed_0", "portal_speed_1", "portal_speed_2",
    "portal_speed_3", "portal_speed_4",                           # PORTAL_SPEED
    "goal",                                                       # GOAL
)

# Тип -> семантический класс. Единственное место, где живёт это соответствие:
# и физика, и растеризатор карты обязаны спрашивать здесь.
_TYPE_TO_CLASS: dict[str, int] = {
    "block": SOLID,
    "platform": SOLID,
    "spike": HAZARD,
    "spike_down": HAZARD,
    "spike_left": HAZARD,
    "spike_right": HAZARD,
    "saw": HAZARD,
    "pad_yellow": PAD,
    "pad_pink": PAD,
    "pad_red": PAD,
    "orb_yellow": ORB,
    "orb_pink": ORB,
    "orb_red": ORB,
    "portal_gravity_down": PORTAL_GRAVITY,
    "portal_gravity_up": PORTAL_GRAVITY,
    "portal_cube": PORTAL_MODE,
    "portal_ship": PORTAL_MODE,
    "portal_wave": PORTAL_MODE,
    "portal_speed_0": PORTAL_SPEED,
    "portal_speed_1": PORTAL_SPEED,
    "portal_speed_2": PORTAL_SPEED,
    "portal_speed_3": PORTAL_SPEED,
    "portal_speed_4": PORTAL_SPEED,
    "goal": GOAL,
}

# Полуразмеры хитбокса/фигуры. Хитбокс шипа намеренно сильно меньше картинки —
# так в оригинальной игре, иначе прыжки становятся нечестными.
_SAW_HALF: float = 0.40        # пила крупнее шипа, но всё равно «прощающая»
_PLATFORM_HALF_Y: float = 0.25  # тонкая платформа: 1x0.5 тайла
_GOAL_HALF_Y: float = 6.0       # финиш — вертикальная полоса во весь экран

_TYPE_TO_HALF: dict[str, tuple[float, float]] = {
    "block": (0.5, 0.5),
    "platform": (0.5, _PLATFORM_HALF_Y),
    "spike": (HAZARD_HALF, HAZARD_HALF),
    "spike_down": (HAZARD_HALF, HAZARD_HALF),
    "spike_left": (HAZARD_HALF, HAZARD_HALF),
    "spike_right": (HAZARD_HALF, HAZARD_HALF),
    "saw": (_SAW_HALF, _SAW_HALF),
    "pad_yellow": (PAD_HALF_X, PAD_HALF_Y),
    "pad_pink": (PAD_HALF_X, PAD_HALF_Y),
    "pad_red": (PAD_HALF_X, PAD_HALF_Y),
    "orb_yellow": (ORB_HALF, ORB_HALF),
    "orb_pink": (ORB_HALF, ORB_HALF),
    "orb_red": (ORB_HALF, ORB_HALF),
    "portal_gravity_down": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_gravity_up": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_cube": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_ship": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_wave": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_speed_0": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_speed_1": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_speed_2": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_speed_3": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "portal_speed_4": (PORTAL_HALF_X, PORTAL_HALF_Y),
    "goal": (0.5, _GOAL_HALF_Y),
}

SOLID_TYPES: frozenset[str] = frozenset(t for t, c in _TYPE_TO_CLASS.items() if c == SOLID)
HAZARD_TYPES: frozenset[str] = frozenset(t for t, c in _TYPE_TO_CLASS.items() if c == HAZARD)
PAD_TYPES: frozenset[str] = frozenset(t for t, c in _TYPE_TO_CLASS.items() if c == PAD)
ORB_TYPES: frozenset[str] = frozenset(t for t, c in _TYPE_TO_CLASS.items() if c == ORB)
PORTAL_TYPES: frozenset[str] = frozenset(
    t for t, c in _TYPE_TO_CLASS.items()
    if c in (PORTAL_GRAVITY, PORTAL_MODE, PORTAL_SPEED)
)

# Запас в бакетах при поиске: самый широкий объект — финиш (0.5), но берём с
# запасом, чтобы объект, чей центр лежит в соседнем бакете, не потерялся.
_BUCKET_MARGIN: int = 2

MODES: tuple[str, ...] = ("cube", "ship", "wave")


def type_semantic_class(obj_type: str) -> int:
    """Семантический класс по типу объекта (0..9)."""
    try:
        return _TYPE_TO_CLASS[obj_type]
    except KeyError as exc:
        raise ValueError(
            f"Неизвестный тип объекта {obj_type!r}. Допустимые: {OBJECT_TYPES}"
        ) from exc


def type_half_extent(obj_type: str) -> tuple[float, float]:
    """Полуразмеры (hx, hy) хитбокса объекта по его типу."""
    try:
        return _TYPE_TO_HALF[obj_type]
    except KeyError as exc:
        raise ValueError(
            f"Неизвестный тип объекта {obj_type!r}. Допустимые: {OBJECT_TYPES}"
        ) from exc


@dataclass
class LevelObject:
    """Один игровой объект уровня.

    Зачем хранить только тип и центр: любая другая информация (цвет, поворот,
    свечение) — это декорация, и её нельзя пускать в игровую логику, иначе
    агент начнёт зависеть от оформления.
    """

    type: str
    x: float          # центр объекта в тайлах
    y: float          # центр объекта в тайлах, y растёт вверх, пол при y=0

    def __post_init__(self) -> None:
        if self.type not in _TYPE_TO_CLASS:
            raise ValueError(
                f"Неизвестный тип объекта {self.type!r}. Допустимые: {OBJECT_TYPES}"
            )
        self.x = float(self.x)
        self.y = float(self.y)

    def semantic_class(self) -> int:
        """Класс объекта на канонической карте (см. gdai/constants.py)."""
        return _TYPE_TO_CLASS[self.type]

    def half_extent(self) -> tuple[float, float]:
        """Полуразмеры для хитбокса и растеризации."""
        return _TYPE_TO_HALF[self.type]

    def bounds(self) -> tuple[float, float, float, float]:
        """AABB объекта как (x0, y0, x1, y1) — удобно для растеризации."""
        hx, hy = self.half_extent()
        return (self.x - hx, self.y - hy, self.x + hx, self.y + hy)

    def to_dict(self) -> dict[str, Any]:
        """Компактное представление для JSON."""
        return {"type": self.type, "x": _round(self.x), "y": _round(self.y)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LevelObject":
        """Разбор одного объекта из JSON."""
        return cls(type=str(d["type"]), x=float(d["x"]), y=float(d["y"]))


def _round(value: float) -> float:
    """Округление до 4 знаков.

    Зачем: файлы уровней должны быть человекочитаемыми, а не полными
    «0.30000000000000004»; на физику 1e-4 тайла не влияет.
    """
    return round(float(value), 4)


@dataclass
class Level:
    """Уровень: список объектов + стартовые настройки игрока.

    Зачем бакет-индекс: физика на каждом кадре спрашивает «что рядом со мной»,
    и линейный перебор тысяч объектов сделал бы шаг в сотни раз дороже, чем
    сама физика. Индекс строится по int(x) и обновляется `rebuild_index()`.
    """

    name: str
    length: float                     # длина в тайлах (x финиша)
    objects: list[LevelObject] = field(default_factory=list)
    start_mode: str = "cube"          # cube|ship|wave
    start_speed_index: int = 1
    start_gravity: int = 1            # 1 вниз, -1 вверх
    ceiling_y: float = 12.0
    theme_hint: str | None = None
    checkpoints: list[float] = field(default_factory=list)  # x-координаты для practice
    _buckets: dict[int, list[LevelObject]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.start_mode not in MODES:
            raise ValueError(f"start_mode должен быть одним из {MODES}, а не {self.start_mode!r}")
        if self.start_gravity not in (1, -1):
            raise ValueError("start_gravity: 1 (вниз) или -1 (вверх)")
        self.length = float(self.length)
        self.ceiling_y = float(self.ceiling_y)
        self.start_speed_index = int(self.start_speed_index)
        self.checkpoints = [float(c) for c in self.checkpoints]
        self.rebuild_index()

    # --- индекс -------------------------------------------------------------
    def rebuild_index(self) -> None:
        """Пересобрать бакеты по int(x). Вызывать после любого изменения objects."""
        buckets: dict[int, list[LevelObject]] = {}
        for obj in self.objects:
            buckets.setdefault(math.floor(obj.x), []).append(obj)
        self._buckets = buckets

    def add(self, obj: LevelObject) -> None:
        """Добавить объект и сразу поддержать индекс (без полной пересборки)."""
        self.objects.append(obj)
        self._buckets.setdefault(math.floor(obj.x), []).append(obj)

    def extend(self, objs: Iterable[LevelObject]) -> None:
        """Добавить несколько объектов."""
        for obj in objs:
            self.add(obj)

    def objects_in_range(self, x0: float, x1: float) -> list[LevelObject]:
        """Объекты, чей хитбокс пересекает полосу [x0, x1]. Сложность O(k).

        Зачем полоса, а не точка: физика проверяет коробку игрока, поэтому
        спрашивает диапазон вокруг себя; берём соседние бакеты с запасом,
        так как центр широкого объекта может лежать за границей полосы.
        """
        if x1 < x0:
            x0, x1 = x1, x0
        result: list[LevelObject] = []
        lo = math.floor(x0) - _BUCKET_MARGIN
        hi = math.floor(x1) + _BUCKET_MARGIN
        buckets = self._buckets
        for key in range(lo, hi + 1):
            bucket = buckets.get(key)
            if not bucket:
                continue
            for obj in bucket:
                hx = _TYPE_TO_HALF[obj.type][0]
                if obj.x + hx >= x0 and obj.x - hx <= x1:
                    result.append(obj)
        return result

    # --- сериализация -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-представление уровня (человекочитаемое, версия формата = 1)."""
        return {
            "version": LEVEL_FORMAT_VERSION,
            "name": self.name,
            "length": _round(self.length),
            "start_mode": self.start_mode,
            "start_speed_index": int(self.start_speed_index),
            "start_gravity": int(self.start_gravity),
            "ceiling_y": _round(self.ceiling_y),
            "theme_hint": self.theme_hint,
            "checkpoints": [_round(c) for c in self.checkpoints],
            "objects": [o.to_dict() for o in self.objects],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Level":
        """Собрать уровень из словаря; неизвестная версия — явная ошибка."""
        version = int(d.get("version", LEVEL_FORMAT_VERSION))
        if version != LEVEL_FORMAT_VERSION:
            raise ValueError(
                f"Версия формата уровня {version} не поддерживается "
                f"(ожидалась {LEVEL_FORMAT_VERSION})"
            )
        return cls(
            name=str(d.get("name", "level")),
            length=float(d.get("length", 100.0)),
            objects=[LevelObject.from_dict(o) for o in d.get("objects", [])],
            start_mode=str(d.get("start_mode", "cube")),
            start_speed_index=int(d.get("start_speed_index", 1)),
            start_gravity=int(d.get("start_gravity", 1)),
            ceiling_y=float(d.get("ceiling_y", 12.0)),
            theme_hint=d.get("theme_hint"),
            checkpoints=[float(c) for c in d.get("checkpoints", [])],
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Атомарно записать уровень в JSON (чтобы не терять файл при сбое)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        try:
            tmp.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Level":
        """Прочитать уровень из JSON-файла."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Файл уровня не найден: {target}")
        with target.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # --- прочее -------------------------------------------------------------
    def copy(self) -> "Level":
        """Глубокая копия — чтобы правки уровня не задевали чужие среды."""
        return Level.from_dict(self.to_dict())

    def __len__(self) -> int:
        return len(self.objects)


__all__ = [
    "LEVEL_FORMAT_VERSION",
    "OBJECT_TYPES",
    "MODES",
    "SOLID_TYPES",
    "HAZARD_TYPES",
    "PAD_TYPES",
    "ORB_TYPES",
    "PORTAL_TYPES",
    "LevelObject",
    "Level",
    "type_semantic_class",
    "type_half_extent",
]
