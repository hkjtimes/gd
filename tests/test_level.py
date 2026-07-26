"""Тесты формата уровня (SPEC §4).

Уровень — единственный «источник правды» для физики, карты и рендера, поэтому
здесь проверяются ровно два свойства: он переживает запись-чтение без потерь и
его пространственный индекс отвечает то же самое, что честный перебор. Ошибка
в индексе не падает — она просто делает часть объектов невидимой для физики,
и уровень тихо становится другим.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gdai.constants import (
    GOAL,
    HAZARD,
    NUM_CLASSES,
    ORB,
    PAD,
    PORTAL_GRAVITY,
    PORTAL_MODE,
    PORTAL_SPEED,
    SOLID,
)
from gdai.env.level import (
    LEVEL_FORMAT_VERSION,
    OBJECT_TYPES,
    Level,
    LevelObject,
    type_half_extent,
    type_semantic_class,
)


def _brute_force_range(level: Level, x0: float, x1: float) -> list[tuple[str, float, float]]:
    """Честный перебор всех объектов — эталон для проверки бакет-индекса."""
    result = []
    for obj in level.objects:
        hx = obj.half_extent()[0]
        if obj.x + hx >= x0 and obj.x - hx <= x1:
            result.append((obj.type, obj.x, obj.y))
    return sorted(result)


def _random_level(rng: np.random.Generator, count: int = 300) -> Level:
    """Уровень со случайной россыпью объектов всех типов."""
    objects = [
        LevelObject(
            str(rng.choice(OBJECT_TYPES)),
            float(rng.uniform(0.0, 250.0)),
            float(rng.uniform(0.0, 11.0)),
        )
        for _ in range(count)
    ]
    return Level(name="random", length=250.0, objects=objects, ceiling_y=12.0)


# ---------------------------------------------------------------------------
# объекты
# ---------------------------------------------------------------------------
def test_all_object_types_have_class_and_extent() -> None:
    """Каждый тип из контракта знает свой класс и полуразмеры."""
    for obj_type in OBJECT_TYPES:
        cls = type_semantic_class(obj_type)
        assert 0 <= cls < NUM_CLASSES
        hx, hy = type_half_extent(obj_type)
        assert hx > 0.0 and hy > 0.0
        obj = LevelObject(obj_type, 1.0, 2.0)
        assert obj.semantic_class() == cls
        assert obj.half_extent() == (hx, hy)


def test_object_type_to_class_mapping() -> None:
    """Соответствие «тип -> семантический класс» из SPEC §4 соблюдено."""
    expected = {
        "block": SOLID,
        "platform": SOLID,
        "spike": HAZARD,
        "spike_down": HAZARD,
        "spike_left": HAZARD,
        "spike_right": HAZARD,
        "saw": HAZARD,
        "pad_yellow": PAD,
        "orb_pink": ORB,
        "portal_gravity_up": PORTAL_GRAVITY,
        "portal_wave": PORTAL_MODE,
        "portal_speed_4": PORTAL_SPEED,
        "goal": GOAL,
    }
    for obj_type, cls in expected.items():
        assert LevelObject(obj_type, 0.0, 0.0).semantic_class() == cls


def test_unknown_object_type_raises() -> None:
    """Опечатка в типе — явная ошибка, а не тихо «пустой» объект."""
    with pytest.raises(ValueError, match="Неизвестный тип"):
        LevelObject("spike_diagonal", 1.0, 1.0)


def test_object_bounds_match_half_extent() -> None:
    """AABB объекта строится ровно из его полуразмеров."""
    obj = LevelObject("block", 4.0, 2.0)
    hx, hy = obj.half_extent()
    assert obj.bounds() == (4.0 - hx, 2.0 - hy, 4.0 + hx, 2.0 + hy)


# ---------------------------------------------------------------------------
# сериализация
# ---------------------------------------------------------------------------
def test_dict_round_trip(demo_level: Level) -> None:
    """`to_dict` -> `from_dict` не теряет ни одного поля."""
    demo_level.checkpoints = [7.0, 19.5]
    demo_level.theme_hint = "neon"
    restored = Level.from_dict(demo_level.to_dict())
    assert restored.to_dict() == demo_level.to_dict()
    assert restored.name == demo_level.name
    assert restored.length == demo_level.length
    assert len(restored.objects) == len(demo_level.objects)
    assert restored.checkpoints == demo_level.checkpoints
    assert restored.theme_hint == "neon"


def test_json_file_round_trip(tmp_path: Path) -> None:
    """`save`/`load` дают тот же уровень, а файл остаётся человекочитаемым."""
    level = Level(
        name="saved",
        length=88.5,
        objects=[LevelObject("spike", 10.0, 0.5), LevelObject("goal", 88.5, 6.0)],
        start_mode="ship",
        start_speed_index=3,
        start_gravity=-1,
        ceiling_y=10.0,
        theme_hint="lava",
        checkpoints=[20.0, 40.0],
    )
    path = level.save(tmp_path / "sub" / "level.json")
    assert path.exists()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == LEVEL_FORMAT_VERSION
    assert raw["name"] == "saved"
    assert isinstance(raw["objects"], list) and raw["objects"][0]["type"] == "spike"

    loaded = Level.load(path)
    assert loaded.to_dict() == level.to_dict()
    assert loaded.start_mode == "ship"
    assert loaded.start_gravity == -1
    assert loaded.start_speed_index == 3
    assert loaded.ceiling_y == 10.0


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Отсутствующий файл — понятная ошибка, а не пустой уровень."""
    with pytest.raises(FileNotFoundError):
        Level.load(tmp_path / "nope.json")


def test_unsupported_version_raises(demo_level: Level) -> None:
    """Чужая версия формата отвергается явно."""
    payload = demo_level.to_dict()
    payload["version"] = LEVEL_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="[Вв]ерсия"):
        Level.from_dict(payload)


def test_invalid_start_settings_raise() -> None:
    """Недопустимые стартовые настройки ловятся при создании уровня."""
    with pytest.raises(ValueError):
        Level(name="bad", length=10.0, objects=[], start_mode="rocket")
    with pytest.raises(ValueError):
        Level(name="bad", length=10.0, objects=[], start_gravity=0)


def test_copy_is_deep(demo_level: Level) -> None:
    """`copy()` не делит объекты с оригиналом — среды не должны мешать друг другу."""
    clone = demo_level.copy()
    clone.objects[0].x += 100.0
    clone.rebuild_index()
    assert demo_level.objects[0].x != clone.objects[0].x
    assert demo_level.to_dict() != clone.to_dict()


# ---------------------------------------------------------------------------
# индекс
# ---------------------------------------------------------------------------
def test_objects_in_range_matches_brute_force(rng: np.random.Generator) -> None:
    """Бакет-индекс обязан отвечать ровно то же, что честный перебор."""
    level = _random_level(rng)
    for _ in range(300):
        x0 = float(rng.uniform(-10.0, 260.0))
        x1 = x0 + float(rng.uniform(0.0, 25.0))
        got = sorted((o.type, o.x, o.y) for o in level.objects_in_range(x0, x1))
        assert got == _brute_force_range(level, x0, x1), f"диапазон [{x0}, {x1}]"


def test_objects_in_range_accepts_reversed_bounds(demo_level: Level) -> None:
    """Перепутанные границы не должны молча возвращать пустоту."""
    direct = demo_level.objects_in_range(8.0, 16.0)
    reversed_ = demo_level.objects_in_range(16.0, 8.0)
    assert {id(o) for o in direct} == {id(o) for o in reversed_}
    assert direct


def test_objects_in_range_is_local(rng: np.random.Generator) -> None:
    """Запрос узкой полосы возвращает O(k), а не весь уровень.

    Зачем проверять: индекс существует ровно ради этого — физика спрашивает
    окрестность игрока на каждом кадре, и линейный перебор тысяч объектов
    стоил бы дороже самой физики.
    """
    level = _random_level(rng, count=2000)
    near = level.objects_in_range(100.0, 101.0)
    assert len(near) < len(level.objects) / 10
    assert near == [o for o in near if abs(o.x - 100.5) < 6.0]


def test_add_keeps_index_in_sync(demo_level: Level) -> None:
    """`add` поддерживает индекс без полной пересборки."""
    before = len(demo_level.objects_in_range(49.0, 51.0))
    demo_level.add(LevelObject("spike", 50.0, 0.5))
    after = demo_level.objects_in_range(49.0, 51.0)
    assert len(after) == before + 1
    assert any(o.type == "spike" and o.x == 50.0 for o in after)


def test_rebuild_index_after_direct_mutation(demo_level: Level) -> None:
    """Прямое изменение `objects` требует `rebuild_index` — и он всё чинит."""
    demo_level.objects.append(LevelObject("block", 60.0, 0.5))
    assert not demo_level.objects_in_range(59.0, 61.0)
    demo_level.rebuild_index()
    assert len(demo_level.objects_in_range(59.0, 61.0)) == 1


def test_extend_and_len(demo_level: Level) -> None:
    """`extend` добавляет пачку объектов и обновляет длину коллекции."""
    n = len(demo_level)
    demo_level.extend([LevelObject("spike", 40.0, 0.5), LevelObject("spike", 41.0, 0.5)])
    assert len(demo_level) == n + 2
    assert len(demo_level.objects_in_range(39.5, 41.5)) == 2
