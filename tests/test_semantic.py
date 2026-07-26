"""Тесты канонической карты (SPEC §7).

Карта — ground truth всего проекта: на ней учится политика и ею размечается
датасет зрения. Поэтому проверяется не «красиво ли получилось», а свойства, от
нарушения которых обучение молча портится:

* игрок ВСЕГДА присутствует на карте (карта без игрока для политики пуста);
* значения строго в диапазоне классов;
* приоритет отрисовки: шип поверх блока, игрок поверх всего;
* камера чистая и детерминированная — рендер декораций обязан использовать
  ту же самую, иначе кадр и разметка разъедутся.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdai.constants import (
    EMPTY,
    GOAL,
    GROUND_Y,
    HAZARD,
    NUM_CLASSES,
    OBS_H,
    OBS_W,
    PLAYER,
    PLAYER_X_IN_VIEW,
    PX_PER_TILE,
    SOLID,
)
from gdai.env.level import Level, LevelObject, OBJECT_TYPES
from gdai.env.physics import PlayerState
from gdai.env.semantic import (
    camera_origin,
    class_priority,
    downsample_semantic,
    render_semantic,
    semantic_to_rgb,
    view_bounds,
    world_to_pixel,
)


def _random_states(rng: np.random.Generator, count: int) -> list[PlayerState]:
    """Пёстрый набор состояний: разные режимы, гравитация, высоты и скорости."""
    modes = ("cube", "ship", "wave")
    return [
        PlayerState(
            x=float(rng.uniform(0.0, 200.0)),
            y=float(rng.uniform(-2.0, 14.0)),
            vy=float(rng.uniform(-30.0, 30.0)),
            mode=str(modes[int(rng.integers(len(modes)))]),
            gravity=int(rng.choice([-1, 1])),
            speed_index=int(rng.integers(0, 5)),
            on_ground=bool(rng.integers(2)),
        )
        for _ in range(count)
    ]


def _busy_level(rng: np.random.Generator, count: int = 250) -> Level:
    """Уровень, забитый объектами всех типов — чтобы работали все ветки фигур."""
    objects = [
        LevelObject(
            str(OBJECT_TYPES[int(rng.integers(len(OBJECT_TYPES)))]),
            float(rng.uniform(0.0, 210.0)),
            float(rng.uniform(0.0, 11.0)),
        )
        for _ in range(count)
    ]
    return Level(name="busy", length=210.0, objects=objects, ceiling_y=12.0)


# ---------------------------------------------------------------------------
# форма и диапазон
# ---------------------------------------------------------------------------
def test_shape_and_dtype(demo_level: Level, demo_state: PlayerState) -> None:
    """Карта — uint8 (OBS_H, OBS_W), как ждут и политика, и U-Net."""
    sem = render_semantic(demo_level, demo_state)
    assert sem.shape == (OBS_H, OBS_W)
    assert sem.dtype == np.uint8


def test_custom_view_size(demo_level: Level, demo_state: PlayerState) -> None:
    """Нестандартный размер кадра поддерживается (нужно вьюеру и отладке)."""
    sem = render_semantic(demo_level, demo_state, 64, 40)
    assert sem.shape == (40, 64)
    with pytest.raises(ValueError):
        render_semantic(demo_level, demo_state, 0, 40)


def test_classes_are_in_range_and_player_present(rng: np.random.Generator) -> None:
    """На любой карте есть игрок, и ни одного значения вне 0..NUM_CLASSES-1."""
    level = _busy_level(rng)
    for state in _random_states(rng, 150):
        sem = render_semantic(level, state)
        assert int(sem.max()) < NUM_CLASSES
        assert (sem == PLAYER).any(), f"игрока нет на карте при state={state}"


def test_player_box_matches_hitbox(demo_level: Level) -> None:
    """Коробка игрока на карте — это его хитбокс, а не декоративный спрайт."""
    state = PlayerState(x=10.0, y=3.0, mode="cube", on_ground=False)
    sem = render_semantic(demo_level, state)
    rows, cols = np.nonzero(sem == PLAYER)
    height_tiles = (rows.max() - rows.min() + 1) / PX_PER_TILE
    width_tiles = (cols.max() - cols.min() + 1) / PX_PER_TILE
    assert height_tiles == pytest.approx(0.9, abs=0.15)
    assert width_tiles == pytest.approx(0.9, abs=0.15)


# ---------------------------------------------------------------------------
# приоритеты
# ---------------------------------------------------------------------------
def test_priority_order_matches_spec() -> None:
    """EMPTY < SOLID < PORTAL_* < PAD < ORB < GOAL < HAZARD < PLAYER (SPEC §7)."""
    from gdai.constants import ORB, PAD, PORTAL_GRAVITY, PORTAL_MODE, PORTAL_SPEED

    assert class_priority(EMPTY) < class_priority(SOLID)
    for portal in (PORTAL_GRAVITY, PORTAL_MODE, PORTAL_SPEED):
        assert class_priority(SOLID) < class_priority(portal)
        assert class_priority(portal) < class_priority(PAD)
    assert class_priority(PAD) < class_priority(ORB) < class_priority(GOAL)
    assert class_priority(GOAL) < class_priority(HAZARD) < class_priority(PLAYER)
    assert class_priority(PLAYER) == max(class_priority(c) for c in range(NUM_CLASSES))
    with pytest.raises(ValueError):
        class_priority(NUM_CLASSES)


def test_hazard_drawn_over_solid() -> None:
    """Шип на блоке виден как шип: иначе политика примет опасность за опору."""
    level = Level(
        name="stack",
        length=50.0,
        objects=[LevelObject("block", 10.0, 0.5), LevelObject("spike", 10.0, 0.5)],
    )
    state = PlayerState(x=8.0, y=0.45, on_ground=True)
    sem = render_semantic(level, state)
    cam = camera_origin(state)
    px, py = world_to_pixel(10.0, 0.5, cam)
    assert sem[py, px] == HAZARD


def test_player_drawn_over_everything() -> None:
    """Игрок перекрывает даже шип — карта без игрока бесполезна."""
    level = Level(name="overlap", length=50.0, objects=[LevelObject("spike", 10.0, 0.5)])
    state = PlayerState(x=10.0, y=0.5, on_ground=True)
    sem = render_semantic(level, state)
    cam = camera_origin(state)
    px, py = world_to_pixel(10.0, 0.5, cam)
    assert sem[py, px] == PLAYER


# ---------------------------------------------------------------------------
# камера и мир
# ---------------------------------------------------------------------------
def test_camera_is_pure_and_deterministic() -> None:
    """`camera_origin` зависит только от состояния — это её главный контракт."""
    state = PlayerState(x=17.3, y=2.5, vy=-3.0)
    assert camera_origin(state) == camera_origin(state)
    same = PlayerState(x=17.3, y=2.5, vy=99.0, speed_index=4, mode="ship")
    assert camera_origin(state)[0] == camera_origin(same)[0]


def test_camera_keeps_player_at_fixed_column() -> None:
    """По X игрок всегда стоит на PLAYER_X_IN_VIEW от левого края."""
    for x in (0.0, 5.0, 123.75):
        state = PlayerState(x=x, y=0.45, on_ground=True)
        cam_x, _cam_y = camera_origin(state)
        assert cam_x == pytest.approx(x - PLAYER_X_IN_VIEW)
        px, _py = world_to_pixel(x, 0.45, (cam_x, _cam_y))
        assert px == int(PLAYER_X_IN_VIEW * PX_PER_TILE)


def test_camera_stays_low_for_ordinary_jump() -> None:
    """Обычный прыжок камеру не двигает — пол остаётся неподвижным ориентиром."""
    ground = camera_origin(PlayerState(x=10.0, y=0.45, on_ground=True))
    apex = camera_origin(PlayerState(x=10.0, y=2.85))
    assert ground[1] == pytest.approx(apex[1], abs=1e-9)
    high = camera_origin(PlayerState(x=10.0, y=9.0))
    assert high[1] > ground[1], "на большой высоте камера обязана поехать за игроком"


def test_floor_and_ceiling_are_solid(demo_level: Level) -> None:
    """Пол и потолок мира — сплошной SOLID до края кадра, как твёрдые поверхности."""
    state = PlayerState(x=10.0, y=0.45, on_ground=True)
    sem = render_semantic(demo_level, state)
    cam = camera_origin(state)
    _px, py = world_to_pixel(10.0, GROUND_Y - 0.5, cam)
    assert (sem[py, :] == SOLID).all()

    high = PlayerState(x=10.0, y=11.0, mode="ship")
    sem_high = render_semantic(demo_level, high)
    cam_high = camera_origin(high)
    _px, py = world_to_pixel(10.0, demo_level.ceiling_y + 0.5, cam_high)
    if 0 <= py < OBS_H:
        assert (sem_high[py, :] == SOLID).all()


def test_view_bounds_match_camera(demo_state: PlayerState) -> None:
    """Видимый прямоугольник считается из той же камеры и того же масштаба."""
    cam = camera_origin(demo_state)
    x0, y0, x1, y1 = view_bounds(cam)
    assert (x0, y0) == cam
    assert x1 - x0 == pytest.approx(OBS_W / PX_PER_TILE)
    assert y1 - y0 == pytest.approx(OBS_H / PX_PER_TILE)


def test_objects_outside_view_do_not_appear() -> None:
    """Объект далеко за кадром не рисуется — камера действительно обрезает мир."""
    level = Level(name="far", length=300.0, objects=[LevelObject("spike", 250.0, 0.5)])
    state = PlayerState(x=10.0, y=0.45, on_ground=True)
    sem = render_semantic(level, state)
    assert not (sem == HAZARD).any()


# ---------------------------------------------------------------------------
# производные представления
# ---------------------------------------------------------------------------
def test_semantic_to_rgb(demo_level: Level, demo_state: PlayerState) -> None:
    """Раскраска даёт (H, W, 3) uint8 и отвергает битые карты."""
    sem = render_semantic(demo_level, demo_state)
    rgb = semantic_to_rgb(sem)
    assert rgb.shape == (OBS_H, OBS_W, 3)
    assert rgb.dtype == np.uint8
    with pytest.raises(ValueError):
        semantic_to_rgb(np.full((4, 4), NUM_CLASSES, dtype=np.uint8))
    with pytest.raises(ValueError):
        semantic_to_rgb(np.zeros((4, 4, 3), dtype=np.uint8))


def test_downsample_keeps_dangerous_classes() -> None:
    """Сжатие отдаёт блок самому приоритетному классу: шип не имеет права исчезнуть."""
    sem = np.zeros((4, 4), dtype=np.uint8)
    sem[0, 0] = HAZARD
    sem[1, 1] = SOLID
    small = downsample_semantic(sem, 2)
    assert small.shape == (2, 2)
    assert small[0, 0] == HAZARD

    sem2 = np.full((2, 2), SOLID, dtype=np.uint8)
    sem2[1, 1] = PLAYER
    assert downsample_semantic(sem2, 2)[0, 0] == PLAYER


def test_downsample_shape_and_errors(demo_level: Level, demo_state: PlayerState) -> None:
    """Сжатие вдвое даёт 36x64, а неделимый размер — явную ошибку."""
    sem = render_semantic(demo_level, demo_state)
    assert downsample_semantic(sem, 2).shape == (OBS_H // 2, OBS_W // 2)
    assert downsample_semantic(sem, 1).shape == sem.shape
    with pytest.raises(ValueError):
        downsample_semantic(sem, 5)
    with pytest.raises(ValueError):
        downsample_semantic(sem, 0)
    with pytest.raises(ValueError):
        downsample_semantic(np.zeros((2, 2, 2), dtype=np.uint8), 2)


def test_render_is_deterministic(demo_level: Level, demo_state: PlayerState) -> None:
    """Одинаковый вход — байт-в-байт одинаковая карта (разметка не имеет права плыть)."""
    first = render_semantic(demo_level, demo_state)
    second = render_semantic(demo_level, demo_state)
    assert np.array_equal(first, second)
