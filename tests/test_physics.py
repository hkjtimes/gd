"""Тесты физики (SPEC §5).

Что здесь проверяется и почему именно это
-----------------------------------------
Физика — единственный модуль, ошибка в котором не видна ни на одном графике:
агент будет исправно учиться, просто игре, которой не существует. Поэтому
тесты держат ровно те свойства, из которых выведена вся геометрия уровней:

* **высота прыжка 2.4 тайла** — от неё считаются все зазоры в генераторе;
* **чистота и детерминизм** — без них не работают ни поиск проходимости, ни
  воспроизводимость обучения по seed;
* **правила взаимодействий** — шип убивает, пад срабатывает сам, кольцо только
  по фронту нажатия, порталы переключают состояние.
"""

from __future__ import annotations

import copy

import pytest

from gdai.constants import (
    DT,
    GRAVITY,
    JUMP_V,
    ORB_RED_V,
    PAD_YELLOW_V,
    PLAYER_HALF,
    SPEED_TILES_PER_SEC,
)
from gdai.env.level import Level, LevelObject
from gdai.env.physics import (
    PlayerState,
    make_initial_state,
    player_half_extent,
    speed_of,
    step_physics,
)


def _simulate(state: PlayerState, level: Level, holds: list[bool]) -> list[PlayerState]:
    """Прогнать последовательность нажатий и вернуть все промежуточные состояния."""
    out: list[PlayerState] = []
    current = state
    for hold in holds:
        current, _events = step_physics(current, level, hold)
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# базовые инварианты
# ---------------------------------------------------------------------------
def test_initial_state_stands_on_ground(flat_level: Level) -> None:
    """Старт: игрок стоит нижней гранью ровно на полу и жив."""
    state = make_initial_state(flat_level)
    assert state.y == pytest.approx(PLAYER_HALF)
    assert state.on_ground is True
    assert state.alive is True and state.finished is False
    assert state.mode == flat_level.start_mode
    assert state.gravity == flat_level.start_gravity


def test_jump_height_is_2_4_tiles(flat_level: Level, start_state: PlayerState) -> None:
    """Высота прыжка ≈ 2.4 тайла — базовая константа геометрии всех уровней."""
    state, events = step_physics(start_state, flat_level, True)
    assert events["jumped"] is True

    apex = state.y
    for _ in range(200):
        state, _ = step_physics(state, flat_level, False)
        apex = max(apex, state.y)
        if state.on_ground:
            break
    assert state.on_ground is True, "куб обязан вернуться на землю"
    assert apex - start_state.y == pytest.approx(2.4, abs=0.02)


def test_jump_flight_time_about_half_second(
    flat_level: Level, start_state: PlayerState
) -> None:
    """Полёт длится 2*JUMP_V/GRAVITY ≈ 0.5 с — из этого считаются все зазоры."""
    state, _ = step_physics(start_state, flat_level, True)
    frames = 1
    while not state.on_ground and frames < 200:
        state, _ = step_physics(state, flat_level, False)
        frames += 1
    expected_frames = (2.0 * JUMP_V / GRAVITY) / DT
    assert frames == pytest.approx(expected_frames, abs=2)


def test_step_physics_is_pure(flat_level: Level, start_state: PlayerState) -> None:
    """Функция не мутирует вход: без этого невозможен откат состояний в поиске."""
    before = copy.deepcopy(start_state)
    new_state, _ = step_physics(start_state, flat_level, True)
    assert start_state == before
    assert new_state is not start_state


def test_step_physics_is_deterministic(flat_level: Level, start_state: PlayerState) -> None:
    """Один и тот же вход даёт бит-в-бит одинаковый выход."""
    holds = [bool(i % 7 < 3) for i in range(240)]
    first = _simulate(start_state, flat_level, holds)
    second = _simulate(start_state, flat_level, holds)
    assert first == second


def test_horizontal_speed_matches_speed_table(
    flat_level: Level, start_state: PlayerState
) -> None:
    """Горизонталь: x += SPEED_TILES_PER_SEC[speed_index] * dt."""
    state, _ = step_physics(start_state, flat_level, False)
    assert state.x - start_state.x == pytest.approx(SPEED_TILES_PER_SEC[1] * DT)
    assert speed_of(start_state) == SPEED_TILES_PER_SEC[1]


# ---------------------------------------------------------------------------
# смертельные взаимодействия
# ---------------------------------------------------------------------------
def test_spike_kills() -> None:
    """Шип убивает: пересечение хитбокса HAZARD = мгновенная смерть."""
    level = Level(name="spike", length=100.0, objects=[LevelObject("spike", 10.0, 0.5)])
    state = make_initial_state(level)
    for _ in range(300):
        state, events = step_physics(state, level, False)
        if events["died"]:
            break
    assert events["died"] is True
    assert state.alive is False
    assert state.x == pytest.approx(10.0, abs=1.0)


def test_dead_player_is_frozen() -> None:
    """Мёртвый игрок не двигается: среда сама решает, когда сбросить эпизод."""
    level = Level(name="spike", length=100.0, objects=[LevelObject("spike", 10.0, 0.5)])
    state = make_initial_state(level)
    while state.alive:
        state, _ = step_physics(state, level, False)
    frozen = copy.deepcopy(state)
    after, events = step_physics(state, level, True)
    assert after.x == frozen.x and after.y == frozen.y and after.vy == frozen.vy
    assert events["died"] is False and after.alive is False


def test_side_collision_with_block_kills() -> None:
    """Удар в стену сбоку — смерть (высокая колонна непроходима)."""
    wall = [LevelObject("block", 10.0, 0.5 + i) for i in range(4)]
    level = Level(name="wall", length=100.0, objects=wall)
    state = make_initial_state(level)
    died = False
    for _ in range(200):
        state, events = step_physics(state, level, False)
        died = died or events["died"]
        if died:
            break
    assert died is True


def test_landing_on_block_sets_on_ground() -> None:
    """Приземление сверху на блок: snap по высоте, vy = 0, on_ground = True."""
    level = Level(name="step", length=100.0, objects=[LevelObject("block", 10.0, 0.5)])
    state = make_initial_state(level)
    landed = False
    for _ in range(300):
        # Прыгаем сильно заранее: к стенке блока (x = 9.05 с учётом хитбоксов)
        # игрок обязан подойти уже выше её верха, иначе это удар в лоб.
        hold = 5.9 <= state.x <= 6.2 and state.on_ground
        state, events = step_physics(state, level, hold)
        if events["died"]:
            pytest.fail(f"игрок умер на x={state.x:.2f}, y={state.y:.2f}")
        if state.on_ground and state.y > 0.9:
            landed = True
            break
    assert landed is True
    assert state.y == pytest.approx(1.0 + PLAYER_HALF, abs=1e-6)
    assert state.vy == pytest.approx(0.0)


def test_finish_at_level_length(flat_level: Level) -> None:
    """`x >= level.length` (или касание финиша) завершает уровень победой."""
    short = Level(name="short", length=3.0, objects=[])
    state = make_initial_state(short)
    for _ in range(300):
        state, events = step_physics(state, short, False)
        if events["finished"]:
            break
    assert events["finished"] is True
    assert state.finished is True and state.alive is True


# ---------------------------------------------------------------------------
# пады и кольца
# ---------------------------------------------------------------------------
def test_pad_triggers_without_press() -> None:
    """Пад срабатывает САМ и подбрасывает ровно на свою скорость."""
    level = Level(name="pad", length=100.0, objects=[LevelObject("pad_yellow", 10.0, 0.25)])
    state = make_initial_state(level)
    start_y = state.y
    apex = state.y
    used = False
    for _ in range(300):
        state, events = step_physics(state, level, False)
        used = used or events["used_pad"]
        apex = max(apex, state.y)
    assert used is True, "пад обязан срабатывать без нажатия"
    expected_height = PAD_YELLOW_V**2 / (2.0 * GRAVITY)
    assert apex - start_y == pytest.approx(expected_height, abs=0.05)


def test_orb_needs_press_edge() -> None:
    """Кольцо срабатывает только по ФРОНТУ нажатия, а не по факту удержания."""
    level = Level(name="orb", length=100.0, objects=[LevelObject("orb_red", 10.0, 0.9)])

    # Постоянно зажатая кнопка: фронт был на первом кадре, далеко от кольца.
    state = make_initial_state(level)
    triggers = 0
    for _ in range(200):
        state, events = step_physics(state, level, True)
        triggers += int(events["used_orb"])
    assert triggers == 0, "зажатая кнопка не должна использовать кольцо"

    # Осмысленное нажатие в момент касания — кольцо задаёт свою скорость.
    state = make_initial_state(level)
    used = False
    for _ in range(200):
        hold = (not used) and abs(state.x - 10.0) < 0.5 and state.on_ground
        state, events = step_physics(state, level, hold)
        if events["used_orb"]:
            used = True
            break
    assert used is True
    # За тот же кадр гравитация уже успела откусить GRAVITY*dt.
    assert state.vy == pytest.approx(ORB_RED_V - GRAVITY * DT, abs=1e-6)


def test_orb_beats_pad_on_same_frame() -> None:
    """Кольцо перебивает пад: игрок нажал осознанно (SPEC §5.2).

    Состояние собирается вручную ровно в точке касания обоих объектов —
    подгадать такой кадр «прогоном с начала» невозможно, потому что пад
    срабатывает сам и уносит игрока раньше.
    """
    level = Level(
        name="both",
        length=100.0,
        objects=[LevelObject("pad_yellow", 10.0, 0.25), LevelObject("orb_red", 10.0, 0.9)],
    )
    state = PlayerState(
        x=10.0, y=PLAYER_HALF, vy=0.0, mode="cube", gravity=1, speed_index=1,
        on_ground=True, alive=True, finished=False, hold_prev=False,
    )
    state, events = step_physics(state, level, True)
    assert events["used_pad"] is True and events["used_orb"] is True
    assert state.vy == pytest.approx(ORB_RED_V - GRAVITY * DT, abs=1e-6)


# ---------------------------------------------------------------------------
# порталы
# ---------------------------------------------------------------------------
def test_gravity_portal_flips_and_lands_on_ceiling() -> None:
    """Портал гравитации переворачивает мир: игрок «падает» на потолок."""
    level = Level(
        name="grav",
        length=100.0,
        ceiling_y=12.0,
        objects=[LevelObject("portal_gravity_up", 10.0, 1.25)],
    )
    state = make_initial_state(level)
    flipped = False
    for _ in range(400):
        state, events = step_physics(state, level, False)
        if events["portal"] == "portal_gravity_up":
            flipped = True
            assert state.gravity == -1
    assert flipped is True
    assert state.alive is True
    assert state.on_ground is True
    assert state.y == pytest.approx(level.ceiling_y - PLAYER_HALF, abs=1e-6)


def test_mode_portal_switches_mode_and_hitbox() -> None:
    """Портал режима меняет и режим, и хитбокс игрока."""
    level = Level(
        name="ship",
        length=100.0,
        objects=[LevelObject("portal_ship", 10.0, 1.25)],
    )
    state = make_initial_state(level)
    for _ in range(200):
        state, events = step_physics(state, level, False)
        if events["portal"] == "portal_ship":
            break
    assert state.mode == "ship"
    assert player_half_extent(state.mode) == player_half_extent("ship")

    level_wave = Level(
        name="wave",
        length=100.0,
        objects=[LevelObject("portal_wave", 10.0, 1.25)],
    )
    state = make_initial_state(level_wave)
    for _ in range(200):
        state, events = step_physics(state, level_wave, True)
        if events["portal"] == "portal_wave":
            break
    assert state.mode == "wave"
    assert player_half_extent("wave")[0] < player_half_extent("cube")[0]


def test_speed_portal_changes_horizontal_speed() -> None:
    """Портал скорости переключает индекс скорости и, значит, шаг по x."""
    level = Level(
        name="speed",
        length=200.0,
        objects=[LevelObject("portal_speed_3", 10.0, 1.25)],
    )
    state = make_initial_state(level)
    assert state.speed_index == 1
    for _ in range(200):
        prev_x = state.x
        state, events = step_physics(state, level, False)
        if events["portal"] == "portal_speed_3":
            assert state.speed_index == 3
            # Сам кадр портала ещё едет на старой скорости — новая с следующего.
            assert state.x - prev_x == pytest.approx(SPEED_TILES_PER_SEC[1] * DT)
            state, _ = step_physics(state, level, False)
            break
    assert state.speed_index == 3


# ---------------------------------------------------------------------------
# режимы корабля и волны
# ---------------------------------------------------------------------------
def test_ship_rises_while_held_and_falls_when_released() -> None:
    """Корабль: удержание — тяга вверх, отпускание — падение вниз."""
    level = Level(name="ship", length=200.0, objects=[], ceiling_y=12.0)
    level.start_mode = "ship"
    state = make_initial_state(level)
    up = state
    for _ in range(20):
        up, _ = step_physics(up, level, True)
    assert up.vy > 0.0 and up.y > state.y

    down = up
    for _ in range(40):
        down, _ = step_physics(down, level, False)
    assert down.vy < 0.0


def test_wave_dies_on_floor_contact() -> None:
    """Волна умирает от касания пола — этим она и отличается от куба."""
    level = Level(name="wave", length=200.0, objects=[], ceiling_y=12.0)
    level.start_mode = "wave"
    state = make_initial_state(level)
    assert state.on_ground is False
    died = False
    for _ in range(400):
        state, events = step_physics(state, level, False)
        if events["died"]:
            died = True
            break
    assert died is True, "волна обязана погибнуть, уткнувшись в пол"


def test_wave_moves_diagonally() -> None:
    """Волна идёт строго по диагонали: |dy| == |dx| при WAVE_SPEED_RATIO = 1."""
    level = Level(name="wave", length=200.0, objects=[], ceiling_y=12.0)
    level.start_mode = "wave"
    state = make_initial_state(level)
    prev = state
    state, _ = step_physics(state, level, True)
    assert abs(state.y - prev.y) == pytest.approx(abs(state.x - prev.x), rel=1e-6)
