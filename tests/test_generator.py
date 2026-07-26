"""Тесты процедурного генератора (SPEC §6).

Главное требование одно: **сгенерированный уровень обязан быть проходимым**.
Агент, который умирает не из-за своей ошибки, а потому что пройти было нельзя,
получает шум вместо обучающего сигнала — и никакой алгоритм этого не исправит.
Поэтому центральный тест здесь — 20 уровней разной сложности через
`is_solvable`, а рядом — проверки того, что сам `is_solvable` не «всегда да»:
на заведомо непроходимых уровнях он обязан говорить «нет».

Уровни в быстрых тестах намеренно короткие (`length=45`): стоимость и
генерации, и проверки проходимости линейна по длине, а свойства проверяются те
же. Полноразмерные уровни проверяются отдельным тестом с маркером `slow`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdai.env.generator import (
    PATTERN_SPECS,
    PATTERNS,
    PatternContext,
    generate_level,
    make_checkpoints,
)
from gdai.env.level import Level, LevelObject
from gdai.env.physics import make_initial_state, step_physics
from gdai.env.solver import is_solvable, solve_actions, state_key

# Длина уровня для быстрых тестов: достаточно, чтобы уложились 3-5 участков.
SHORT_LENGTH: float = 45.0


def _difficulties(count: int) -> list[float]:
    """Равномерная сетка сложностей от 0 до 1 включительно."""
    return [i / (count - 1) for i in range(count)]


# ---------------------------------------------------------------------------
# главное: проходимость
# ---------------------------------------------------------------------------
def test_twenty_levels_are_solvable() -> None:
    """20 уровней разной сложности проходимы (SPEC §16)."""
    rng = np.random.default_rng(20240726)
    unsolvable: list[tuple[float, str]] = []
    for difficulty in _difficulties(20):
        level = generate_level(
            difficulty, rng, name=f"d{difficulty:.2f}", length=SHORT_LENGTH
        )
        if not is_solvable(level):
            unsolvable.append((difficulty, level.name))
    assert not unsolvable, f"непроходимые уровни: {unsolvable}"


@pytest.mark.slow
def test_full_length_levels_are_solvable() -> None:
    """Полноразмерные уровни (длина по умолчанию) тоже проходимы."""
    rng = np.random.default_rng(7)
    for difficulty in (0.0, 0.25, 0.5, 0.75, 1.0):
        level = generate_level(difficulty, rng, name=f"full-{difficulty}")
        assert level.length > 60.0
        assert is_solvable(level), f"уровень сложности {difficulty} непроходим"


def test_solution_replays_to_the_finish() -> None:
    """Найденный путь действительно доводит до финиша при проигрывании физикой.

    Зачем не доверять `is_solvable` на слово: он считает по своим правилам
    склейки состояний, и единственная честная проверка — прогнать найденную
    последовательность через ту же `step_physics`, что и настоящая игра.
    """
    rng = np.random.default_rng(5)
    level = generate_level(0.4, rng, length=SHORT_LENGTH)
    actions = solve_actions(level)
    assert actions is not None, "решение не найдено"

    state = make_initial_state(level)
    for action in actions:
        state, _events = step_physics(state, level, action == 1)
    assert state.alive is True
    assert state.finished is True


def test_is_solvable_rejects_impossible_levels() -> None:
    """Непроходимое — значит «нет»: иначе проверка бессмысленна."""
    wall = [LevelObject("block", 20.0, 0.5 + i) for i in range(6)]
    blocked = Level(
        name="wall",
        length=40.0,
        objects=wall + [LevelObject("goal", 40.0, 6.0)],
        ceiling_y=12.0,
    )
    assert is_solvable(blocked, max_frames=1500) is False

    field = [LevelObject("spike", 20.0 + i * 0.5, 0.5) for i in range(20)]
    spikes = Level(
        name="spikes",
        length=40.0,
        objects=field + [LevelObject("goal", 40.0, 6.0)],
        ceiling_y=12.0,
    )
    assert is_solvable(spikes, max_frames=1500) is False


def test_is_solvable_accepts_empty_level() -> None:
    """Пустая дорожка проходима без единого нажатия."""
    level = Level(name="empty", length=40.0, objects=[LevelObject("goal", 40.0, 6.0)])
    assert is_solvable(level) is True


# ---------------------------------------------------------------------------
# структура сгенерированного уровня
# ---------------------------------------------------------------------------
def test_generated_level_structure() -> None:
    """У уровня есть финиш, положительная длина и корректные стартовые поля."""
    rng = np.random.default_rng(11)
    level = generate_level(0.5, rng, name="structure", length=SHORT_LENGTH)
    assert level.name == "structure"
    assert level.length > 0.0
    assert level.start_mode == "cube"
    assert level.start_gravity == 1
    assert any(obj.type == "goal" for obj in level.objects)
    # Все объекты — внутри уровня и не под полом.
    for obj in level.objects:
        assert -1.0 <= obj.x <= level.length + 2.0
        assert -1.0 <= obj.y <= level.ceiling_y + 1.0


def test_difficulty_is_clamped_and_monotone_in_density() -> None:
    """Сложность зажимается в [0, 1], а плотность препятствий с ней растёт.

    Проверяется именно тенденция (среднее по нескольким уровням), а не строгое
    неравенство: генерация случайна, и требовать монотонности на каждой паре
    значило бы получить хронически мигающий тест.
    """
    def density(difficulty: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        total = 0.0
        for k in range(3):
            level = generate_level(difficulty, rng, length=SHORT_LENGTH)
            total += len(level.objects) / level.length
        return total / 3.0

    low = density(0.0, 1)
    high = density(0.9, 1)
    assert high > low

    rng = np.random.default_rng(3)
    assert generate_level(-5.0, rng, length=SHORT_LENGTH).length > 0.0
    assert generate_level(42.0, rng, length=SHORT_LENGTH).length > 0.0


def test_generation_is_reproducible_by_seed() -> None:
    """Один seed — один и тот же уровень: без этого нельзя воспроизвести прогон."""
    first = generate_level(0.35, np.random.default_rng(99), length=SHORT_LENGTH)
    second = generate_level(0.35, np.random.default_rng(99), length=SHORT_LENGTH)
    assert first.to_dict() == second.to_dict()

    other = generate_level(0.35, np.random.default_rng(100), length=SHORT_LENGTH)
    assert other.to_dict() != first.to_dict()


# ---------------------------------------------------------------------------
# чекпойнты
# ---------------------------------------------------------------------------
def test_make_checkpoints_are_sorted_and_inside_level() -> None:
    """Чекпойнты возрастают, лежат внутри уровня и не липнут к финишу."""
    level = Level(name="empty", length=120.0, objects=[LevelObject("goal", 120.0, 6.0)])
    checkpoints = make_checkpoints(level, every=25.0)
    assert checkpoints == sorted(checkpoints)
    assert checkpoints
    for x in checkpoints:
        assert 0.0 < x < level.length - 5.0


def test_make_checkpoints_avoids_hazards() -> None:
    """Чекпойнт не ставится вплотную к шипу — с него нельзя было бы стартовать."""
    spikes = [LevelObject("spike", 25.0 + i, 0.5) for i in range(4)]
    level = Level(
        name="cp",
        length=120.0,
        objects=spikes + [LevelObject("goal", 120.0, 6.0)],
    )
    for x in make_checkpoints(level, every=25.0):
        nearest = min(abs(obj.x - x) for obj in spikes)
        assert nearest >= 3.0, f"чекпойнт {x} стоит в шипах"


def test_generated_level_has_usable_checkpoints() -> None:
    """Сгенерированный уровень несёт свои чекпойнты и они внутри уровня."""
    rng = np.random.default_rng(21)
    level = generate_level(0.4, rng, length=80.0)
    assert level.checkpoints == sorted(level.checkpoints)
    for x in level.checkpoints:
        assert 0.0 < x < level.length


# ---------------------------------------------------------------------------
# паттерны и поиск
# ---------------------------------------------------------------------------
def test_patterns_registry_is_consistent() -> None:
    """`PATTERNS` — это ровно функции из `PATTERN_SPECS`, и все они вызываемы."""
    assert PATTERNS == tuple(spec.func for spec in PATTERN_SPECS)
    assert len(PATTERNS) >= 10
    rng = np.random.default_rng(0)
    ctx = PatternContext()
    for spec in PATTERN_SPECS:
        objects, x_end, new_ctx = spec.func(rng, 0.5, 10.0, ctx)
        assert x_end > 10.0, f"участок {spec.name} не продвинул x"
        assert isinstance(new_ctx, PatternContext)
        for obj in objects:
            assert isinstance(obj, LevelObject)


def test_state_key_quantizes_as_specified() -> None:
    """Ключ склейки округляет x/y/vy по квантам 0.25/0.25/0.5 (SPEC §6)."""
    base = make_initial_state(Level(name="e", length=10.0, objects=[]))
    same = type(base)(**{**base.__dict__, "x": base.x + 0.2})
    other = type(base)(**{**base.__dict__, "x": base.x + 0.6})
    assert state_key(base) == state_key(same)
    assert state_key(base) != state_key(other)


def test_solvable_respects_budget() -> None:
    """Крошечный бюджет кадров честно возвращает «не нашли», а не зависает."""
    level = Level(name="long", length=400.0, objects=[LevelObject("goal", 400.0, 6.0)])
    assert is_solvable(level, max_frames=5) is False
    assert is_solvable(level, max_frames=20000) is True
