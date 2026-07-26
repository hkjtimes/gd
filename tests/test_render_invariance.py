"""Ключевой тест проекта: дизайн меняется — смысл не меняется (SPEC §9, §16).

Зачем этот файл важнее остальных
--------------------------------
Вся архитектура GDAI держится на одном утверждении: «каким бы ни был дизайн,
каноническая карта остаётся той же». Если оно перестанет быть верным хотя бы
на пиксель, произойдёт худшее из возможного — ничего не упадёт. Зрение начнёт
учиться на систематически кривой разметке, политика будет получать сдвинутую
геометрию, а метрики покажут «просто чуть хуже сходится».

Поэтому здесь проверяются ровно два взаимно противоположных свойства:

1. **Карта одинакова БАЙТ-В-БАЙТ** для 30 случайных тем и всех встроенных, при
   любом уровне декора и на разных состояниях игрока (куб, корабль, волна,
   перевёрнутая гравитация).
2. **Кадры при этом действительно разные.** Инвариантность, полученная тем,
   что рендер рисует одно и то же, ничего не стоит: доменной рандомизации не
   будет, и зрение переобучится на первую же тему.
"""

from __future__ import annotations

import numpy as np
import pytest

from gdai.env.level import Level, LevelObject
from gdai.env.physics import PlayerState
from gdai.env.render import Renderer, make_renderer
from gdai.env.semantic import camera_origin, render_semantic
from gdai.env.themes import (
    BUILTIN_THEMES,
    THEME_NAMES,
    Theme,
    random_theme,
    theme_by_name,
    to_inverted,
    to_monochrome,
)

# Сколько случайных тем берём в основной тест (SPEC §16: 30).
RANDOM_THEMES: int = 30
# Порог «кадры существенно различаются»: средняя попиксельная разница по всем
# каналам. Значение выбрано с большим запасом — реальная медиана на этом наборе
# около 70, а совсем похожие пары (две тёмные палитры) дают 6-10.
MIN_MEAN_FRAME_DIFF: float = 20.0
# Доля пар кадров, обязанных различаться заметно. Не 100%: две случайные тёмные
# палитры со слабым декором честно могут оказаться похожими, и требовать
# обратного — значит писать мигающий тест.
MIN_DISTINCT_FRACTION: float = 0.9


@pytest.fixture(scope="module")
def rich_level() -> Level:
    """Уровень, в кадре которого есть объект каждого семантического класса.

    Зачем: инвариантность надо проверять там, где рендеру есть что испортить.
    На пустой дорожке совпадут любые две реализации.
    """
    objects = [
        LevelObject("block", 7.0, 0.5),
        LevelObject("block", 7.0, 1.5),
        LevelObject("platform", 9.0, 3.25),
        LevelObject("spike", 11.0, 0.5),
        LevelObject("spike_down", 11.0, 5.0),
        LevelObject("spike_left", 13.0, 2.5),
        LevelObject("spike_right", 14.0, 2.5),
        LevelObject("saw", 16.0, 1.0),
        LevelObject("orb_yellow", 12.5, 2.2),
        LevelObject("orb_red", 15.5, 3.4),
        LevelObject("pad_pink", 10.0, 0.25),
        LevelObject("portal_gravity_up", 17.0, 1.25),
        LevelObject("portal_ship", 18.5, 1.25),
        LevelObject("portal_speed_3", 20.0, 1.25),
        LevelObject("goal", 22.0, 6.0),
    ]
    return Level(name="rich", length=24.0, objects=objects, ceiling_y=12.0)


@pytest.fixture(scope="module")
def states() -> tuple[PlayerState, ...]:
    """Набор состояний, покрывающий все режимы и обе гравитации."""
    return (
        PlayerState(x=10.0, y=0.45, vy=0.0, mode="cube", on_ground=True),
        PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False),
        PlayerState(x=14.0, y=6.0, vy=4.0, mode="ship", on_ground=False),
        PlayerState(x=16.0, y=8.5, vy=-10.0, mode="wave", speed_index=3),
        PlayerState(x=13.0, y=11.55, vy=0.0, mode="cube", gravity=-1, on_ground=True),
    )


def _all_themes(seed: int = 4242) -> list[Theme]:
    """Все встроенные темы + `RANDOM_THEMES` случайных (SPEC §16)."""
    rng = np.random.default_rng(seed)
    return list(BUILTIN_THEMES) + [random_theme(rng) for _ in range(RANDOM_THEMES)]


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Средняя попиксельная разница двух кадров в единицах яркости 0..255."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---------------------------------------------------------------------------
# 1. Семантика не зависит от оформления
# ---------------------------------------------------------------------------
def test_semantic_map_is_byte_identical_across_themes(
    rich_level: Level, states: tuple[PlayerState, ...]
) -> None:
    """Карта одинакова байт-в-байт для всех тем — главный инвариант проекта."""
    themes = _all_themes()
    assert len(themes) >= len(BUILTIN_THEMES) + RANDOM_THEMES

    renderer = Renderer(seed=0)
    for state in states:
        reference = render_semantic(rich_level, state)
        reference_cam = camera_origin(state)
        for theme in themes:
            renderer.set_theme(theme)
            # Кадр рисуется НАМЕРЕННО: только так проверяется, что рендер не
            # трогает ни камеру, ни геометрию через общие кэши.
            renderer.render(rich_level, state, 17)
            again = render_semantic(rich_level, state)
            assert np.array_equal(reference, again), (
                f"тема {theme.name!r} изменила семантическую карту"
            )
            assert camera_origin(state) == reference_cam
    renderer.close()


def test_semantic_map_ignores_decoration_level(
    rich_level: Level, states: tuple[PlayerState, ...]
) -> None:
    """Уровень декора влияет на картинку, но не на разметку."""
    reference = render_semantic(rich_level, states[1])
    for level in (0.0, 0.25, 0.5, 1.0):
        renderer = Renderer(decoration_level=level, seed=3)
        renderer.render(rich_level, states[1], 5)
        assert np.array_equal(render_semantic(rich_level, states[1]), reference)
        renderer.close()


def test_semantic_map_ignores_animation_time(
    rich_level: Level, states: tuple[PlayerState, ...]
) -> None:
    """Номер кадра `t` двигает только анимацию — геометрия от него не зависит."""
    renderer = make_renderer("cyberpunk", seed=1)
    reference = render_semantic(rich_level, states[0])
    frames = [renderer.render(rich_level, states[0], t) for t in (0, 7, 41, 120)]
    for _frame in frames:
        assert np.array_equal(render_semantic(rich_level, states[0]), reference)
    # А сама анимация обязана быть видна — иначе «неизменность» тривиальна.
    assert any(not np.array_equal(frames[0], f) for f in frames[1:])
    renderer.close()


# ---------------------------------------------------------------------------
# 2. Кадры действительно разные
# ---------------------------------------------------------------------------
def test_frames_differ_strongly_across_themes(
    rich_level: Level, states: tuple[PlayerState, ...]
) -> None:
    """Одна и та же сцена в разных темах даёт существенно разные кадры."""
    themes = _all_themes()
    renderer = Renderer(seed=0)
    state = states[1]
    frames: list[np.ndarray] = []
    for theme in themes:
        renderer.set_theme(theme)
        frames.append(renderer.render(rich_level, state, 11).copy())
    renderer.close()

    stack = np.stack(frames)
    assert stack.shape[0] == len(themes)

    diffs = [
        _mean_abs_diff(frames[i], frames[j])
        for i in range(len(frames))
        for j in range(i + 1, len(frames))
    ]
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    assert float(diffs_arr.mean()) > MIN_MEAN_FRAME_DIFF, (
        f"кадры слишком похожи: средняя разница {diffs_arr.mean():.2f}"
    )
    assert float(np.median(diffs_arr)) > MIN_MEAN_FRAME_DIFF
    distinct = float((diffs_arr > 5.0).mean())
    assert distinct >= MIN_DISTINCT_FRACTION, (
        f"только {distinct:.0%} пар кадров различимы"
    )
    # Ни одна пара тем не должна давать буквально одинаковый кадр.
    assert float(diffs_arr.min()) > 0.0


def test_every_builtin_theme_produces_its_own_look(rich_level: Level) -> None:
    """Каждая встроенная тема выглядит по-своему: дубликатов в наборе нет."""
    state = PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False)
    renderer = Renderer(seed=0)
    seen: dict[bytes, str] = {}
    for name in THEME_NAMES:
        renderer.set_theme(theme_by_name(name))
        frame = renderer.render(rich_level, state, 3)
        key = frame.tobytes()
        assert key not in seen, f"темы {name!r} и {seen[key]!r} рисуют одинаково"
        seen[key] = name
    renderer.close()
    assert len(seen) == len(THEME_NAMES) >= 10


def test_randomize_changes_look_but_not_meaning(rich_level: Level) -> None:
    """`Renderer.randomize` — это доменная рандомизация, а не смена уровня."""
    state = PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False)
    reference = render_semantic(rich_level, state)
    renderer = Renderer(seed=0)
    rng = np.random.default_rng(2024)
    frames: list[np.ndarray] = []
    names: set[str] = set()
    for _ in range(12):
        renderer.randomize(rng)
        names.add(renderer.theme.name)
        frames.append(renderer.render(rich_level, state, 9).copy())
        assert np.array_equal(render_semantic(rich_level, state), reference)
    renderer.close()
    assert len(names) == 12, "randomize обязан каждый раз давать новую тему"
    diffs = [_mean_abs_diff(frames[0], f) for f in frames[1:]]
    assert float(np.mean(diffs)) > MIN_MEAN_FRAME_DIFF


# ---------------------------------------------------------------------------
# 3. Свойства самих тем
# ---------------------------------------------------------------------------
def test_builtin_theme_set_is_large_enough() -> None:
    """Встроенных тем не меньше десяти (SPEC §9) и имена уникальны."""
    assert len(BUILTIN_THEMES) >= 10
    assert len(set(THEME_NAMES)) == len(THEME_NAMES)
    for name in THEME_NAMES:
        assert theme_by_name(name).name == name
        assert theme_by_name(name.upper()).name == name
    with pytest.raises(ValueError, match="Неизвестная тема"):
        theme_by_name("no-such-theme")


def test_random_themes_never_shake() -> None:
    """`shake` случайных тем всегда 0: сдвиг кадра без сдвига карты — это ложь.

    SPEC §9 разрешает тряску только как пост-эффект, применённый К ОБОИМ
    массивам; поэтому источник случайных тем обязан держать её выключенной.
    """
    rng = np.random.default_rng(1)
    for _ in range(RANDOM_THEMES):
        assert random_theme(rng).shake == 0.0
    for theme in BUILTIN_THEMES:
        assert theme.shake == 0.0


def test_random_themes_are_varied() -> None:
    """Случайные темы действительно покрывают пространство стилей, а не один угол."""
    rng = np.random.default_rng(77)
    themes = [random_theme(rng) for _ in range(60)]
    assert len({t.block_style for t in themes}) >= 4
    assert len({t.bg_style for t in themes}) >= 4
    assert len({t.hazard_style for t in themes}) >= 3
    assert len({t.name for t in themes}) == len(themes)
    assert any(t.monochrome for t in themes) or any(t.inverted for t in themes)
    # «Злые» темы: цвет шипа иногда почти совпадает с цветом блока.
    close_pairs = sum(
        1
        for t in themes
        if abs(sum(t.hazard_fill) - sum(t.block_fill)) < 60
    )
    assert close_pairs > 0, "нет ни одной темы, где шип похож на блок по цвету"


def test_theme_transforms_keep_geometry(rich_level: Level) -> None:
    """Монохром и инверсия меняют палитру, но не карту."""
    state = PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False)
    reference = render_semantic(rich_level, state)
    base = theme_by_name("neon")
    renderer = Renderer(seed=0)
    frames = []
    for theme in (base, to_monochrome(base), to_inverted(base)):
        renderer.set_theme(theme)
        frames.append(renderer.render(rich_level, state, 4).copy())
        assert np.array_equal(render_semantic(rich_level, state), reference)
    renderer.close()
    assert _mean_abs_diff(frames[0], frames[1]) > 5.0
    assert _mean_abs_diff(frames[0], frames[2]) > MIN_MEAN_FRAME_DIFF


def test_renderer_is_reproducible_by_seed(rich_level: Level) -> None:
    """Два рендерера с одним seed и темой дают идентичный кадр.

    Зачем: датасет зрения обязан воспроизводиться, иначе сравнивать два прогона
    обучения бессмысленно.
    """
    state = PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False)
    a = make_renderer("space", seed=7)
    b = make_renderer("space", seed=7)
    assert np.array_equal(a.render(rich_level, state, 5), b.render(rich_level, state, 5))

    c = Renderer(seed=7)
    d = Renderer(seed=7)
    c.randomize(np.random.default_rng(99))
    d.randomize(np.random.default_rng(99))
    assert c.theme.name == d.theme.name
    assert np.array_equal(c.render(rich_level, state, 5), d.render(rich_level, state, 5))
    for renderer in (a, b, c, d):
        renderer.close()


def test_frame_shape_matches_semantic_map(rich_level: Level) -> None:
    """Кадр и карта одного размера — иначе они не соответствуют попиксельно."""
    state = PlayerState(x=12.0, y=2.6, vy=-8.0, mode="cube", on_ground=False)
    renderer = Renderer(seed=0)
    frame = renderer.render(rich_level, state, 0)
    sem = render_semantic(rich_level, state)
    assert frame.shape[:2] == sem.shape
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    renderer.close()


@pytest.mark.slow
def test_invariance_holds_on_many_states(rich_level: Level) -> None:
    """Расширенная проверка: 30 случайных тем на 25 случайных состояниях."""
    rng = np.random.default_rng(31337)
    themes = [random_theme(rng) for _ in range(RANDOM_THEMES)]
    modes = ("cube", "ship", "wave")
    renderer = Renderer(seed=1)
    for _ in range(25):
        state = PlayerState(
            x=float(rng.uniform(2.0, 22.0)),
            y=float(rng.uniform(0.3, 11.5)),
            vy=float(rng.uniform(-20.0, 20.0)),
            mode=str(modes[int(rng.integers(3))]),
            gravity=int(rng.choice([-1, 1])),
            speed_index=int(rng.integers(0, 5)),
            on_ground=bool(rng.integers(2)),
        )
        reference = render_semantic(rich_level, state)
        for theme in themes:
            renderer.set_theme(theme)
            renderer.render(rich_level, state, int(rng.integers(0, 200)))
            assert np.array_equal(render_semantic(rich_level, state), reference)
    renderer.close()
