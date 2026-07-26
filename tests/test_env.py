"""Тесты среды (SPEC §8).

Среда — контракт между игрой и обучением, поэтому проверяется именно контракт:
формы наблюдений по режимам, состав `info`, формула награды, поведение
`terminated`/`truncated`, воспроизводимость по seed и practice-чекпойнты.

Почти все тесты работают на ЗАДАННОМ уровне (`reset(level=...)` или
`level_path`): процедурная генерация стоит около секунды на уровень, и без
этого тесты среды превратились бы в тесты генератора.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gdai.config import EnvConfig
from gdai.constants import (
    ACTION_HOLD,
    ACTION_NONE,
    MAX_FALL_V,
    NUM_ACTIONS,
    OBS_H,
    OBS_W,
    SPEED_TILES_PER_SEC,
    SPEEDS,
)
from gdai.env.gd_env import FEATURE_DIM, REWARD_PROGRESS_SCALE, GeometryDashEnv
from gdai.env.level import Level, LevelObject


@pytest.fixture
def runway() -> Level:
    """Длинная ровная дорожка с чекпойнтами и одним шипом в конце.

    Зачем такой уровень: он позволяет проверить и прогресс, и смерть, и
    практику по чекпойнтам, не завися от случайного дизайна.
    """
    return Level(
        name="runway",
        length=60.0,
        objects=[LevelObject("spike", 30.0, 0.5), LevelObject("goal", 60.0, 6.0)],
        ceiling_y=12.0,
        checkpoints=[10.0, 20.0],
    )


def _run_until_done(env: GeometryDashEnv, action: int = ACTION_NONE) -> dict:
    """Доиграть эпизод одним и тем же действием и вернуть последний `info`."""
    while True:
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info


# ---------------------------------------------------------------------------
# контракт наблюдений
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obs_mode, expected",
    [
        ("semantic", {"semantic", "features"}),
        ("pixels", {"pixels", "features"}),
        ("both", {"semantic", "pixels", "features"}),
    ],
)
def test_observation_keys_and_shapes(runway: Level, obs_mode: str, expected: set) -> None:
    """Состав и формы наблюдения строго по `obs_mode` (SPEC §8)."""
    with GeometryDashEnv(EnvConfig(obs_mode=obs_mode, max_steps=50)) as env:
        obs, info = env.reset(level=runway)
        assert set(obs) == expected
        assert set(env.observation_shapes) == expected
        if "semantic" in obs:
            assert obs["semantic"].shape == (OBS_H, OBS_W)
            assert obs["semantic"].dtype == np.uint8
        if "pixels" in obs:
            assert obs["pixels"].shape == (OBS_H, OBS_W, 3)
            assert obs["pixels"].dtype == np.uint8
        assert obs["features"].shape == (FEATURE_DIM,)
        assert obs["features"].dtype == np.float32
        assert isinstance(info, dict)


def test_unknown_obs_mode_raises() -> None:
    """Опечатка в `obs_mode` — ошибка при создании, а не пустое наблюдение."""
    with pytest.raises(ValueError, match="obs_mode"):
        GeometryDashEnv(EnvConfig(obs_mode="rgb"))


def test_feature_vector_layout(runway: Level) -> None:
    """`features` = [vy_norm, on_ground, gravity, cube, ship, wave, speed, progress]."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        obs, _info = env.reset(level=runway)
        feat = obs["features"]
        assert feat[0] == pytest.approx(0.0)          # стоим, vy = 0
        assert feat[1] == pytest.approx(1.0)          # на земле
        assert feat[2] == pytest.approx(1.0)          # гравитация вниз
        assert (feat[3], feat[4], feat[5]) == (1.0, 0.0, 0.0)   # режим куба
        assert feat[6] == pytest.approx(1.0 / (len(SPEEDS) - 1))
        assert feat[7] == pytest.approx(0.0)          # прогресс в начале

        obs, _r, _t, _tr, _i = env.step(ACTION_HOLD)
        assert obs["features"][1] == pytest.approx(0.0)         # в прыжке
        assert obs["features"][0] == pytest.approx(env.state.vy / MAX_FALL_V, abs=1e-6)
        assert -1.0 <= obs["features"][0] <= 1.0
        assert obs["features"][7] > 0.0


def test_info_contract(runway: Level) -> None:
    """`info` содержит поля из SPEC §8 (плюс служебные для логов обучения)."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        _obs, info = env.reset(level=runway)
        for key in ("x", "progress", "died", "finished", "level_name", "difficulty"):
            assert key in info
        assert info["level_name"] == "runway"
        assert info["progress"] == pytest.approx(0.0)
        _obs, _r, _t, _tr, info = env.step(ACTION_NONE)
        assert info["x"] > 0.0
        assert 0.0 <= info["progress"] <= 1.0


# ---------------------------------------------------------------------------
# награда
# ---------------------------------------------------------------------------
def test_progress_reward_formula(runway: Level) -> None:
    """Награда за кадр = reward_progress * dx / 10 (+ reward_alive)."""
    cfg = EnvConfig(obs_mode="semantic", reward_progress=1.0, reward_alive=0.0)
    with GeometryDashEnv(cfg) as env:
        env.reset(level=runway)
        _obs, reward, _t, _tr, _info = env.step(ACTION_NONE)
        expected = SPEED_TILES_PER_SEC[1] / 60.0 / REWARD_PROGRESS_SCALE
        assert reward == pytest.approx(expected, rel=1e-6)


def test_alive_bonus_is_added(runway: Level) -> None:
    """`reward_alive` добавляется каждый кадр."""
    base = EnvConfig(obs_mode="semantic", reward_alive=0.0)
    bonus = EnvConfig(obs_mode="semantic", reward_alive=0.5)
    with GeometryDashEnv(base) as a, GeometryDashEnv(bonus) as b:
        a.reset(level=runway)
        b.reset(level=runway)
        _o, r_a, *_ = a.step(ACTION_NONE)
        _o, r_b, *_ = b.step(ACTION_NONE)
        assert r_b - r_a == pytest.approx(0.5)


def test_death_reward_and_termination(runway: Level) -> None:
    """Смерть: `reward_death` в награде, `terminated=True`, `died=True`."""
    cfg = EnvConfig(obs_mode="semantic", max_steps=5000, reward_death=-1.0,
                    practice_checkpoints=False)
    with GeometryDashEnv(cfg) as env:
        env.reset(level=runway)
        while True:
            _obs, reward, terminated, truncated, info = env.step(ACTION_NONE)
            if terminated or truncated:
                break
        assert info["died"] is True and info["finished"] is False
        assert terminated is True and truncated is False
        assert reward < 0.0, "штраф за смерть обязан перевесить награду за кадр"


def test_finish_reward_and_termination() -> None:
    """Финиш: `reward_finish`, `terminated=True`, `finished=True`."""
    level = Level(name="short", length=8.0, objects=[LevelObject("goal", 8.0, 6.0)])
    cfg = EnvConfig(obs_mode="semantic", max_steps=5000, reward_finish=10.0)
    with GeometryDashEnv(cfg) as env:
        env.reset(level=level)
        while True:
            _obs, reward, terminated, truncated, info = env.step(ACTION_NONE)
            if terminated or truncated:
                break
        assert info["finished"] is True and info["died"] is False
        assert terminated is True
        assert reward > 9.0


def test_truncation_on_max_steps(runway: Level) -> None:
    """Лимит шагов даёт `truncated`, но НЕ `terminated` — эпизод не закончился."""
    cfg = EnvConfig(obs_mode="semantic", max_steps=12)
    with GeometryDashEnv(cfg) as env:
        env.reset(level=runway)
        for step in range(12):
            _obs, _r, terminated, truncated, info = env.step(ACTION_NONE)
        assert terminated is False and truncated is True
        assert info["steps"] == 12
        with pytest.raises(RuntimeError, match="[Ээ]пизод"):
            env.step(ACTION_NONE)


# ---------------------------------------------------------------------------
# жизненный цикл и ошибки
# ---------------------------------------------------------------------------
def test_step_before_reset_raises() -> None:
    """Шаг до `reset` — ошибка: играть на несуществующем уровне нельзя."""
    env = GeometryDashEnv(EnvConfig(obs_mode="semantic"))
    with pytest.raises(RuntimeError):
        env.step(ACTION_NONE)
    with pytest.raises(RuntimeError):
        _ = env.level
    with pytest.raises(RuntimeError):
        _ = env.state
    env.close()


def test_invalid_action_raises(runway: Level) -> None:
    """Действие вне 0..1 отвергается явно."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        env.reset(level=runway)
        with pytest.raises(ValueError):
            env.step(NUM_ACTIONS)
        with pytest.raises(ValueError):
            env.step(-1)


def test_close_is_idempotent(runway: Level) -> None:
    """`close` можно звать сколько угодно раз, после него среда не работает."""
    env = GeometryDashEnv(EnvConfig(obs_mode="semantic"))
    env.reset(level=runway)
    env.close()
    env.close()
    with pytest.raises(RuntimeError):
        env.reset(level=runway)


def test_render_returns_frame(runway: Level) -> None:
    """`render` всегда отдаёт «красивый» кадр, даже в семантическом режиме."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        env.reset(level=runway)
        frame = env.render()
        assert frame.shape == (OBS_H, OBS_W, 3)
        assert frame.dtype == np.uint8
        with pytest.raises(ValueError):
            env.render(mode="ascii")


def test_level_from_file(level_file: Path) -> None:
    """`level_path` загружает фиксированный уровень и держится за него."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic", level_path=str(level_file))) as env:
        _obs, info = env.reset()
        assert info["level_name"] == "demo"
        _obs, info = env.reset()
        assert info["level_name"] == "demo"


def test_reset_with_start_x(runway: Level) -> None:
    """`start_x` переносит старт: практика конкретного места уровня."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        _obs, info = env.reset(level=runway, start_x=25.0)
        assert info["x"] == pytest.approx(25.0)
        assert env.state.on_ground is True
        assert env.state.alive is True


def test_set_difficulty_is_clamped(runway: Level) -> None:
    """Сложность зажимается в [0, 1] и попадает в `info`."""
    with GeometryDashEnv(EnvConfig(obs_mode="semantic", difficulty=0.3)) as env:
        env.reset(level=runway)
        assert env.difficulty == pytest.approx(0.3)
        env.set_difficulty(5.0)
        assert env.difficulty == pytest.approx(1.0)
        env.set_difficulty(-1.0)
        assert env.difficulty == pytest.approx(0.0)
        _obs, _r, _t, _tr, info = env.step(ACTION_NONE)
        assert info["difficulty"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# воспроизводимость
# ---------------------------------------------------------------------------
def _rollout(seed: int, steps: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Короткий прогон со случайными действиями, полностью заданный seed."""
    cfg = EnvConfig(obs_mode="semantic", max_steps=200, difficulty=0.05, seed=seed)
    env = GeometryDashEnv(cfg)
    obs, _info = env.reset(seed=seed)
    maps = [obs["semantic"].copy()]
    rewards: list[float] = []
    actions = np.random.default_rng(0).integers(0, NUM_ACTIONS, size=steps)
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(int(action))
        rewards.append(reward)
        maps.append(obs["semantic"].copy())
        if terminated or truncated:
            obs, _info = env.reset()
            maps.append(obs["semantic"].copy())
    env.close()
    return np.stack(maps), np.asarray(rewards)


def test_same_seed_gives_same_episode() -> None:
    """Один seed — идентичные наблюдения и награды (иначе не сравнить прогоны)."""
    maps_a, rewards_a = _rollout(101)
    maps_b, rewards_b = _rollout(101)
    assert np.array_equal(maps_a, maps_b)
    assert np.allclose(rewards_a, rewards_b)


def test_different_seed_gives_different_episode() -> None:
    """Разные seed дают разные уровни — иначе восемь сред учат одному и тому же."""
    maps_a, _ = _rollout(101)
    maps_c, _ = _rollout(202)
    assert maps_a.shape != maps_c.shape or not np.array_equal(maps_a, maps_c)


def test_semantic_noise_corrupts_map(runway: Level) -> None:
    """`semantic_noise` портит долю пикселей — так политика учится терпеть ошибки зрения."""
    clean_cfg = EnvConfig(obs_mode="semantic", semantic_noise=0.0, seed=1)
    noisy_cfg = EnvConfig(obs_mode="semantic", semantic_noise=0.25, seed=1)
    with GeometryDashEnv(clean_cfg) as clean, GeometryDashEnv(noisy_cfg) as noisy:
        clean_obs, _ = clean.reset(level=runway)
        noisy_obs, _ = noisy.reset(level=runway)
        changed = float((clean_obs["semantic"] != noisy_obs["semantic"]).mean())
    assert 0.1 < changed < 0.5, f"доля испорченных пикселей {changed:.2f} неправдоподобна"


# ---------------------------------------------------------------------------
# practice-чекпойнты
# ---------------------------------------------------------------------------
def test_practice_restarts_from_checkpoint(runway: Level) -> None:
    """После смерти за чекпойнтом следующий эпизод начинается с него."""
    cfg = EnvConfig(
        obs_mode="semantic",
        max_steps=5000,
        practice_checkpoints=True,
        checkpoint_prob=1.0,
        seed=3,
    )
    with GeometryDashEnv(cfg) as env:
        _obs, _info = env.reset(level=runway)
        assert env.checkpoints == [10.0, 20.0]
        info = _run_until_done(env)
        assert info["died"] is True and info["x"] > 20.0

        _obs, info = env.reset()
        assert info["from_checkpoint"] is True
        assert info["x"] == pytest.approx(20.0, abs=1.0)
        assert env.state.alive is True and env.state.hold_prev is False


def test_practice_can_be_disabled(runway: Level) -> None:
    """`checkpoint_prob=0` — практики нет, эпизод всегда с начала."""
    cfg = EnvConfig(
        obs_mode="semantic",
        max_steps=5000,
        practice_checkpoints=True,
        checkpoint_prob=0.0,
        seed=3,
    )
    with GeometryDashEnv(cfg) as env:
        env.reset(level=runway)
        _run_until_done(env)
        _obs, info = env.reset()
        assert info["from_checkpoint"] is False
        assert info["x"] == pytest.approx(0.0)


def test_practice_flag_off_disables_checkpoints(runway: Level) -> None:
    """`practice_checkpoints=False` полностью выключает механизм."""
    cfg = EnvConfig(
        obs_mode="semantic",
        max_steps=5000,
        practice_checkpoints=False,
        checkpoint_prob=1.0,
        seed=3,
    )
    with GeometryDashEnv(cfg) as env:
        env.reset(level=runway)
        _run_until_done(env)
        _obs, info = env.reset()
        assert info["from_checkpoint"] is False
        assert info["x"] == pytest.approx(0.0)


def test_finishing_clears_pending_practice() -> None:
    """Пройдя уровень целиком, практиковать больше нечего."""
    level = Level(
        name="easy",
        length=30.0,
        objects=[LevelObject("goal", 30.0, 6.0)],
        checkpoints=[10.0, 20.0],
    )
    cfg = EnvConfig(
        obs_mode="semantic", max_steps=5000, practice_checkpoints=True,
        checkpoint_prob=1.0, seed=3,
    )
    with GeometryDashEnv(cfg) as env:
        env.reset(level=level)
        info = _run_until_done(env)
        assert info["finished"] is True
        _obs, info = env.reset()
        assert info["from_checkpoint"] is False
        assert info["x"] == pytest.approx(0.0)


@pytest.mark.slow
def test_procedural_levels_are_playable() -> None:
    """Среда без файла уровня генерирует свои — и они играются без исключений."""
    cfg = EnvConfig(obs_mode="semantic", max_steps=400, difficulty=0.2, seed=5)
    with GeometryDashEnv(cfg) as env:
        rng = np.random.default_rng(0)
        obs, _info = env.reset()
        for _ in range(600):
            obs, _r, terminated, truncated, info = env.step(int(rng.integers(2)))
            assert obs["semantic"].shape == (OBS_H, OBS_W)
            assert 0.0 <= info["progress"] <= 1.0
            if terminated or truncated:
                obs, _info = env.reset()
