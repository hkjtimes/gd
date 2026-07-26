"""Тесты полного агента `gdai.pipeline.GDAgent` (SPEC §12, §16).

Зачем это отдельный слой
------------------------
`GDAgent` — единственное место, где зрение и политика встречаются, и именно
здесь ломается связка: карта, предсказанная сетью, обязана иметь ровно тот же
формат, что и эталонная, иначе политика получит на вход мусор и никто об этом
не узнает (сеть не падает от неправильных чисел, она просто играет плохо).

Ключевое требование SPEC §16: агент обязан работать БЕЗ обученных весов, на
случайной инициализации. Это делает возможными и smoke-тесты, и `selfcheck`, и
демонстрацию до того, как хоть что-то обучено.

`importorskip` оставлен намеренно: `gdai.pipeline` — верхний слой связки, и в
урезанном окружении (без torch) он недоступен. Пропуск здесь честнее падения:
он не маскирует поломки в нижних модулях, которые в таком окружении работают.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pipeline = pytest.importorskip(
    "gdai.pipeline",
    reason="gdai/pipeline.py ещё не реализован (SPEC §12, фаза 4)",
)

from gdai.config import EnvConfig  # noqa: E402
from gdai.constants import NUM_ACTIONS, NUM_CLASSES, OBS_H, OBS_W  # noqa: E402
from gdai.env.gd_env import GeometryDashEnv  # noqa: E402
from gdai.env.level import Level, LevelObject  # noqa: E402


@pytest.fixture
def runway() -> Level:
    """Короткий детерминированный уровень — без процедурной генерации в тестах."""
    return Level(
        name="pipeline-runway",
        length=40.0,
        objects=[LevelObject("spike", 20.0, 0.5), LevelObject("goal", 40.0, 6.0)],
        ceiling_y=12.0,
    )


def test_agent_works_without_trained_weights() -> None:
    """`GDAgent` собирается на случайной инициализации (SPEC §16)."""
    agent = pipeline.GDAgent(device="cpu")
    assert agent is not None
    agent.reset()


def test_see_returns_canonical_map(runway: Level) -> None:
    """`see` переводит кадр (H,W,3) в карту классов (H,W) с корректным диапазоном."""
    agent = pipeline.GDAgent(device="cpu", use_perception=True)
    with GeometryDashEnv(EnvConfig(obs_mode="pixels")) as env:
        obs, _info = env.reset(level=runway)
        semantic = agent.see(obs["pixels"])
    assert semantic.shape == (OBS_H, OBS_W)
    assert semantic.dtype == np.uint8
    assert int(semantic.max()) < NUM_CLASSES


def test_act_returns_valid_action(runway: Level) -> None:
    """`act` возвращает допустимое действие и по эталонной карте, и по предсказанной."""
    with GeometryDashEnv(EnvConfig(obs_mode="both")) as env:
        obs, _info = env.reset(level=runway)
        for use_perception in (False, True):
            agent = pipeline.GDAgent(device="cpu", use_perception=use_perception)
            agent.reset()
            action = agent.act(obs, deterministic=True)
            assert isinstance(action, (int, np.integer))
            assert 0 <= int(action) < NUM_ACTIONS


def test_deterministic_act_is_stable(runway: Level) -> None:
    """При `deterministic=True` одно и то же наблюдение даёт одно и то же действие."""
    agent = pipeline.GDAgent(device="cpu", use_perception=False)
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        obs, _info = env.reset(level=runway)
        first = agent.act(obs, deterministic=True)
        second = agent.act(obs, deterministic=True)
    assert int(first) == int(second)


def test_agent_plays_an_episode(runway: Level) -> None:
    """Агент проходит цикл среды целиком, не ломая контракт `step`."""
    agent = pipeline.GDAgent(device="cpu", use_perception=False)
    with GeometryDashEnv(EnvConfig(obs_mode="semantic", max_steps=120)) as env:
        obs, _info = env.reset(level=runway)
        agent.reset()
        for _ in range(120):
            action = agent.act(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            if terminated or truncated:
                break


def test_evaluate_returns_metrics(runway: Level) -> None:
    """`evaluate` считает метрики из SPEC §12 и держит их в разумных пределах."""
    agent = pipeline.GDAgent(device="cpu", use_perception=False)
    with GeometryDashEnv(EnvConfig(obs_mode="semantic", max_steps=150)) as env:
        env.reset(level=runway)
        metrics = pipeline.evaluate(agent, env, episodes=2, use_perception=False)
    for key in ("success_rate", "mean_progress", "mean_reward", "mean_len"):
        assert key in metrics
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["mean_progress"] <= 1.0
    assert metrics["mean_len"] >= 0.0
    assert metrics["episodes"] == 2


def test_evaluate_with_perception_requires_pixels(runway: Level) -> None:
    """Оценка «честным» зрением без кадров — понятная ошибка, а не тихий эталон."""
    agent = pipeline.GDAgent(device="cpu", use_perception=True)
    with GeometryDashEnv(EnvConfig(obs_mode="semantic", max_steps=40)) as env:
        env.reset(level=runway)
        with pytest.raises(ValueError, match="pixels"):
            pipeline.evaluate(agent, env, episodes=1, use_perception=True)


def test_perception_path_runs_end_to_end(runway: Level) -> None:
    """Полная цепочка «пиксели -> зрение -> политика» работает на случайных весах."""
    agent = pipeline.GDAgent(device="cpu", use_perception=True)
    with GeometryDashEnv(EnvConfig(obs_mode="both", max_steps=60)) as env:
        env.reset(level=runway)
        metrics = pipeline.evaluate(agent, env, episodes=1, use_perception=True)
    assert 0.0 <= metrics["mean_progress"] <= 1.0
    assert metrics["use_perception"] is True


def test_agent_without_perception_rejects_pixels_only(runway: Level) -> None:
    """Без зрения и без эталонной карты политике не на что смотреть."""
    agent = pipeline.GDAgent(device="cpu", use_perception=False)
    with GeometryDashEnv(EnvConfig(obs_mode="pixels")) as env:
        obs, _info = env.reset(level=runway)
        with pytest.raises(ValueError):
            agent.act(obs)


def test_telemetry_is_updated_and_reset(runway: Level) -> None:
    """Телеметрия решения заполняется и очищается `reset` — её показывает вьюер."""
    agent = pipeline.GDAgent(device="cpu", use_perception=False)
    with GeometryDashEnv(EnvConfig(obs_mode="semantic")) as env:
        obs, _info = env.reset(level=runway)
        action, p_hold, value = agent.decide(obs, deterministic=True)
    assert 0 <= action < NUM_ACTIONS
    assert 0.0 <= p_hold <= 1.0
    assert np.isfinite(value)
    assert agent.frames_seen == 1
    assert agent.last_semantic is not None

    agent.reset()
    assert agent.frames_seen == 0
    assert agent.last_semantic is None


def test_describe_reports_random_weights() -> None:
    """`describe` честно сообщает, что веса случайные, и считает параметры."""
    agent = pipeline.GDAgent(device="cpu", use_perception=True)
    info = agent.describe()
    assert info["policy"] == pipeline.RANDOM_WEIGHTS
    assert info["perception"] == pipeline.RANDOM_WEIGHTS
    assert info["policy_params"] > 0
    assert 0 < info["perception_params"] < 500_000
    assert info["obs_size"] == (OBS_H, OBS_W)


def test_missing_checkpoint_raises(tmp_path) -> None:
    """Несуществующий чекпойнт — понятная ошибка, а не молча случайные веса."""
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        pipeline.GDAgent(policy_path=str(tmp_path / "nope.pt"), device="cpu")
