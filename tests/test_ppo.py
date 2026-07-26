"""Тесты политики и обучения (SPEC §11, §16).

Smoke-тест обучения здесь — не «проверим, что агент научился» (за 2000 шагов
никто не научится), а проверка того, что вся связка вообще жива: среды,
буфер, GAE, оптимизация, логи и чекпойнты. Плюс отдельные точечные тесты на
те части, где ошибка не видна по логам:

* **GAE** — арифметика на игрушечных числах, где правильный ответ выводится
  на бумаге. Это единственный способ поймать протечку награды через границу
  эпизода: на реальных данных она выглядит как «чуть хуже сходится».
* **Curriculum** — правила повышения ступени.
* **SyncVectorEnv** — авто-reset и `final_observation`.

Обучение запускается на ФИКСИРОВАННОМ уровне: процедурная генерация стоит
около секунды на уровень, и smoke-тест PPO превратился бы в тест генератора.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gdai.agent.buffer import MiniBatch, RolloutBuffer  # noqa: E402
from gdai.agent.curriculum import Curriculum  # noqa: E402
from gdai.agent.networks import (  # noqa: E402
    HOLD_PRIOR,
    POLICY_SEM_H,
    POLICY_SEM_W,
    ActorCritic,
    semantic_to_indices,
    semantic_to_tensor,
)
from gdai.agent.ppo import load_policy, train_agent  # noqa: E402
from gdai.agent.vecenv import SyncVectorEnv, make_env_fns  # noqa: E402
from gdai.config import CurriculumConfig, EnvConfig, PPOConfig  # noqa: E402
from gdai.constants import ACTION_HOLD, NUM_ACTIONS, NUM_CLASSES, OBS_H, OBS_W  # noqa: E402
from gdai.env.gd_env import FEATURE_DIM  # noqa: E402
from gdai.env.semantic import downsample_semantic  # noqa: E402


# ---------------------------------------------------------------------------
# сеть
# ---------------------------------------------------------------------------
def test_actor_critic_shapes() -> None:
    """Вход one-hot (B,10,36,64) + признаки (B,8) -> логиты (B,2) и ценность (B,)."""
    model = ActorCritic(feature_dim=FEATURE_DIM)
    sem = torch.zeros(4, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W)
    feat = torch.zeros(4, FEATURE_DIM)
    logits, value = model(sem, feat)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4,)


def test_semantic_to_tensor_is_one_hot() -> None:
    """`semantic_to_tensor` = сжатие вдвое + one-hot, согласованное с эталоном."""
    rng = np.random.default_rng(0)
    sem = rng.integers(0, NUM_CLASSES, size=(OBS_H, OBS_W)).astype(np.uint8)
    tensor = semantic_to_tensor(sem)
    assert tensor.shape == (1, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W)
    # Ровно один единичный канал на пиксель.
    assert torch.allclose(tensor.sum(dim=1), torch.ones(1, POLICY_SEM_H, POLICY_SEM_W))
    indices = semantic_to_indices(sem)
    assert np.array_equal(indices, downsample_semantic(sem, 2))
    assert torch.equal(tensor.argmax(dim=1)[0], torch.as_tensor(indices.astype(np.int64)))


def test_semantic_to_indices_batch_matches_single() -> None:
    """Векторизованное сжатие партии совпадает с поштучным эталоном."""
    rng = np.random.default_rng(1)
    batch = rng.integers(0, NUM_CLASSES, size=(5, OBS_H, OBS_W)).astype(np.uint8)
    assert np.array_equal(
        semantic_to_indices(batch),
        np.stack([downsample_semantic(s, 2) for s in batch]),
    )


def test_act_and_evaluate_actions() -> None:
    """`act` даёт действие/logprob/value, `evaluate_actions` их воспроизводит."""
    torch.manual_seed(0)
    model = ActorCritic(feature_dim=FEATURE_DIM)
    sem = torch.zeros(6, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W)
    feat = torch.zeros(6, FEATURE_DIM)

    action, log_prob, value = model.act(sem, feat)
    assert action.shape == (6,) and log_prob.shape == (6,) and value.shape == (6,)
    assert int(action.min()) >= 0 and int(action.max()) < NUM_ACTIONS

    new_logp, entropy, new_value = model.evaluate_actions(sem, feat, action)
    assert torch.allclose(new_logp, log_prob, atol=1e-6)
    assert torch.allclose(new_value, value, atol=1e-6)
    assert (entropy >= 0.0).all()

    greedy, _lp, _v = model.act(sem, feat, deterministic=True)
    assert torch.equal(greedy, model.act(sem, feat, deterministic=True)[0])


def test_initial_policy_matches_hold_prior() -> None:
    """Стартовое распределение задаётся ровно смещением головы политики.

    Голова инициализируется с gain 0.01, поэтому её веса почти нулевые и на
    старте политика не зависит от входа — её задаёт только bias. Он выставлен
    так, чтобы `P(hold) == HOLD_PRIOR`.

    Прижимать старт к равномерному распределению здесь было бы ошибкой:
    удержание половину кадров означает, что куб прыгает почти сразу после
    каждого приземления, и роллаут вырождается в однообразные быстрые смерти.
    """
    torch.manual_seed(0)
    model = ActorCritic(feature_dim=FEATURE_DIM)

    probs = model.action_probs(
        torch.zeros(1, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W),
        torch.zeros(1, FEATURE_DIM),
    )
    assert probs.shape == (1, NUM_ACTIONS)
    assert float(probs[0, ACTION_HOLD]) == pytest.approx(HOLD_PRIOR, abs=1e-4)

    # Веса почти нулевые: на произвольном входе распределение то же самое.
    noisy = model.action_probs(
        torch.rand(1, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W),
        torch.randn(1, FEATURE_DIM),
    )
    assert float((noisy - probs).abs().max()) < 0.05


# ---------------------------------------------------------------------------
# буфер и GAE
# ---------------------------------------------------------------------------
def _fill(
    buffer: RolloutBuffer,
    rewards: list[float],
    dones: list[float],
    values: list[float] | None = None,
    bootstraps: list[float] | None = None,
) -> None:
    """Заполнить буфер одной средой заданными наградами и флагами завершения."""
    n = buffer.num_envs
    for t in range(buffer.num_steps):
        buffer.add(
            sem=np.zeros((n, *buffer.sem_shape), dtype=np.uint8),
            features=np.zeros((n, buffer.feature_dim), dtype=np.float32),
            actions=np.zeros(n),
            log_probs=np.zeros(n, dtype=np.float32),
            values=np.full(n, 0.0 if values is None else values[t], dtype=np.float32),
            rewards=np.full(n, rewards[t], dtype=np.float32),
            dones=np.full(n, dones[t], dtype=np.float32),
            bootstrap_values=(
                None if bootstraps is None
                else np.full(n, bootstraps[t], dtype=np.float32)
            ),
        )


def test_buffer_fill_and_flatten() -> None:
    """Буфер набирается ровно `num_steps` раз и отдаёт минибатчи нужного размера."""
    buffer = RolloutBuffer(num_steps=4, num_envs=3, sem_shape=(6, 8), feature_dim=2)
    buffer.reset()
    _fill(buffer, [1.0] * 4, [0.0] * 4)
    assert buffer.full is True and buffer.size == 4
    with pytest.raises(RuntimeError):
        _fill(buffer, [1.0], [0.0])

    buffer.compute_returns_and_advantages(
        last_values=np.zeros(3, dtype=np.float32),
        last_dones=np.zeros(3, dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )
    batches = list(buffer.minibatches(2, np.random.default_rng(0)))
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == buffer.batch_size
    first = batches[0]
    assert isinstance(first, MiniBatch)
    assert first.sem.shape[1:] == (6, 8)
    assert first.features.shape[1] == 2
    # Преимущества нормализованы внутри минибатча.
    assert abs(float(first.advantages.mean())) < 1e-5


def test_buffer_requires_gae_before_minibatches() -> None:
    """Минибатчи до расчёта преимуществ — ошибка, а не мусор в градиенте."""
    buffer = RolloutBuffer(num_steps=2, num_envs=1, sem_shape=(2, 2), feature_dim=1)
    buffer.reset()
    _fill(buffer, [1.0, 1.0], [0.0, 0.0])
    with pytest.raises(RuntimeError):
        list(buffer.minibatches(1))
    with pytest.raises(RuntimeError):
        buffer.explained_variance()


def test_gae_without_terminals_is_discounted_sum() -> None:
    """Без завершений GAE(λ=1) при нулевом критике = сумма будущих наград."""
    buffer = RolloutBuffer(num_steps=3, num_envs=1, sem_shape=(2, 2), feature_dim=1)
    buffer.reset()
    _fill(buffer, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    buffer.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        last_dones=np.zeros(1, dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert buffer.advantages.reshape(-1).tolist() == pytest.approx([3.0, 2.0, 1.0])


def test_gae_cuts_at_episode_boundary() -> None:
    """Преимущество не имеет права протекать через конец эпизода (SPEC §11).

    Шаг 1 завершает эпизод (`dones[1] = 1`), значит:
      adv[2] = 1, adv[1] = 1 (будущего нет), adv[0] = 1 + adv[1] = 2.

    Регрессия: раньше `next_non_terminal` был сдвинут на шаг, и награда нового
    эпизода протекала в предыдущий — получалось [1, 2, 1].
    """
    buffer = RolloutBuffer(num_steps=3, num_envs=1, sem_shape=(2, 2), feature_dim=1)
    buffer.reset()
    _fill(buffer, [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    buffer.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        last_dones=np.zeros(1, dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert buffer.advantages.reshape(-1).tolist() == pytest.approx([2.0, 1.0, 1.0])


def test_gae_truncation_uses_bootstrap_only() -> None:
    """Обрыв по лимиту дооценивается ровно один раз — своим `bootstrap_value`.

    Шаг 0 оборван (`dones[0] = 1`, `bootstrap = 5`), значит adv[0] = 5.

    Регрессия: раньше к сохранённому `bootstrap_value` прибавлялась ещё и
    ценность первого кадра нового эпизода V(s_1) = 7, будущее считалось дважды.
    """
    buffer = RolloutBuffer(num_steps=2, num_envs=1, sem_shape=(2, 2), feature_dim=1)
    buffer.reset()
    _fill(buffer, [0.0, 0.0], [1.0, 0.0], bootstraps=[5.0, 0.0])
    buffer.compute_returns_and_advantages(
        last_values=np.full(1, 7.0, dtype=np.float32),
        last_dones=np.zeros(1, dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert buffer.advantages.reshape(-1)[0] == pytest.approx(5.0)


def test_explained_variance_is_perfect_for_exact_critic() -> None:
    """Идеальный критик даёт explained_variance == 1."""
    buffer = RolloutBuffer(num_steps=3, num_envs=1, sem_shape=(2, 2), feature_dim=1)
    buffer.reset()
    _fill(buffer, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], values=[6.0, 5.0, 3.0])
    buffer.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        last_dones=np.zeros(1, dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert buffer.explained_variance() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# учебный план
# ---------------------------------------------------------------------------
def test_curriculum_requires_full_window() -> None:
    """Ступень не повышается, пока окно эпизодов не набрано целиком."""
    curriculum = Curriculum(CurriculumConfig(window=4, start_difficulty=0.1, step=0.1))
    for _ in range(3):
        curriculum.record_episode(True)
    assert curriculum.maybe_promote() is False
    curriculum.record_episode(True)
    assert curriculum.maybe_promote() is True
    assert curriculum.current_difficulty() == pytest.approx(0.2)
    # После повышения окно очищено — сразу второй раз подняться нельзя.
    assert curriculum.maybe_promote() is False


def test_curriculum_does_not_promote_on_failures() -> None:
    """Низкая доля прохождений ступень не двигает."""
    curriculum = Curriculum(CurriculumConfig(window=4, promote_success_rate=0.7))
    for finished in (True, False, False, False):
        curriculum.record_episode(finished)
    assert curriculum.maybe_promote() is False
    assert curriculum.success_rate() == pytest.approx(0.25)


def test_curriculum_stops_at_max() -> None:
    """Потолок сложности соблюдается."""
    curriculum = Curriculum(
        CurriculumConfig(window=2, start_difficulty=0.95, step=0.1, max_difficulty=1.0)
    )
    curriculum.record_episode(True)
    curriculum.record_episode(True)
    assert curriculum.maybe_promote() is True
    assert curriculum.current_difficulty() == pytest.approx(1.0)
    assert curriculum.at_max is True
    curriculum.record_episode(True)
    curriculum.record_episode(True)
    assert curriculum.maybe_promote() is False


def test_curriculum_state_round_trip() -> None:
    """Снимок плана восстанавливается — иначе дообучение начнёт ступень заново."""
    curriculum = Curriculum(CurriculumConfig(window=5, start_difficulty=0.2, step=0.1))
    for finished in (True, False, True):
        curriculum.record_episode(finished)
    state = curriculum.state_dict()

    restored = Curriculum(CurriculumConfig(window=5, start_difficulty=0.2, step=0.1))
    restored.load_state_dict(state)
    assert restored.current_difficulty() == pytest.approx(curriculum.current_difficulty())
    assert restored.success_rate() == pytest.approx(curriculum.success_rate())
    assert restored.window_size == curriculum.window_size


# ---------------------------------------------------------------------------
# векторная среда
# ---------------------------------------------------------------------------
def test_vector_env_auto_reset_and_final_observation(level_file: Path) -> None:
    """Авто-reset не теряет последний кадр эпизода: он уходит в `final_observation`."""
    cfg = EnvConfig(obs_mode="semantic", max_steps=8, level_path=str(level_file))
    vec = SyncVectorEnv(make_env_fns(cfg, 3, seed=0))
    try:
        obs, infos = vec.reset(seed=0)
        assert obs["semantic"].shape == (3, OBS_H, OBS_W)
        assert obs["features"].shape == (3, FEATURE_DIM)
        assert len(infos) == 3
        assert vec.num_envs == 3 and len(vec) == 3

        seen_final = False
        for _ in range(20):
            obs, rewards, terminated, truncated, infos = vec.step(np.zeros(3, dtype=int))
            assert rewards.shape == (3,)
            done = np.logical_or(terminated, truncated)
            for i in np.nonzero(done)[0]:
                assert "final_observation" in infos[i]
                assert "final_info" in infos[i]
                assert infos[i]["final_observation"]["semantic"].shape == (OBS_H, OBS_W)
                seen_final = True
        assert seen_final, "за 20 шагов ни один эпизод не завершился"
    finally:
        vec.close()


def test_vector_env_propagates_difficulty(level_file: Path) -> None:
    """`set_difficulty` доходит до каждой среды — иначе учебный план бессилен."""
    cfg = EnvConfig(obs_mode="semantic", difficulty=0.1, level_path=str(level_file))
    with SyncVectorEnv(make_env_fns(cfg, 2, seed=1)) as vec:
        vec.reset(seed=1)
        vec.set_difficulty(0.7)
        assert vec.difficulty == pytest.approx(0.7)
        assert all(env.difficulty == pytest.approx(0.7) for env in vec.envs)


def test_env_factories_use_distinct_seeds(level_file: Path) -> None:
    """Фабрики разводят seed по средам: одинаковый seed убил бы разнообразие опыта."""
    cfg = EnvConfig(obs_mode="semantic", level_path=str(level_file))
    fns = make_env_fns(cfg, 4, seed=123)
    seeds = [fn().config.seed for fn in fns]
    assert len(set(seeds)) == 4


def test_vector_env_rejects_wrong_action_count(level_file: Path) -> None:
    """Число действий обязано совпадать с числом сред."""
    cfg = EnvConfig(obs_mode="semantic", level_path=str(level_file))
    with SyncVectorEnv(make_env_fns(cfg, 2, seed=0)) as vec:
        vec.reset(seed=0)
        with pytest.raises(ValueError):
            vec.step(np.zeros(3, dtype=int))


# ---------------------------------------------------------------------------
# smoke-обучение
# ---------------------------------------------------------------------------
def test_train_agent_smoke(tmp_path: Path, level_file: Path) -> None:
    """2000 шагов обучения проходят без падений, лосс конечен, чекпойнт грузится."""
    out_dir = tmp_path / "run"
    cfg = PPOConfig(
        num_envs=2,
        rollout_steps=64,
        total_steps=2048,
        minibatches=2,
        epochs=2,
        device="cpu",
        out_dir=str(out_dir),
    )
    env_cfg = EnvConfig(
        obs_mode="semantic", max_steps=250, level_path=str(level_file), seed=0
    )
    result = train_agent(cfg, env_cfg)

    assert result["global_step"] >= 2000
    assert result["iterations"] == 16
    assert np.isfinite(result["policy_loss"])
    assert np.isfinite(result["value_loss"])
    assert np.isfinite(result["entropy"])
    assert result["episodes"] > 0
    assert 0.0 <= result["success_rate"] <= 1.0
    assert result["stop_reason"] == "completed"

    metrics_path = Path(result["metrics_path"])
    assert metrics_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert len(records) == 16
    for key in ("iteration", "global_step", "mean_reward", "entropy", "approx_kl", "difficulty"):
        assert key in records[0]

    last = Path(result["last_path"])
    assert last.exists()
    model = load_policy(last, device="cpu")
    logits, value = model(
        torch.zeros(1, NUM_CLASSES, POLICY_SEM_H, POLICY_SEM_W),
        torch.zeros(1, FEATURE_DIM),
    )
    assert logits.shape == (1, NUM_ACTIONS) and value.shape == (1,)


def test_train_agent_rejects_pixel_only_mode(tmp_path: Path, level_file: Path) -> None:
    """Политика учится на карте: `obs_mode='pixels'` обязан быть отвергнут явно."""
    cfg = PPOConfig(num_envs=1, rollout_steps=4, total_steps=8, device="cpu",
                    out_dir=str(tmp_path / "bad"))
    env_cfg = EnvConfig(obs_mode="pixels", level_path=str(level_file))
    with pytest.raises(ValueError, match="semantic"):
        train_agent(cfg, env_cfg)


def test_on_iteration_callback_can_stop_training(tmp_path: Path, level_file: Path) -> None:
    """Колбэк `on_iteration`, вернувший False, аккуратно останавливает обучение."""
    seen: list[int] = []

    def callback(metrics: dict) -> bool:
        seen.append(metrics["iteration"])
        return len(seen) < 2

    cfg = PPOConfig(
        num_envs=1, rollout_steps=16, total_steps=16 * 10, minibatches=1, epochs=1,
        device="cpu", out_dir=str(tmp_path / "stop"),
    )
    env_cfg = EnvConfig(
        obs_mode="semantic", max_steps=120, level_path=str(level_file), seed=0
    )
    result = train_agent(cfg, env_cfg, on_iteration=callback)
    assert seen == [1, 2]
    assert result["stop_reason"] == "stopped_by_callback"
    assert Path(result["last_path"]).exists()


@pytest.mark.slow
def test_train_agent_with_curriculum(tmp_path: Path, level_file: Path) -> None:
    """Учебный план подключается к обучению и попадает в итоговые метрики."""
    cfg = PPOConfig(
        num_envs=2, rollout_steps=32, total_steps=256, minibatches=2, epochs=1,
        device="cpu", out_dir=str(tmp_path / "curr"),
    )
    env_cfg = EnvConfig(
        obs_mode="semantic", max_steps=150, level_path=str(level_file), seed=1
    )
    result = train_agent(cfg, env_cfg, CurriculumConfig(window=4, step=0.1))
    assert "curriculum" in result
    assert 0.0 <= result["curriculum"]["difficulty"] <= 1.0
