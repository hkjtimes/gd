"""PPO: обучение политики на канонических картах.

Почему именно PPO
-----------------
Задача дискретная (два действия), эпизоды короткие, а среда дешёвая — здесь
выигрывает on-policy метод с большим числом параллельных сред. PPO вдобавок
терпим к грубо подобранным гиперпараметрам: клип отношения вероятностей не даёт
одному удачному эпизоду перекосить политику настолько, что она перестанет
исследовать. Для игры, где ошибка на одном кадре из шестидесяти убивает,
это решающее свойство.

Что здесь принципиально
-----------------------
* **Преимущества нормализуются по минибатчу.** Масштаб наград меняется на
  порядки: в начале обучения агент собирает доли тайла прогресса, после первых
  прохождений — десятки. Без нормализации один learning rate не годится для
  обеих фаз.
* **Clipped value loss.** Критик обучается на своих же прошлых оценках, и без
  ограничения шага он способен «убежать» за одну эпоху, после чего преимущества
  превращаются в шум.
* **Ранняя остановка по KL.** Четыре эпохи по одному куску опыта иногда уводят
  политику слишком далеко от той, что этот опыт собирала. Как только средний
  approx_kl превышает порог, эпохи прекращаются — это дешевле, чем
  восстанавливаться после развала политики.
* **Затухание learning rate.** К концу обучения нужна точность в конкретных
  кадрах прыжка, а не крупные шаги.
* **Успех считается только по эпизодам с начала уровня.** Среда умеет
  практиковать сложные места, начиная эпизод с чекпойнта; считать такие
  прохождения победами — значит обманывать и учебный план, и себя.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from gdai.agent.buffer import RolloutBuffer
from gdai.agent.curriculum import Curriculum
from gdai.agent.networks import (
    POLICY_SEM_H,
    POLICY_SEM_W,
    ActorCritic,
    indices_to_tensor,
    semantic_to_indices,
)
from gdai.agent.vecenv import SyncVectorEnv, make_env_fns
from gdai.config import CurriculumConfig, EnvConfig, PPOConfig
from gdai.env.gd_env import FEATURE_DIM
from gdai.utils.checkpoint import load_checkpoint, save_checkpoint
from gdai.utils.logging import JsonlLogger, get_logger
from gdai.utils.seeding import make_rng, seed_from, set_global_seed

_log = get_logger("agent.ppo")

# Порог ранней остановки эпох по расхождению политик. 0.03 — компромисс:
# меньше режет обучение почти всегда, больше перестаёт защищать.
DEFAULT_TARGET_KL: float = 0.03
# Нижняя граница затухания lr: полностью замороженная политика в последних
# итерациях перестала бы доучивать как раз самые тонкие места.
LR_FLOOR_FRACTION: float = 0.1

# --- адаптация шага по KL ---------------------------------------------------
# Зачем вообще: в этой игре сигнал политики крайне слаб. Действие влияет на
# исход только в те кадры, когда куб стоит на земле, а эксперт (солвер) нажимает
# кнопку примерно на 1% кадров — всё остальное чистый шум в градиенте. Замер:
# при lr=3e-4 (дефолт SPEC §3) наблюдаемый approx_kl держится около 1e-4, то
# есть в ТРИСТА раз меньше разрешённого `target_kl`. Политика буквально стоит на
# месте: за 100k шагов на одном уровне прогресс не сдвигался с 0.49 (уровень
# случайной политики). Поднятый вручную lr=3e-3 на том же прогоне давал 15%
# прохождений — то есть дело не в алгоритме, а в размере шага.
#
# Прибивать другой lr константой нельзя: он записан в контракте конфигурации, и
# для другой награды/архитектуры «правильное» значение будет иным. Поэтому шаг
# ведёт регулятор: `cfg.lr` — стартовая точка, а множитель подстраивается так,
# чтобы фактический KL держался около `target_kl`. Это классический адаптивный
# PPO, только регулируется lr, а не коэффициент штрафа.
# Регулятор держит KL в КОРИДОРЕ, а не «лишь бы не выше цели». Почему коридор:
# при правиле «поднимать, пока KL < target» множитель залипал на потолке даже
# после того, как политика научилась играть (KL 0.01-0.02 при цели 0.03), и на
# 240k шагов прогон разваливался — доля прохождений падала с 0.89 до 0.19 за
# двадцать итераций. Верхняя граница коридора вдвое ниже `target_kl`, который
# остаётся аварийным порогом ранней остановки эпох, а не рабочей точкой.
LR_ADAPT_UP: float = 1.3      # во сколько раз поднимать шаг, если KL мал
LR_ADAPT_DOWN: float = 2.0    # во сколько раз ронять, если KL велик (быстрее, чем растим)
LR_SCALE_MAX: float = 12.0    # потолок множителя к cfg.lr
LR_SCALE_MIN: float = 0.2     # пол множителя
LR_ADAPT_LOW_FRAC: float = 0.25   # ниже этой доли target_kl шаг «слишком мал»
LR_ADAPT_HIGH_FRAC: float = 0.5   # выше этой доли — «слишком велик»

# --- регулятор бонуса за энтропию -------------------------------------------
# Оптимальная политика здесь почти детерминирована: эксперт (солвер) нажимает
# кнопку примерно на 1% кадров, то есть его энтропия около 0.06 нат при
# максимуме ln(2)=0.693. Постоянный бонус 0.01 (дефолт SPEC §3) держит политику
# у равномерной, а равномерная политика для куба означает «прыгать при каждом
# приземлении»: игрок почти всё время в воздухе, где действие не влияет вообще
# ни на что, — и подавляющая часть собранного опыта превращается в шум.
# Замер на фиксированном уровне за 60k шагов: coef=0.01 -> прогресс 0.49
# (ровно уровень случайной политики), 0.003 -> 0.62, 0.001 -> 0.66.
#
# Фиксировать «правильный» коэффициент числом нельзя: он зависит от масштаба
# преимуществ, который сам меняется по ходу обучения. Поэтому регулируется не
# коэффициент, а САМА энтропия: цель плавно едет от «почти равномерно» к
# «почти детерминированно», а `cfg.entropy_coef` задаёт лишь опорный масштаб
# бонуса, вокруг которого регулятор двигается.
# Поэтому бонус линейно затухает: он нужен в начале, пока идёт исследование, и
# только мешает в конце, когда пора закреплять найденное. Пробовался и полный
# регулятор «по целевой энтропии» — на замерах он оказался хуже простого
# затухания (прогресс 0.52 против 0.61), потому что сбрасывал бонус в пол за
# несколько итераций и терял исследование в самом начале.
ENTROPY_FINAL_FRACTION: float = 0.1
# По скольким последним эпизодам считаются метрики отчётности.
STAT_WINDOW: int = 100
# Как часто печатать человекочитаемую строку прогресса (секунды).
LOG_INTERVAL_SECONDS: float = 30.0
# eps у Adam: 1e-5 вместо 1e-8 — классическая для PPO поправка, спасает от
# гигантских шагов, когда градиенты почти нулевые (а в начале они такие).
ADAM_EPS: float = 1e-5


def resolve_device(name: str = "auto") -> torch.device:
    """Превратить `"auto"` в реальное устройство (cuda, если есть, иначе cpu)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_policy(
    path: str | Path, device: str = "auto", strict: bool = True
) -> ActorCritic:
    """Поднять обученную политику из чекпойнта `best.pt`/`last.pt`.

    Зачем в этом модуле: архитектура задаётся не только классом, но и
    гиперпараметрами (`hidden`, `feature_dim`), которые сохраняются рядом с
    весами. Собирать сеть «на глаз» на стороне потребителя — верный способ
    получить несовпадение форм после первого же изменения конфигурации.
    """
    payload = load_checkpoint(path, map_location="cpu")
    arch = payload.get("arch") or {}
    model = ActorCritic(
        num_classes=int(arch.get("num_classes", 10)),
        feature_dim=int(arch.get("feature_dim", FEATURE_DIM)),
        hidden=int(arch.get("hidden", 256)),
        num_actions=int(arch.get("num_actions", 2)),
    )
    model.load_state_dict(payload["state_dict"], strict=strict)
    model.to(resolve_device(device))
    model.eval()
    return model


def _mean(values: Any, default: float = 0.0) -> float:
    """Среднее по последовательности, устойчивое к пустоте (метрики стартуют пустыми)."""
    seq = list(values)
    if not seq:
        return float(default)
    return float(np.mean(seq))


class _EpisodeStats:
    """Скользящая статистика по завершённым эпизодам.

    Успех считается двумя способами: `success_rate` — только по эпизодам,
    сыгранным с начала уровня (честное «прошёл уровень»), `success_rate_all` —
    по всем, включая практику с чекпойнта. Первое — критерий приёмки и вход
    учебного плана, второе показывает, помогает ли практика вообще.
    """

    def __init__(self, window: int = STAT_WINDOW) -> None:
        self.returns: deque[float] = deque(maxlen=window)
        self.lengths: deque[int] = deque(maxlen=window)
        self.progress: deque[float] = deque(maxlen=window)
        self.full_runs: deque[bool] = deque(maxlen=window)
        self.all_runs: deque[bool] = deque(maxlen=window)
        self.total_episodes: int = 0
        self.total_finished: int = 0

    def record(self, info: dict[str, Any]) -> bool:
        """Учесть завершённый эпизод; вернуть True, если он был «с начала уровня»."""
        finished = bool(info.get("finished", False))
        from_checkpoint = bool(info.get("from_checkpoint", False))
        self.returns.append(float(info.get("episode_return", 0.0)))
        self.lengths.append(int(info.get("steps", 0)))
        self.progress.append(float(info.get("progress", 0.0)))
        self.all_runs.append(finished)
        self.total_episodes += 1
        self.total_finished += int(finished)
        if not from_checkpoint:
            self.full_runs.append(finished)
            return True
        return False

    @property
    def success_rate(self) -> float:
        """Доля пройденных целиком уровней среди эпизодов «с начала»."""
        if not self.full_runs:
            return 0.0
        return sum(1 for ok in self.full_runs if ok) / len(self.full_runs)

    @property
    def success_rate_all(self) -> float:
        """Доля прохождений среди всех эпизодов (включая практику)."""
        if not self.all_runs:
            return 0.0
        return sum(1 for ok in self.all_runs if ok) / len(self.all_runs)


def train_agent(
    cfg: PPOConfig,
    env_cfg: EnvConfig,
    curriculum_cfg: CurriculumConfig | None = None,
    on_iteration: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Обучить политику PPO и вернуть итоговые метрики.

    * `cfg` — гиперпараметры PPO; `env_cfg` — конфигурация каждой из
      `cfg.num_envs` сред (seed разводится по средам автоматически).
    * `curriculum_cfg` — если задан, сложность ведёт учебный план, а
      `env_cfg.difficulty` служит только начальным значением до первого
      `reset`.
    * `on_iteration(metrics)` — колбэк после каждой итерации (для CLI-прогресса
      и тестов). Если он вернёт `False`, обучение аккуратно завершится: это
      единственный способ остановить длинный прогон снаружи, не убивая процесс
      и не теряя чекпойнт.

    Побочные эффекты: `out_dir/metrics.jsonl`, `out_dir/last.pt`,
    `out_dir/best.pt` (лучший по success_rate, при равенстве — по награде).
    """
    if env_cfg.obs_mode == "pixels":
        raise ValueError(
            "Политика обучается на семантической карте: obs_mode='pixels' не "
            "даёт ключа 'semantic'. Используйте 'semantic' (быстро) или 'both'."
        )

    device = resolve_device(cfg.device)
    seed = int(env_cfg.seed) if env_cfg.seed is not None else 0
    set_global_seed(seed)
    shuffle_rng = make_rng(seed_from("ppo.minibatch", seed))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = Curriculum(curriculum_cfg) if curriculum_cfg is not None else None
    start_difficulty = (
        curriculum.current_difficulty() if curriculum is not None else env_cfg.difficulty
    )
    envs_cfg = replace(env_cfg, difficulty=float(start_difficulty))

    vec = SyncVectorEnv(make_env_fns(envs_cfg, cfg.num_envs, seed))
    model = ActorCritic(feature_dim=FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), eps=ADAM_EPS)
    buffer = RolloutBuffer(
        num_steps=int(cfg.rollout_steps),
        num_envs=int(cfg.num_envs),
        sem_shape=(POLICY_SEM_H, POLICY_SEM_W),
        feature_dim=FEATURE_DIM,
        device=device,
    )
    logger = JsonlLogger(out_dir, append=False)

    steps_per_iter = int(cfg.num_envs) * int(cfg.rollout_steps)
    iterations = max(1, int(cfg.total_steps) // steps_per_iter)
    target_kl = float(getattr(cfg, "target_kl", DEFAULT_TARGET_KL))
    anneal_lr = bool(getattr(cfg, "anneal_lr", True))

    _log.info(
        "PPO: device=%s, envs=%d, rollout=%d, итераций=%d (по %d шагов), "
        "параметров=%d, сложность=%.2f%s",
        device,
        cfg.num_envs,
        cfg.rollout_steps,
        iterations,
        steps_per_iter,
        model.num_parameters(),
        start_difficulty,
        ", учебный план включён" if curriculum is not None else "",
    )

    stats = _EpisodeStats()
    obs, _infos = vec.reset(seed=seed)
    next_sem = semantic_to_indices(obs["semantic"])
    next_feat = np.asarray(obs["features"], dtype=np.float32)
    next_done = np.zeros(cfg.num_envs, dtype=np.float32)

    global_step = 0
    best_key: tuple[float, float] = (-1.0, -np.inf)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    # Множитель шага, который ведёт регулятор по KL (см. LR_ADAPT_*).
    lr_scale = 1.0
    entropy_coef_now = float(cfg.entropy_coef)
    t_start = time.time()
    t_log = t_start
    stop_reason = "completed"

    def _save(path: Path, extra_meta: dict[str, Any]) -> None:
        """Записать чекпойнт со всем, что нужно для дообучения и инференса."""
        save_checkpoint(
            path,
            model.state_dict(),
            config=asdict(cfg),
            meta={"kind": "policy", **extra_meta},
            extra={
                "arch": model.config_dict(),
                "env_config": asdict(env_cfg),
                "curriculum": curriculum.state_dict() if curriculum else None,
                "optimizer_state": optimizer.state_dict(),
                "global_step": global_step,
            },
        )

    try:
        for iteration in range(1, iterations + 1):
            t_iter = time.time()

            progress_frac = (iteration - 1) / iterations
            if anneal_lr:
                frac = max(1.0 - progress_frac, LR_FLOOR_FRACTION)
            else:
                frac = 1.0
            lr_now = float(cfg.lr) * frac * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            entropy_coef_now = float(cfg.entropy_coef) * (
                1.0 - progress_frac * (1.0 - ENTROPY_FINAL_FRACTION)
            )

            # ---------- сбор опыта ----------
            buffer.reset()
            model.eval()
            for _step in range(int(cfg.rollout_steps)):
                sem_t = indices_to_tensor(next_sem, device=device)
                feat_t = torch.as_tensor(next_feat, device=device)
                action, log_prob, value = model.act(sem_t, feat_t)
                actions_np = action.cpu().numpy()

                obs, rewards, terminated, truncated, infos = vec.step(actions_np)
                done = np.logical_or(terminated, truncated)
                global_step += int(cfg.num_envs)

                # Эпизоды, оборванные лимитом шагов, не закончились — их будущее
                # дооценивается критиком по последнему кадру.
                bootstraps = np.zeros(cfg.num_envs, dtype=np.float32)
                trunc_idx = np.nonzero(truncated)[0]
                if trunc_idx.size:
                    final_sem = np.stack(
                        [infos[i]["final_observation"]["semantic"] for i in trunc_idx]
                    )
                    final_feat = np.stack(
                        [infos[i]["final_observation"]["features"] for i in trunc_idx]
                    ).astype(np.float32)
                    with torch.no_grad():
                        _logits, final_value = model(
                            indices_to_tensor(
                                semantic_to_indices(final_sem), device=device
                            ),
                            torch.as_tensor(final_feat, device=device),
                        )
                    bootstraps[trunc_idx] = final_value.cpu().numpy()

                buffer.add(
                    sem=next_sem,
                    features=next_feat,
                    actions=actions_np,
                    log_probs=log_prob.cpu().numpy(),
                    values=value.cpu().numpy(),
                    rewards=rewards,
                    dones=done.astype(np.float32),
                    bootstrap_values=bootstraps,
                )

                for i in np.nonzero(done)[0]:
                    final_info = infos[i].get("final_info", infos[i])
                    is_full_run = stats.record(final_info)
                    if curriculum is not None and is_full_run:
                        curriculum.record_episode(bool(final_info.get("finished")))

                next_sem = semantic_to_indices(obs["semantic"])
                next_feat = np.asarray(obs["features"], dtype=np.float32)
                next_done = done.astype(np.float32)

            # ---------- GAE ----------
            with torch.no_grad():
                _logits, last_value = model(
                    indices_to_tensor(next_sem, device=device),
                    torch.as_tensor(next_feat, device=device),
                )
            buffer.compute_returns_and_advantages(
                last_values=last_value.cpu().numpy(),
                last_dones=next_done,
                gamma=float(cfg.gamma),
                gae_lambda=float(cfg.gae_lambda),
            )

            # ---------- оптимизация ----------
            model.train()
            pg_losses: list[float] = []
            v_losses: list[float] = []
            entropies: list[float] = []
            kls: list[float] = []
            clip_fracs: list[float] = []
            epochs_done = 0
            for _epoch in range(int(cfg.epochs)):
                epoch_kls: list[float] = []
                for mb in buffer.minibatches(int(cfg.minibatches), shuffle_rng):
                    sem_mb = indices_to_tensor(mb.sem, device=device)
                    new_logp, entropy, new_value = model.evaluate_actions(
                        sem_mb, mb.features, mb.actions
                    )
                    log_ratio = new_logp - mb.log_probs
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # Оценка KL по Шульману: несмещённее и всегда >= 0,
                        # в отличие от простого -mean(log_ratio).
                        approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                        clip_frac = float(
                            ((ratio - 1.0).abs() > float(cfg.clip_eps)).float().mean()
                        )

                    pg_loss = torch.max(
                        -mb.advantages * ratio,
                        -mb.advantages
                        * torch.clamp(
                            ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps)
                        ),
                    ).mean()

                    # Клип ценности вокруг старой оценки — та же логика доверия,
                    # что и у политики: критик не должен прыгать за одну эпоху.
                    v_unclipped = (new_value - mb.returns) ** 2
                    v_clipped = mb.values + torch.clamp(
                        new_value - mb.values,
                        -float(cfg.clip_eps),
                        float(cfg.clip_eps),
                    )
                    v_loss = 0.5 * torch.max(
                        v_unclipped, (v_clipped - mb.returns) ** 2
                    ).mean()

                    entropy_mean = entropy.mean()
                    loss = (
                        pg_loss
                        - entropy_coef_now * entropy_mean
                        + float(cfg.value_coef) * v_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.max_grad_norm)
                    )
                    optimizer.step()

                    pg_losses.append(float(pg_loss.detach()))
                    v_losses.append(float(v_loss.detach()))
                    entropies.append(float(entropy_mean.detach()))
                    kls.append(approx_kl)
                    epoch_kls.append(approx_kl)
                    clip_fracs.append(clip_frac)
                epochs_done += 1
                if target_kl > 0.0 and _mean(epoch_kls) > target_kl:
                    # Политика ушла слишком далеко от той, что собрала данные —
                    # оставшиеся эпохи только навредят.
                    break

            # ---------- регулятор шага по KL ----------
            # Держим фактическое расхождение политик в коридоре
            # [0.25, 0.5] * target_kl. Слишком маленький KL означает, что шаг
            # почти ничего не меняет (именно в этом режиме политика не могла
            # сдвинуться с равномерной), слишком большой — что политика
            # убегает от собранных данных и прогон рискует развалиться.
            iter_kl = _mean(kls)
            if target_kl > 0.0:
                if iter_kl < LR_ADAPT_LOW_FRAC * target_kl:
                    lr_scale = min(lr_scale * LR_ADAPT_UP, LR_SCALE_MAX)
                elif iter_kl > LR_ADAPT_HIGH_FRAC * target_kl:
                    lr_scale = max(lr_scale / LR_ADAPT_DOWN, LR_SCALE_MIN)

            # ---------- учебный план ----------
            promoted = False
            if curriculum is not None and curriculum.maybe_promote():
                vec.set_difficulty(curriculum.current_difficulty())
                promoted = True

            # ---------- метрики ----------
            now = time.time()
            iter_seconds = max(now - t_iter, 1e-9)
            metrics = {
                "iteration": iteration,
                "global_step": global_step,
                "mean_reward": _mean(stats.returns),
                "mean_ep_len": _mean(stats.lengths),
                "mean_progress": _mean(stats.progress),
                "success_rate": stats.success_rate,
                "success_rate_all": stats.success_rate_all,
                "episodes": stats.total_episodes,
                "difficulty": vec.difficulty,
                "promoted": promoted,
                "policy_loss": _mean(pg_losses),
                "value_loss": _mean(v_losses),
                "entropy": _mean(entropies),
                "approx_kl": _mean(kls),
                "clip_fraction": _mean(clip_fracs),
                "explained_variance": buffer.explained_variance(),
                "epochs_done": epochs_done,
                "lr": lr_now,
                "lr_scale": lr_scale,
                "entropy_coef": entropy_coef_now,
                "fps": steps_per_iter / iter_seconds,
                "elapsed": now - t_start,
            }
            logger.log(metrics)
            history.append(metrics)

            # Лучший чекпойнт — сначала по доле прохождений, при равенстве по
            # средней награде: доля прохождений квантована окном эпизодов, и
            # без второго ключа сотни итераций считались бы одинаковыми.
            key = (metrics["success_rate"], metrics["mean_reward"])
            if stats.total_episodes > 0 and key > best_key:
                best_key = key
                _save(
                    best_path,
                    {
                        "best": True,
                        "iteration": iteration,
                        "success_rate": metrics["success_rate"],
                        "mean_reward": metrics["mean_reward"],
                    },
                )
            _save(last_path, {"best": False, "iteration": iteration})

            if now - t_log >= LOG_INTERVAL_SECONDS or iteration in (1, iterations):
                t_log = now
                _log.info(
                    "it %d/%d  шаг %d  reward %.2f  успех %.2f (all %.2f)  "
                    "прогресс %.2f  длина %.0f  d=%.2f  H=%.3f  KL=%.4f  %.0f fps",
                    iteration,
                    iterations,
                    global_step,
                    metrics["mean_reward"],
                    metrics["success_rate"],
                    metrics["success_rate_all"],
                    metrics["mean_progress"],
                    metrics["mean_ep_len"],
                    metrics["difficulty"],
                    metrics["entropy"],
                    metrics["approx_kl"],
                    metrics["fps"],
                )

            if on_iteration is not None and on_iteration(metrics) is False:
                stop_reason = "stopped_by_callback"
                _log.info("обучение остановлено колбэком on_iteration")
                break

    except KeyboardInterrupt:  # pragma: no cover - интерактивный сценарий
        stop_reason = "interrupted"
        _log.warning("прервано с клавиатуры — сохраняю last.pt")
    finally:
        _save(last_path, {"best": False, "final": True})
        logger.close()
        vec.close()

    total_seconds = time.time() - t_start
    result: dict[str, Any] = {
        "global_step": global_step,
        "iterations": len(history),
        "elapsed": total_seconds,
        "fps": global_step / max(total_seconds, 1e-9),
        "mean_reward": metrics.get("mean_reward", 0.0),
        "mean_ep_len": metrics.get("mean_ep_len", 0.0),
        "mean_progress": metrics.get("mean_progress", 0.0),
        "success_rate": stats.success_rate,
        "success_rate_all": stats.success_rate_all,
        "best_success_rate": max(best_key[0], 0.0),
        "episodes": stats.total_episodes,
        "finished_episodes": stats.total_finished,
        "difficulty": metrics.get("difficulty", start_difficulty),
        "entropy": metrics.get("entropy", 0.0),
        "approx_kl": metrics.get("approx_kl", 0.0),
        "policy_loss": metrics.get("policy_loss", 0.0),
        "value_loss": metrics.get("value_loss", 0.0),
        "explained_variance": metrics.get("explained_variance", 0.0),
        "out_dir": str(out_dir),
        "best_path": str(best_path) if best_path.exists() else None,
        "last_path": str(last_path),
        "metrics_path": str(logger.path),
        "stop_reason": stop_reason,
    }
    if curriculum is not None:
        result["curriculum"] = curriculum.state_dict()
    _log.info(
        "готово: %d шагов за %.1f c (%.0f fps), успех %.2f, лучший %.2f",
        global_step,
        total_seconds,
        result["fps"],
        result["success_rate"],
        result["best_success_rate"],
    )
    return result


__all__ = ["train_agent", "load_policy", "resolve_device", "DEFAULT_TARGET_KL"]
