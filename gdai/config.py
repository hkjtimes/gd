"""Конфигурации всех подсистем — только данные, никакой логики.

Зачем отдельный модуль: конфиг кладётся в чекпойнт целиком (`asdict`), поэтому
он обязан быть простым, сериализуемым в JSON и не тянуть за собой torch. Так
любой сохранённый прогон воспроизводится по одному файлу.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Параметры среды: что агент видит, как долго живёт эпизод и чем платят."""

    obs_mode: str = "semantic"      # "semantic" | "pixels" | "both"
    max_steps: int = 6000
    difficulty: float = 0.3         # 0..1, для процедурной генерации
    level_path: str | None = None   # если задан — фиксированный уровень из файла
    seed: int | None = None
    randomize_theme: bool = True    # менять тему на каждом эпизоде (для obs_mode с pixels)
    decoration_level: float = 1.0   # 0 = голый уровень, 1 = максимум декора
    practice_checkpoints: bool = True
    checkpoint_prob: float = 0.5    # вероятность старта с чекпойнта
    semantic_noise: float = 0.0     # вероятность порчи пикселя карты (робастность политики)
    reward_progress: float = 1.0
    reward_death: float = -1.0
    reward_finish: float = 10.0
    reward_alive: float = 0.0


@dataclass
class PerceptionConfig:
    """Параметры обучения зрения (U-Net на синтетике с доменной рандомизацией)."""

    base_channels: int = 24
    depth: int = 3
    steps: int = 4000
    batch_size: int = 16
    lr: float = 3e-4
    val_every: int = 250
    device: str = "auto"
    out_dir: str = "runs/perception"
    augment: bool = True


@dataclass
class PPOConfig:
    """Параметры PPO: политика учится только на канонических картах."""

    num_envs: int = 8
    rollout_steps: int = 256
    total_steps: int = 2_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    out_dir: str = "runs/agent"


@dataclass
class CurriculumConfig:
    """Учебный план: сложность растёт, только когда агент реально справляется."""

    start_difficulty: float = 0.05
    max_difficulty: float = 1.0
    step: float = 0.05
    promote_success_rate: float = 0.7   # доля пройденных уровней для повышения
    window: int = 50


__all__ = ["EnvConfig", "PerceptionConfig", "PPOConfig", "CurriculumConfig"]
