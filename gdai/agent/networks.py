"""Actor-critic политики: свёрточная сеть поверх канонической карты.

Зачем именно так
----------------
Политика принципиально не видит пикселей: на вход идёт **семантическая карта**
(классы 0..9), развёрнутая в one-hot. Это не косметика, а способ сделать вход
инвариантным к дизайну — цвет, свечение и партиклы до сюда не доходят вообще,
поэтому переобучиться на них нельзя.

One-hot, а не «класс как число», потому что классы — это категории, а не шкала:
для сети `HAZARD=2` не должен быть «в два раза больше» `SOLID=1` и «между»
пустотой и игроком. Каналы делают каждый класс независимым признаком, а свёртка
поверх них учится геометрии («шип на моей высоте через три тайла»).

Карта ужимается вдвое (72x128 -> 36x64) до one-hot: при `PX_PER_TILE=8` один
тайл занимает 4 пикселя и после сжатия, а вычислений становится вчетверо
меньше. Сжатие приоритетное (`downsample_semantic`) — шип шириной в пару
пикселей не имеет права исчезнуть.

Геометрия свёрток подобрана под тайлы: первый слой (ядро 6, шаг 4, паддинг 1)
даёт ровно 9x16 клеток — сетку тайлов камеры (`VIEW_TILES_H` x `VIEW_TILES_W`)
с перекрытием рецептивных полей. То есть первый же слой рассуждает в единицах
игры, а не в случайных «пятнах».

Вектор `features` (скорость по вертикали, режим, гравитация, прогресс)
подмешивается после свёрток: по статичной карте не понять, летит игрок вверх
или вниз, а от этого зависит противоположное действие.

Инициализация ортогональная: голова политики с gain 0.01, чтобы стартовая
политика была почти равномерной (иначе PPO первые тысячи шагов лечит
собственный перекос), голова ценности с gain 1.0, свёртки и MLP — sqrt(2)
под ReLU.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdai.constants import (
    ACTION_HOLD,
    NUM_ACTIONS,
    NUM_CLASSES,
    OBS_H,
    OBS_W,
    VIEW_TILES_H,
    VIEW_TILES_W,
)
from gdai.env.semantic import class_priority, downsample_semantic

# Во сколько раз политика сжимает каноническую карту (SPEC §11: "one-hot + downsample x2").
POLICY_DOWNSAMPLE: int = 2
# Форма входа политики после сжатия: 36x64.
POLICY_SEM_H: int = OBS_H // POLICY_DOWNSAMPLE
POLICY_SEM_W: int = OBS_W // POLICY_DOWNSAMPLE

# Ортогональные gain'ы: sqrt(2) — канон для ReLU, 0.01 — «почти равномерная»
# стартовая политика, 1.0 — честная шкала для ценности.
GAIN_HIDDEN: float = math.sqrt(2.0)
GAIN_POLICY: float = 0.01
GAIN_VALUE: float = 1.0

# Априорная вероятность «держать» у необученной политики.
#
# Зачем не 0.5 (равномерно). У куба удержание что-то меняет ТОЛЬКО в кадрах,
# когда он стоит на земле; в полёте действие не влияет ни на что, и такой кадр
# даёт в градиенте чистый шум. А удержание само по себе отправляет куба в полёт
# — и чем чаще политика жмёт, тем меньше остаётся кадров, на которых вообще
# можно чему-то научиться. Замер на сгенерированных уровнях: при p=0.5 доля
# кадров «на земле» 6.9%, при p=0.1 — 22%, при p=0.02 — 55%. То есть
# равномерный старт сам себе выкалывает глаза: 93% опыта не несёт информации о
# действии.
#
# Значение согласовано с поведением эксперта: путь, найденный солвером,
# содержит удержание примерно на 1% кадров. Берём 0.1 — на порядок ближе к
# правде, чем 0.5, но с запасом на исследование (за окно прыжка в ~15 кадров
# политика всё ещё нажмёт с вероятностью ~0.8).
HOLD_PRIOR: float = 0.1


def layer_init(layer: nn.Module, gain: float = GAIN_HIDDEN, bias: float = 0.0) -> nn.Module:
    """Ортогонально инициализировать слой и обнулить смещение.

    Зачем ортогональность: она сохраняет норму сигнала при проходе через слой,
    поэтому глубокая сеть не взрывается и не затухает на первых шагах, когда
    градиенты PPO ещё шумные. Возвращает тот же слой — удобно оборачивать
    прямо в `nn.Sequential`.
    """
    weight = getattr(layer, "weight", None)
    if weight is not None:
        nn.init.orthogonal_(weight, gain)
    bias_param = getattr(layer, "bias", None)
    if bias_param is not None:
        nn.init.constant_(bias_param, bias)
    return layer


# Таблицы приоритета классов для пакетного сжатия. Строятся из публичной
# `class_priority`, то есть по определению совпадают с `downsample_semantic`:
# копия логики здесь только ради векторизации сразу по всей партии сред.
_PRIORITY_RANK: np.ndarray = np.array(
    [class_priority(c) for c in range(NUM_CLASSES)], dtype=np.uint8
)
_PRIORITY_CLASS: np.ndarray = np.zeros(int(_PRIORITY_RANK.max()) + 1, dtype=np.uint8)
_PRIORITY_CLASS[_PRIORITY_RANK] = np.arange(NUM_CLASSES, dtype=np.uint8)


def semantic_to_indices(sem_uint8: np.ndarray) -> np.ndarray:
    """Сжать каноническую карту вдвое, оставив классы числами (uint8).

    Зачем отдельный шаг перед one-hot: в буфере роллаута хранится именно эта
    компактная форма. One-hot той же партии занял бы в 40 раз больше памяти
    (10 float-каналов вместо одного байта), а разворачивать его дешевле в
    момент обучения минибатча.

    Принимает (H, W) или (B, H, W); возвращает (36, 64) или (B, 36, 64).
    Одиночная карта уходит в `downsample_semantic` (эталон), партия считается
    векторизованно — иначе сбор опыта восемью средами платил бы за питоновский
    цикл на каждом кадре.
    """
    arr = np.asarray(sem_uint8)
    if arr.ndim == 2:
        return downsample_semantic(arr, POLICY_DOWNSAMPLE)
    if arr.ndim != 3:
        raise ValueError(f"Ожидалась карта (H, W) или (B, H, W), получено {arr.shape}")
    b, h, w = arr.shape
    f = POLICY_DOWNSAMPLE
    if h % f or w % f:
        raise ValueError(
            f"Размер карты {h}x{w} не делится на {f} — сжатие исказило бы геометрию"
        )
    ranks = _PRIORITY_RANK[arr.astype(np.intp, copy=False)]
    blocks = ranks.reshape(b, h // f, f, w // f, f)
    best = blocks.max(axis=4).max(axis=2)
    return _PRIORITY_CLASS[best]


def indices_to_tensor(
    indices: np.ndarray | Tensor,
    device: torch.device | str | None = None,
    num_classes: int = NUM_CLASSES,
) -> Tensor:
    """Развернуть карту классов в one-hot тензор (B, C, H, W) float32.

    Зачем `scatter_`, а не `F.one_hot`: последний сначала строит int64-тензор
    того же объёма (в восемь раз тяжелее итогового float32), и на минибатче в
    512 карт это лишние сотни мегабайт трафика памяти на каждом шаге.
    """
    idx = torch.as_tensor(indices)
    if idx.ndim == 2:
        idx = idx.unsqueeze(0)
    if idx.ndim != 3:
        raise ValueError(f"Ожидались индексы (B, H, W), получено {tuple(idx.shape)}")
    idx = idx.to(device=device, dtype=torch.long, copy=False)
    b, h, w = idx.shape
    out = torch.zeros((b, num_classes, h, w), dtype=torch.float32, device=idx.device)
    out.scatter_(1, idx.unsqueeze(1), 1.0)
    return out


def semantic_to_tensor(
    sem_uint8: np.ndarray, device: torch.device | str | None = None
) -> Tensor:
    """Каноническая карта -> вход политики: сжатие вдвое + one-hot (SPEC §11).

    Зачем публичная функция: ровно этим преобразованием пользуются и обучение,
    и `pipeline.GDAgent`, и визуализация внимания. Если бы каждый делал его
    сам, малейшее расхождение (порядок каналов, способ сжатия) молча ломало бы
    обученную политику.

    Принимает (72, 128) или (B, 72, 128) uint8, возвращает (B, 10, 36, 64).
    """
    return indices_to_tensor(semantic_to_indices(sem_uint8), device=device)


def features_to_tensor(
    features: np.ndarray, device: torch.device | str | None = None
) -> Tensor:
    """Вектор признаков среды -> (B, FEATURE_DIM) float32 на нужном устройстве."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


class ActorCritic(nn.Module):
    """Политика и ценность с общим свёрточным стволом.

    Общий ствол, а не две сети: обе головы нуждаются в одном и том же —
    «где препятствия относительно игрока», — и разделение признаков вдвое
    сокращает и вычисления, и число шагов до появления осмысленных фильтров.

    Вход: `sem` — one-hot (B, 10, 36, 64) float32, `feat` — (B, 8) float32.
    Выход: логиты (B, 2) и ценность (B,).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feature_dim: int = 8,
        hidden: int = 256,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_actions = int(num_actions)
        self.hidden = int(hidden)

        # Первый слой намеренно «тайловый»: ядро 6 / шаг 4 / паддинг 1 на входе
        # 36x64 даёт 9x16 — ровно сетку тайлов камеры.
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(self.num_classes, 24, kernel_size=6, stride=4, padding=1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        conv_out = self._conv_out_dim()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(conv_out + self.feature_dim, self.hidden)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(self.hidden, self.hidden)),
            nn.ReLU(inplace=True),
        )
        self.policy_head = layer_init(nn.Linear(self.hidden, self.num_actions), GAIN_POLICY)
        self.value_head = layer_init(nn.Linear(self.hidden, 1), GAIN_VALUE)
        self._init_action_prior()

    def _init_action_prior(self) -> None:
        """Сместить смещение головы политики так, чтобы старт был не равномерным.

        Веса головы почти нулевые (gain 0.01), поэтому распределение действий у
        свежей сети задаётся ровно смещением: ставим его так, чтобы
        `P(ACTION_HOLD) == HOLD_PRIOR` (см. комментарий к константе — это не
        косметика, а то, что определяет, какая доля собранного опыта вообще
        несёт информацию о действии).
        """
        if self.num_actions < 2 or ACTION_HOLD >= self.num_actions:
            return
        p = min(max(float(HOLD_PRIOR), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.policy_head.bias.zero_()
            self.policy_head.bias[ACTION_HOLD] = math.log(p / (1.0 - p))

    def _conv_out_dim(self) -> int:
        """Размер плоского выхода свёрток — считаем прогоном, а не формулой."""
        with torch.no_grad():
            probe = torch.zeros(1, self.num_classes, POLICY_SEM_H, POLICY_SEM_W)
            return int(self.conv(probe).shape[1])

    @property
    def device(self) -> torch.device:
        """Устройство, на котором лежат веса (нужно, чтобы не гонять входы вручную)."""
        return next(self.parameters()).device

    def forward(self, sem: Tensor, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Логиты действий (B, 2) и ценность (B,)."""
        if sem.dtype != torch.float32:
            sem = sem.float()
        h = self.conv(sem)
        h = self.trunk(torch.cat((h, feat), dim=1))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def distribution(self, sem: Tensor, feat: Tensor) -> torch.distributions.Categorical:
        """Распределение над действиями — общая точка для act/evaluate_actions."""
        logits, _ = self.forward(sem, feat)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self, sem: Tensor, feat: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Выбрать действие: `(action, logprob, value)`.

        `deterministic=True` — argmax, для демонстраций и честной оценки: в игре
        на 60 кадрах в секунду случайное действие раз в сто кадров стоит жизни.
        Во время сбора опыта нужна именно выборка — иначе PPO нечего улучшать.
        """
        logits, value = self.forward(sem, feat)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, sem: Tensor, feat: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Пересчитать `(logprob, entropy, value)` для уже сделанных действий.

        Это ядро PPO: отношение новой и старой logprob даёт surrogate-функцию,
        энтропия удерживает политику от преждевременного схлопывания.
        """
        logits, value = self.forward(sem, feat)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions.long()), dist.entropy(), value

    @torch.no_grad()
    def act_numpy(
        self, sem_uint8: np.ndarray, features: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Удобный мост из numpy-наблюдений среды в действия (для pipeline/viz).

        Зачем: снаружи сети живут `uint8`-карты полного разрешения, и каждый
        потребитель иначе повторял бы цепочку «сжать -> one-hot -> тензор».
        """
        device = self.device
        sem = semantic_to_tensor(sem_uint8, device=device)
        feat = features_to_tensor(features, device=device)
        action, logprob, value = self.act(sem, feat, deterministic=deterministic)
        return (
            action.cpu().numpy(),
            logprob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def action_probs(self, sem: Tensor, feat: Tensor) -> Tensor:
        """Вероятности действий (B, 2) — для панели «уверенность ИИ» в вьюере."""
        logits, _ = self.forward(sem, feat)
        return F.softmax(logits, dim=-1)

    def num_parameters(self) -> int:
        """Число обучаемых параметров — попадает в логи и чекпойнт."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict[str, Any]:
        """Гиперпараметры архитектуры для чекпойнта: без них веса не поднять."""
        return {
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "num_actions": self.num_actions,
            "sem_shape": (POLICY_SEM_H, POLICY_SEM_W),
            "tiles": (VIEW_TILES_H, VIEW_TILES_W),
        }


__all__ = [
    "ActorCritic",
    "semantic_to_tensor",
    "semantic_to_indices",
    "indices_to_tensor",
    "features_to_tensor",
    "layer_init",
    "POLICY_DOWNSAMPLE",
    "POLICY_SEM_H",
    "POLICY_SEM_W",
    "GAIN_HIDDEN",
    "GAIN_POLICY",
    "GAIN_VALUE",
    "HOLD_PRIOR",
]
