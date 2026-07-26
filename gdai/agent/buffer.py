"""Буфер роллаута: хранение опыта, GAE(λ) и нарезка минибатчей.

Зачем отдельный модуль
----------------------
PPO — алгоритм «собрали кусок опыта, несколько раз по нему прошлись». Вся
сложность этого куска не в самом хранении, а в двух местах, где легко получить
молча неверное обучение:

1. **Границы эпизодов.** Преимущество нельзя протаскивать через смерть игрока:
   награда следующего эпизода не имеет отношения к действиям предыдущего.
2. **Обрыв по лимиту шагов (`truncated`).** Эпизод, оборванный счётчиком, НЕ
   закончился — его будущее ещё стоит денег. Если приравнять такой обрыв к
   смерти, критик выучит, что «долго жить плохо». Поэтому для оборванных
   эпизодов буфер принимает `bootstrap_value` — ценность последнего кадра.

Что хранится
------------
Карта лежит в компактном виде — индексы классов uint8 (T, N, 36, 64), а не
one-hot. Роллаут 256x8 в one-hot float32 занял бы ~190 МБ и упёрся бы в память
раньше, чем в вычисления; в индексах это 4.7 МБ, а разворот в one-hot делается
поминибатчево.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from gdai.utils.logging import get_logger

_log = get_logger("agent.buffer")


@dataclass
class MiniBatch:
    """Один минибатч для шага оптимизации PPO.

    `sem` — индексы классов (B, 36, 64) uint8: разворот в one-hot остаётся за
    сетью, чтобы буфер не зависел от формата входа модели.
    """

    sem: Tensor
    features: Tensor
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class RolloutBuffer:
    """Кольцо фиксированной длины на `num_steps` шагов по `num_envs` средам.

    Работает по циклу: `reset()` -> `num_steps` раз `add()` ->
    `compute_returns_and_advantages()` -> несколько эпох `minibatches()`.
    Данные хранятся на CPU: они собираются из numpy и всё равно уезжают на
    устройство минибатчами.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        sem_shape: tuple[int, int],
        feature_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError(
                f"num_steps и num_envs должны быть положительными, получено "
                f"{num_steps} и {num_envs}"
            )
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.sem_shape = (int(sem_shape[0]), int(sem_shape[1]))
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)

        t, n = self.num_steps, self.num_envs
        self.sem = torch.zeros((t, n, *self.sem_shape), dtype=torch.uint8)
        self.features = torch.zeros((t, n, self.feature_dim), dtype=torch.float32)
        self.actions = torch.zeros((t, n), dtype=torch.int64)
        self.log_probs = torch.zeros((t, n), dtype=torch.float32)
        self.values = torch.zeros((t, n), dtype=torch.float32)
        self.rewards = torch.zeros((t, n), dtype=torch.float32)
        self.dones = torch.zeros((t, n), dtype=torch.float32)
        # Ценность последнего кадра оборванного эпизода (для truncated); для
        # «настоящих» терминалов остаётся нулём — там будущего действительно нет.
        self.bootstraps = torch.zeros((t, n), dtype=torch.float32)
        self.advantages = torch.zeros((t, n), dtype=torch.float32)
        self.returns = torch.zeros((t, n), dtype=torch.float32)

        self._pos = 0
        self._ready = False

    # -- наполнение ---------------------------------------------------------
    @property
    def size(self) -> int:
        """Сколько шагов уже записано."""
        return self._pos

    @property
    def full(self) -> bool:
        """Буфер набран полностью — пора считать преимущества."""
        return self._pos >= self.num_steps

    @property
    def batch_size(self) -> int:
        """Сколько переходов даёт полный роллаут (`num_steps * num_envs`)."""
        return self.num_steps * self.num_envs

    def reset(self) -> None:
        """Начать новый роллаут (данные не обнуляем — их перезапишут)."""
        self._pos = 0
        self._ready = False

    def add(
        self,
        sem: np.ndarray | Tensor,
        features: np.ndarray | Tensor,
        actions: np.ndarray | Tensor,
        log_probs: np.ndarray | Tensor,
        values: np.ndarray | Tensor,
        rewards: np.ndarray | Tensor,
        dones: np.ndarray | Tensor,
        bootstrap_values: np.ndarray | Tensor | None = None,
    ) -> None:
        """Записать один шаг всех сред.

        `dones` — эпизод завершён любым способом (смерть, финиш, лимит):
        именно здесь рвётся цепочка GAE. `bootstrap_values` заполняется только
        для оборванных по лимиту эпизодов — см. docstring модуля.
        """
        if self._pos >= self.num_steps:
            raise RuntimeError(
                f"Буфер уже полон ({self.num_steps} шагов) — нужен reset()"
            )
        i = self._pos
        self.sem[i] = torch.as_tensor(np.asarray(sem, dtype=np.uint8))
        self.features[i] = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.actions[i] = torch.as_tensor(np.asarray(actions)).long()
        self.log_probs[i] = torch.as_tensor(np.asarray(log_probs, dtype=np.float32))
        self.values[i] = torch.as_tensor(np.asarray(values, dtype=np.float32))
        self.rewards[i] = torch.as_tensor(np.asarray(rewards, dtype=np.float32))
        self.dones[i] = torch.as_tensor(np.asarray(dones, dtype=np.float32))
        if bootstrap_values is None:
            self.bootstraps[i] = 0.0
        else:
            self.bootstraps[i] = torch.as_tensor(
                np.asarray(bootstrap_values, dtype=np.float32)
            )
        self._pos += 1

    # -- GAE ----------------------------------------------------------------
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray | Tensor,
        last_dones: np.ndarray | Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Посчитать GAE(λ) и цели для критика по всему роллауту.

        Зачем GAE, а не «просто дисконтированная сумма»: на 60 кадрах в секунду
        одна ошибка стоит эпизода, и честная сумма наград имеет огромную
        дисперсию. λ<1 подмешивает оценку критика и меняет часть дисперсии на
        небольшое смещение — без этого PPO здесь не сходится.

        `last_values` — ценность состояния, которое идёт СРАЗУ ЗА последним
        шагом роллаута: роллаут почти всегда обрывается посреди эпизода, и без
        этого хвоста последние шаги остались бы без будущего.

        **Соглашение об индексах** (тут легко получить молча неверное обучение):
        `dones[t]` означает «эпизод завершился ПОСЛЕ действия на шаге t», то
        есть ровно то, что вернул `step()` вместе с `rewards[t]`. Значит маска
        обрыва для шага t — это `1 - dones[t]`, а НЕ `1 - dones[t+1]`
        (последнее — соглашение, где `dones[t]` относится к состоянию на входе
        шага). Перепутанные соглашения дают сдвиг на один кадр: шаг смерти
        дооценивается ценностью первого кадра СЛЕДУЮЩЕГО эпизода, и агент
        выучивает, что умирать выгодно — рестарт возвращает его к началу, где
        снова много награды за прогресс.

        `last_dones` в этом соглашении избыточен (он совпадает с `dones[-1]`) и
        принимается только ради совместимости вызова; расхождение — признак
        того, что вызывающий перепутал соглашение, и о нём сообщается в лог.
        """
        if self._pos != self.num_steps:
            raise RuntimeError(
                f"Роллаут заполнен на {self._pos}/{self.num_steps} шагов — "
                "GAE считать рано"
            )
        next_value = torch.as_tensor(
            np.asarray(last_values, dtype=np.float32)
        ).reshape(self.num_envs)
        tail_dones = torch.as_tensor(
            np.asarray(last_dones, dtype=np.float32)
        ).reshape(self.num_envs)
        if not bool(torch.equal(tail_dones, self.dones[self.num_steps - 1])):
            _log.warning(
                "last_dones не совпадает с dones[-1]: похоже, вызывающий "
                "использует другое соглашение об индексах (см. docstring). "
                "Беру dones[-1] как единственный источник правды."
            )

        gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for t in range(self.num_steps - 1, -1, -1):
            # Маска берётся по ТЕКУЩЕМУ шагу: после терминала будущего нет.
            non_terminal = 1.0 - self.dones[t]
            # Оценка будущего за шагом t: либо ценность следующего кадра
            # роллаута, либо (для обрыва по лимиту) сохранённая ценность
            # финального кадра, либо ноль после настоящего терминала.
            future = non_terminal * next_value + self.bootstraps[t]
            delta = self.rewards[t] + gamma * future - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae
            next_value = self.values[t]
        self.returns = self.advantages + self.values
        self._ready = True

    # -- выдача минибатчей --------------------------------------------------
    def _flat(self) -> tuple[Tensor, ...]:
        """Схлопнуть оси (шаг, среда) в одну — минибатчи режутся по переходам."""
        b = self.batch_size
        return (
            self.sem.reshape(b, *self.sem_shape),
            self.features.reshape(b, self.feature_dim),
            self.actions.reshape(b),
            self.log_probs.reshape(b),
            self.values.reshape(b),
            self.advantages.reshape(b),
            self.returns.reshape(b),
        )

    def minibatches(
        self,
        num_minibatches: int,
        rng: np.random.Generator | None = None,
        normalize_advantages: bool = True,
    ) -> Iterator[MiniBatch]:
        """Итератор случайных минибатчей одной эпохи.

        Нормализация преимуществ делается **внутри минибатча**, а не по всему
        роллауту: масштаб преимуществ меняется от итерации к итерации (в начале
        обучения он крошечный, после первых прохождений — большой), и без
        приведения к нулевому среднему и единичной дисперсии один и тот же
        learning rate то не двигает политику, то рвёт её.

        `rng` передаётся явно — перемешивание тоже обязано воспроизводиться
        по seed.
        """
        if not self._ready:
            raise RuntimeError(
                "Сначала вызовите compute_returns_and_advantages() — "
                "преимущества ещё не посчитаны"
            )
        k = int(num_minibatches)
        if k <= 0:
            raise ValueError(f"num_minibatches должно быть >= 1, получено {k}")
        total = self.batch_size
        if k > total:
            raise ValueError(
                f"Минибатчей ({k}) больше, чем переходов в роллауте ({total})"
            )

        order = (
            rng.permutation(total)
            if rng is not None
            else np.random.permutation(total)
        )
        indices = torch.as_tensor(np.ascontiguousarray(order), dtype=torch.long)
        sem, feat, act, logp, val, adv, ret = self._flat()
        # Хвост от неровного деления доклеивается к последнему минибатчу:
        # выбрасывать переходы нельзя, они уже оплачены средой.
        size = total // k
        for i in range(k):
            start = i * size
            stop = total if i == k - 1 else start + size
            idx = indices[start:stop]
            a = adv[idx]
            if normalize_advantages and a.numel() > 1:
                a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
            yield MiniBatch(
                sem=sem[idx].to(self.device),
                features=feat[idx].to(self.device),
                actions=act[idx].to(self.device),
                log_probs=logp[idx].to(self.device),
                values=val[idx].to(self.device),
                advantages=a.to(self.device),
                returns=ret[idx].to(self.device),
            )

    def explained_variance(self) -> float:
        """Насколько критик объясняет дисперсию доходности (1 — идеально, 0 — бесполезен).

        Зачем в буфере: это первая метрика, по которой видно, что обучение
        сломано (застрявшая около нуля — критик не учится, и все преимущества
        шум).
        """
        if not self._ready:
            raise RuntimeError("Сначала вызовите compute_returns_and_advantages()")
        y = self.returns.reshape(-1)
        pred = self.values.reshape(-1)
        var = torch.var(y, unbiased=False)
        if float(var) <= 1e-12:
            return 0.0
        return float(1.0 - torch.var(y - pred, unbiased=False) / var)


__all__ = ["RolloutBuffer", "MiniBatch"]
