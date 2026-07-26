"""Полный агент: пиксели -> зрение -> политика -> действие (SPEC §12).

Зачем этот модуль отдельно от `agent` и `perception`
----------------------------------------------------
Зрение и политика обучаются раздельно и ничего друг о друге не знают: U-Net
переводит ЛЮБОЙ дизайн кадра в каноническую карту, PPO играет только по этой
карте. Но играть должен кто-то один, и именно здесь две половины склеиваются
в одно существо, которое умеет ровно две вещи: `see` (кадр -> карта) и `act`
(наблюдение -> кнопка).

Два режима одного агента
------------------------
* **честный** (`use_perception=True`) — карта берётся из U-Net по пикселям.
  Так агент играет в настоящую игру и так меряется реальное качество связки:
  ошибка зрения здесь стоит жизни, как и должно быть.
* **быстрый** (`use_perception=False`) — берётся эталонная карта из среды.
  Нужен, чтобы отделить ошибки политики от ошибок зрения: если агент валится
  и на эталоне, виновата политика, а не сегментация.

Работа без обученных весов
--------------------------
Оба пути обязаны подниматься и с пустыми путями к чекпойнтам: сети создаются
со случайной инициализацией. Это не «заглушка ради тестов», а требование
удобства — связку env -> карта -> сеть нужно уметь проверить (и показать
человеку в `python -m gdai watch`) до того, как хоть что-то обучено.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gdai.agent.networks import ActorCritic, features_to_tensor, semantic_to_tensor
from gdai.constants import ACTION_HOLD, NUM_ACTIONS, OBS_H, OBS_W
from gdai.env.gd_env import FEATURE_DIM
from gdai.perception.model import PerceptionNet, load_perception_net, resolve_device
from gdai.utils.logging import get_logger

_log = get_logger("pipeline")

# Метка источника весов, когда чекпойнта нет. Она попадает в HUD визуализатора
# и в отчёт `gdai eval`, чтобы никто не принял случайную сеть за обученную.
RANDOM_WEIGHTS: str = "случайная"


class GDAgent:
    """Полный агент: пиксели -> зрение -> политика -> действие.

    Работает и с ground-truth картой (быстро), и с предсказанной (честно).

    Атрибуты `policy` и `perception` намеренно публичные: визуализатор
    (`gdai.viz.viewer`) берёт из них вероятности действий и карту внимания,
    а `gdai.realgame.play` — только `see`/`act`. Ни один потребитель не обязан
    знать, откуда взялись веса.
    """

    def __init__(
        self,
        policy_path: str | None = None,
        perception_path: str | None = None,
        device: str = "auto",
        use_perception: bool = True,
    ) -> None:
        """Собрать агента; отсутствующие чекпойнты заменяются случайными сетями.

        `use_perception=False` полностью выключает зрение: сеть даже не
        создаётся, и агент умеет играть только по эталонной карте среды —
        так дешевле гонять оценку политики на тысячах эпизодов.
        """
        self.device: torch.device = resolve_device(device)

        self.policy: ActorCritic = self._build_policy(policy_path)
        self.policy_source: str = (
            Path(policy_path).name if policy_path else RANDOM_WEIGHTS
        )

        self.perception: PerceptionNet | None = self._build_perception(
            perception_path, use_perception
        )
        self.perception_source: str = (
            "нет"
            if self.perception is None
            else (Path(perception_path).name if perception_path else RANDOM_WEIGHTS)
        )
        # Флаг «по умолчанию смотреть глазами, а не эталоном». Отдельно от
        # наличия сети: сеть может быть загружена, а конкретный вызов `act`
        # всё равно попросит эталон (см. параметр `use_perception` там).
        self.use_perception: bool = self.perception is not None

        # Телеметрия последнего решения — её показывает вьюер и печатает CLI.
        self.last_semantic: np.ndarray | None = None
        self.last_action: int = 0
        self.last_p_hold: float = 0.0
        self.last_value: float = 0.0
        self.frames_seen: int = 0

        _log.info(
            "агент собран: политика «%s», зрение «%s», устройство %s",
            self.policy_source, self.perception_source, self.device,
        )

    # -- сборка -------------------------------------------------------------
    def _build_policy(self, path: str | None) -> ActorCritic:
        """Политика из чекпойнта или случайная сеть той же архитектуры."""
        if path:
            from gdai.agent.ppo import load_policy

            model = load_policy(path, device=str(self.device))
        else:
            model = ActorCritic(feature_dim=FEATURE_DIM).to(self.device)
        model.eval()
        return model

    def _build_perception(
        self, path: str | None, use_perception: bool
    ) -> PerceptionNet | None:
        """Зрение из чекпойнта, случайное — или ничего, если оно не нужно."""
        if not use_perception:
            return None
        if path:
            model = load_perception_net(path, device=str(self.device))
        else:
            model = PerceptionNet().to(self.device)
        model.eval()
        return model

    # -- зрение -------------------------------------------------------------
    def see(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Кадр (H, W, 3) uint8 -> каноническая карта классов (H, W) uint8.

        Это и есть «что видит нейросеть»: дальше по конвейеру идёт только
        карта, поэтому любые декорации кадра здесь и заканчиваются.
        """
        if self.perception is None:
            raise RuntimeError(
                "Зрение выключено (use_perception=False) — предсказывать карту "
                "нечем. Создайте агента с use_perception=True или подайте "
                "эталонную карту в act()."
            )
        arr = np.asarray(frame_rgb)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Ожидался кадр (H, W, 3), получено {arr.shape}")
        sem = np.asarray(self.perception.predict(arr), dtype=np.uint8)
        self.last_semantic = sem
        return sem

    def semantic_from_obs(
        self, obs: dict[str, Any], use_perception: bool | None = None
    ) -> np.ndarray:
        """Достать карту из наблюдения: предсказать по пикселям или взять эталон.

        Вынесено в публичный метод, потому что визуализации и отладке нужна
        именно карта, а не действие, и повторять эту развилку у себя каждый
        потребитель не должен.
        """
        want_vision = self.use_perception if use_perception is None else bool(use_perception)

        if want_vision:
            already = obs.get("semantic")
            if already is not None and already is self.last_semantic:
                # Карту этого же кадра только что посчитал сам вызывающий
                # (так делает цикл `gdai.realgame.play`: сначала `see` для
                # показа, потом `act`). Второй прогон U-Net дал бы тот же
                # результат вдвое дороже — на 60 кадрах в секунду это половина
                # бюджета времени.
                return self.last_semantic
            frame = obs.get("pixels")
            if frame is None:
                raise ValueError(
                    "Зрение включено, но в наблюдении нет ключа 'pixels'. "
                    "Задайте среде obs_mode='both' (или 'pixels'), либо "
                    "вызовите act(..., use_perception=False) для эталона."
                )
            return self.see(frame)

        sem = obs.get("semantic")
        if sem is None:
            # Эталона нет — это не повод падать, если есть чем смотреть.
            frame = obs.get("pixels")
            if frame is not None and self.perception is not None:
                return self.see(frame)
            raise ValueError(
                "В наблюдении нет ни 'semantic', ни пары «'pixels' + зрение» — "
                "политике не на что смотреть. Проверьте obs_mode среды."
            )
        arr = np.asarray(sem, dtype=np.uint8)
        self.last_semantic = arr
        return arr

    # -- решение ------------------------------------------------------------
    def decide(
        self,
        obs: dict[str, Any],
        deterministic: bool = True,
        use_perception: bool | None = None,
    ) -> tuple[int, float, float]:
        """Решение целиком: `(действие, P(держать), ценность)`.

        Вероятность и ценность нужны не сети, а человеку и логам: по ним сразу
        видно, уверен ли агент перед шипом и считает ли положение выигрышным.
        """
        sem = self.semantic_from_obs(obs, use_perception=use_perception)
        features = self._features(obs)

        with torch.no_grad():
            sem_t = semantic_to_tensor(sem, device=self.device)
            feat_t = features_to_tensor(features, device=self.device)
            logits, value = self.policy(sem_t, feat_t)
            probs = torch.softmax(logits, dim=-1)[0]
            if deterministic:
                action = int(torch.argmax(logits, dim=-1)[0])
            else:
                action = int(torch.multinomial(probs, num_samples=1)[0])
            p_hold = float(probs[ACTION_HOLD]) if NUM_ACTIONS > 1 else 1.0
            v = float(value.reshape(-1)[0])

        self.last_action = action
        self.last_p_hold = p_hold
        self.last_value = v
        self.frames_seen += 1
        return action, p_hold, v

    def act(
        self,
        obs: dict[str, Any],
        deterministic: bool = True,
        use_perception: bool | None = None,
    ) -> int:
        """Одно число — то, что уходит в среду: 0 (ничего) или 1 (держать).

        `deterministic=True` по умолчанию: в игре на 60 кадрах в секунду один
        случайный прыжок раз в сотню кадров стоит жизни, поэтому выборка
        нужна только во время обучения.
        """
        return self.decide(
            obs, deterministic=deterministic, use_perception=use_perception
        )[0]

    def reset(self) -> None:
        """Забыть телеметрию прошлого эпизода (сам агент состояния не имеет).

        Зачем метод вообще есть: потребители (`play_real`, вьюер, `evaluate`)
        обязаны уметь сказать «начался новый эпизод», не зная, есть ли у
        конкретного агента память. Появится рекуррентная политика — менять
        придётся только это тело.
        """
        self.last_semantic = None
        self.last_action = 0
        self.last_p_hold = 0.0
        self.last_value = 0.0
        self.frames_seen = 0

    # -- сервис -------------------------------------------------------------
    @staticmethod
    def _features(obs: dict[str, Any]) -> np.ndarray:
        """Вектор признаков из наблюдения; отсутствие — нули нужной длины.

        Нули — честный «ничего не знаю»: на живом экране вертикальной скорости
        и флага «на земле» взять неоткуда (см. `gdai.realgame.play`), и падать
        из-за этого агент не должен.
        """
        features = obs.get("features")
        if features is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
        if arr.size != FEATURE_DIM:
            raise ValueError(
                f"Ожидался вектор признаков длины {FEATURE_DIM}, получено {arr.size}"
            )
        return arr

    def describe(self) -> dict[str, Any]:
        """Короткая сводка «кто это и на чём считает» — для CLI и логов."""
        return {
            "policy": self.policy_source,
            "perception": self.perception_source,
            "device": str(self.device),
            "policy_params": int(self.policy.num_parameters()),
            "perception_params": (
                int(self.perception.count_parameters()) if self.perception else 0
            ),
            "use_perception": bool(self.use_perception),
            "obs_size": (OBS_H, OBS_W),
        }


def evaluate(
    agent: GDAgent,
    env: Any,
    episodes: int = 20,
    use_perception: bool = False,
    *,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Прогнать агента по эпизодам и вернуть метрики качества.

    Ключи (SPEC §12): `success_rate`, `mean_progress`, `mean_reward`,
    `mean_len`; остальное — подробности для отчёта CLI.

    Каждый эпизод стартует с `start_x=0.0`, то есть С НАЧАЛА уровня. Это
    принципиально: среда умеет тренировочные рестарты с чекпойнта, и без явного
    старта с нуля «доля прохождений» превратилась бы в «долю добеганий с
    середины» — цифру, которой нельзя верить.
    """
    n = max(1, int(episodes))
    finished: list[bool] = []
    progress: list[float] = []
    rewards: list[float] = []
    lengths: list[int] = []
    deaths = 0
    timeouts = 0

    for episode in range(n):
        reset_seed = seed if (episode == 0 and seed is not None) else None
        obs, info = env.reset(seed=reset_seed, start_x=0.0)
        if use_perception and "pixels" not in obs:
            raise ValueError(
                "Оценка со зрением требует кадров: создайте среду с "
                "obs_mode='both' (сейчас в наблюдении нет ключа 'pixels')."
            )
        agent.reset()

        total = 0.0
        steps = 0
        while True:
            action = agent.act(
                obs, deterministic=deterministic, use_perception=use_perception
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break

        ok = bool(info.get("finished", False))
        finished.append(ok)
        progress.append(float(info.get("progress", 0.0)))
        rewards.append(total)
        lengths.append(steps)
        deaths += int(bool(info.get("died", False)))
        timeouts += int(not ok and not bool(info.get("died", False)))

    result: dict[str, Any] = {
        "success_rate": float(np.mean(finished)),
        "mean_progress": float(np.mean(progress)),
        "mean_reward": float(np.mean(rewards)),
        "mean_len": float(np.mean(lengths)),
        "max_progress": float(np.max(progress)),
        "min_progress": float(np.min(progress)),
        "episodes": n,
        "deaths": deaths,
        "timeouts": timeouts,
        "use_perception": bool(use_perception),
        "deterministic": bool(deterministic),
    }
    _log.info(
        "оценка на %d эпизодах: прохождений %.2f, прогресс %.2f, награда %.2f, "
        "длина %.0f (зрение %s)",
        n, result["success_rate"], result["mean_progress"], result["mean_reward"],
        result["mean_len"], "вкл" if use_perception else "выкл",
    )
    return result


__all__ = ["GDAgent", "evaluate", "RANDOM_WEIGHTS"]
