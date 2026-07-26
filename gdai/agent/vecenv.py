"""Синхронный векторный враппер над несколькими копиями среды.

Зачем вектор, а не одна среда
-----------------------------
PPO — on-policy алгоритм: он учится только на свежем опыте, и качество шага
оптимизации определяется тем, насколько разнообразен собранный кусок. Одна
среда дала бы 256 подряд идущих кадров одного эпизода — почти дубликаты друг
друга, и градиент был бы шумной проекцией одной-единственной траектории. Восемь
сред на разных уровнях и в разных местах уровня дают тот же объём данных, но
слабо коррелированный.

Почему синхронный, а не процессы
--------------------------------
Шаг этой среды — чистая физика и растеризация numpy, доли миллисекунды.
Межпроцессная передача наблюдения (72x128 карта) стоила бы дороже самого шага,
поэтому параллелизм по процессам здесь проигрывает простому циклу. Вся тяжёлая
арифметика и так уходит в torch, который сам использует несколько ядер.

Авто-reset
----------
Среды не имеют права простаивать: как только эпизод закончился, среда сразу
начинает новый, а батч наблюдений получает уже НОВЫЙ первый кадр. Финальный
кадр завершённого эпизода не выбрасывается, а кладётся в
`info["final_observation"]` — он нужен PPO, чтобы дооценить будущее эпизода,
оборванного по лимиту шагов (`truncated`).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from gdai.config import EnvConfig
from gdai.env.gd_env import GeometryDashEnv
from gdai.utils.logging import get_logger
from gdai.utils.seeding import seed_from

_log = get_logger("agent.vecenv")

EnvFactory = Callable[[], GeometryDashEnv]


def make_env_fns(
    env_cfg: EnvConfig, num_envs: int, seed: int | None = None
) -> list[EnvFactory]:
    """Собрать фабрики сред с РАЗНЫМИ seed'ами при общем конфиге.

    Зачем отдельная функция: одинаковый seed во всех средах — самая дорогая
    ошибка в векторном обучении. Восемь сред сгенерировали бы одни и те же
    уровни и умирали бы в одном и том же месте, то есть восьмикратно
    продублировали бы один поток опыта, ничего не добавив.
    """
    from dataclasses import replace

    fns: list[EnvFactory] = []
    for i in range(int(num_envs)):
        cfg_i = replace(env_cfg, seed=seed_from("vecenv", seed, i))

        def _factory(cfg: EnvConfig = cfg_i) -> GeometryDashEnv:
            return GeometryDashEnv(cfg)

        fns.append(_factory)
    return fns


class SyncVectorEnv:
    """Список сред за единым интерфейсом `reset`/`step` с батчем наблюдений.

    Наблюдение — словарь массивов с ведущей осью среды:
    `{"semantic": (N, 72, 128) uint8, "features": (N, 8) float32}`.
    """

    def __init__(
        self, env_fns: Sequence[EnvFactory], auto_reset: bool = True
    ) -> None:
        if not env_fns:
            raise ValueError("Нужна хотя бы одна фабрика среды")
        self.envs: list[GeometryDashEnv] = [fn() for fn in env_fns]
        self.num_envs: int = len(self.envs)
        self.auto_reset: bool = bool(auto_reset)
        self.action_space_n: int = self.envs[0].action_space_n
        self._closed = False
        self._obs_keys: tuple[str, ...] = tuple(self.envs[0].observation_shapes)
        self._last_obs: dict[str, np.ndarray] | None = None

    # -- служебное ----------------------------------------------------------
    def _stack(self, obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Склеить наблюдения сред в батч по каждому ключу."""
        return {
            key: np.stack([obs[key] for obs in obs_list], axis=0)
            for key in self._obs_keys
        }

    @property
    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        """Формы наблюдения ОДНОЙ среды — сети нужны они, а не размер батча."""
        return self.envs[0].observation_shapes

    @property
    def difficulty(self) -> float:
        """Текущая сложность (у всех сред она одна — её ведёт учебный план)."""
        return self.envs[0].difficulty

    # -- жизненный цикл -----------------------------------------------------
    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        """Сбросить все среды. `seed` разводится по средам через `seed_from`."""
        if self._closed:
            raise RuntimeError("Векторная среда закрыта — reset невозможен")
        obs_list: list[dict[str, np.ndarray]] = []
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed_from("vecenv.reset", seed, i)
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            infos.append(info)
        self._last_obs = self._stack(obs_list)
        return self._last_obs, infos

    def step(
        self, actions: np.ndarray | Sequence[int]
    ) -> tuple[
        dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
    ]:
        """Шаг всех сред: `(obs, rewards, terminated, truncated, infos)`.

        При авто-reset в `obs` попадает первый кадр НОВОГО эпизода, а последний
        кадр старого — в `info["final_observation"]` вместе с `info["final_info"]`
        (там лежат `finished`, `progress`, `episode_return` — из них считается
        статистика обучения).
        """
        if self._closed:
            raise RuntimeError("Векторная среда закрыта — step невозможен")
        acts = np.asarray(actions).reshape(-1)
        if acts.shape[0] != self.num_envs:
            raise ValueError(
                f"Ожидалось {self.num_envs} действий, получено {acts.shape[0]}"
            )

        obs_list: list[dict[str, np.ndarray]] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(int(acts[i]))
            rewards[i] = reward
            terminated[i] = term
            truncated[i] = trunc
            if (term or trunc) and self.auto_reset:
                final_info = dict(info)
                info = dict(info)
                info["final_observation"] = obs
                info["final_info"] = final_info
                obs, _reset_info = env.reset()
            obs_list.append(obs)
            infos.append(info)

        self._last_obs = self._stack(obs_list)
        return self._last_obs, rewards, terminated, truncated, infos

    # -- управление --------------------------------------------------------
    def set_difficulty(self, difficulty: float) -> None:
        """Передать новую сложность всем средам (вызывает учебный план)."""
        for env in self.envs:
            env.set_difficulty(float(difficulty))

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Вызвать одноимённый метод у всех сред и собрать результаты.

        Зачем: обёртка не обязана знать про каждую редкую операцию среды
        (`render`, `close`, будущие расширения), а дублировать по методу на
        каждую — быстрый способ рассинхронизировать интерфейсы.
        """
        results: list[Any] = []
        for env in self.envs:
            fn = getattr(env, method, None)
            if not callable(fn):
                raise AttributeError(f"У среды нет метода {method!r}")
            results.append(fn(*args, **kwargs))
        return results

    def close(self) -> None:
        """Закрыть все среды; повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        for env in self.envs:
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - зависит от рендерера
                _log.debug("среда не закрылась штатно: %s", exc)

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self) -> "SyncVectorEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["SyncVectorEnv", "make_env_fns", "EnvFactory"]
