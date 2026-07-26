"""Политика: actor-critic на канонических картах и обучение PPO.

Зачем ленивый импорт: `gdai.agent.ppo` тянет torch и весь стек среды, а
`gdai.env`/`gdai.perception` должны импортироваться в окружениях, где ничего
этого не нужно. Имена достаются через `__getattr__` (PEP 562) — то есть только
в момент реального обращения.
"""

from __future__ import annotations

from typing import Any

# Имя -> модуль, в котором оно живёт.
_LAZY: dict[str, str] = {
    "ActorCritic": "gdai.agent.networks",
    "semantic_to_tensor": "gdai.agent.networks",
    "semantic_to_indices": "gdai.agent.networks",
    "indices_to_tensor": "gdai.agent.networks",
    "features_to_tensor": "gdai.agent.networks",
    "layer_init": "gdai.agent.networks",
    "RolloutBuffer": "gdai.agent.buffer",
    "MiniBatch": "gdai.agent.buffer",
    "SyncVectorEnv": "gdai.agent.vecenv",
    "make_env_fns": "gdai.agent.vecenv",
    "Curriculum": "gdai.agent.curriculum",
    "train_agent": "gdai.agent.ppo",
    "load_policy": "gdai.agent.ppo",
    "resolve_device": "gdai.agent.ppo",
}


def __getattr__(name: str) -> Any:
    """Достать имя из его модуля по первому обращению."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'gdai.agent' has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Имя {name!r} живёт в модуле {module_path}, но его не удалось "
            f"импортировать: {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ActorCritic",
    "semantic_to_tensor",
    "semantic_to_indices",
    "indices_to_tensor",
    "features_to_tensor",
    "layer_init",
    "RolloutBuffer",
    "MiniBatch",
    "SyncVectorEnv",
    "make_env_fns",
    "Curriculum",
    "train_agent",
    "load_policy",
    "resolve_device",
]
