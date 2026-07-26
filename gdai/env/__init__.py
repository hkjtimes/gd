"""Симулятор: уровень, физика, каноническая карта, среда, рендер.

Зачем ленивый импорт: `gdai.env.level` и `gdai.env.physics` не зависят ни от
чего тяжёлого, а `render`/`themes` тянут pygame, `gd_env` — весь стек среды.
Импортировать их при `import gdai.env` значит платить секундами за каждый
запуск теста физики, поэтому тяжёлые имена достаются через `__getattr__`
(PEP 562) — то есть только в момент реального обращения.
"""

from __future__ import annotations

from typing import Any

from gdai.env.level import Level, LevelObject, OBJECT_TYPES
from gdai.env.physics import PlayerState, make_initial_state, step_physics

# Имя -> модуль, в котором оно живёт. Модули из более поздних фаз проекта
# могут ещё не существовать; ошибка появится только при обращении к имени.
_LAZY: dict[str, str] = {
    "generate_level": "gdai.env.generator",
    "is_solvable": "gdai.env.generator",
    "make_checkpoints": "gdai.env.generator",
    "render_semantic": "gdai.env.semantic",
    "camera_origin": "gdai.env.semantic",
    "semantic_to_rgb": "gdai.env.semantic",
    "downsample_semantic": "gdai.env.semantic",
    "world_to_pixel": "gdai.env.semantic",
    "GeometryDashEnv": "gdai.env.gd_env",
    "Theme": "gdai.env.themes",
    "BUILTIN_THEMES": "gdai.env.themes",
    "random_theme": "gdai.env.themes",
    "theme_by_name": "gdai.env.themes",
    "Renderer": "gdai.env.render",
}


def __getattr__(name: str) -> Any:
    """Достать «тяжёлое» имя из его модуля по первому обращению."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'gdai.env' has no attribute {name!r}")
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
    "Level",
    "LevelObject",
    "OBJECT_TYPES",
    "PlayerState",
    "step_physics",
    "make_initial_state",
    "generate_level",
    "is_solvable",
    "make_checkpoints",
    "render_semantic",
    "camera_origin",
    "semantic_to_rgb",
    "downsample_semantic",
    "world_to_pixel",
    "GeometryDashEnv",
    "Theme",
    "BUILTIN_THEMES",
    "random_theme",
    "theme_by_name",
    "Renderer",
]
