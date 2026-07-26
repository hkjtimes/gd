"""Визуализация: окно «что видит ИИ», графики обучения и карты внимания.

Три модуля отвечают на три разных вопроса:

* `viewer.py`   — «что нейросеть видит прямо сейчас»: кадр с декорациями,
  предсказанная каноническая карта и эталон с подсветкой ошибок;
* `plots.py`    — «как шло обучение»: сетка графиков из `metrics.jsonl`;
* `saliency.py` — «на что агент смотрит»: градиент логита «держать» по карте.

Зачем ленивый импорт: `viewer` тянет pygame, `plots` — matplotlib, а
`saliency` — torch. Команде `python -m gdai plot` незачем поднимать pygame,
а тестам физики — вообще ничего из этого, поэтому имена достаются из своих
модулей только при обращении (PEP 562), как и в остальных пакетах проекта.
"""

from __future__ import annotations

from typing import Any

_LAZY: dict[str, str] = {
    "Viewer": "gdai.viz.viewer",
    "ViewerConfig": "gdai.viz.viewer",
    "run_viewer": "gdai.viz.viewer",
    "record_demo": "gdai.viz.viewer",
    "assemble_animation": "gdai.viz.viewer",
    "HOTKEYS": "gdai.viz.viewer",
    "plot_run": "gdai.viz.plots",
    "plot_metrics": "gdai.viz.plots",
    "load_metrics": "gdai.viz.plots",
    "available_metrics": "gdai.viz.plots",
    "summarize": "gdai.viz.plots",
    "PanelSpec": "gdai.viz.plots",
    "saliency_map": "gdai.viz.saliency",
    "class_saliency": "gdai.viz.saliency",
    "overlay_saliency": "gdai.viz.saliency",
    "saliency_rgb": "gdai.viz.saliency",
    "save_saliency_png": "gdai.viz.saliency",
    "saliency_from_obs": "gdai.viz.saliency",
}


def __getattr__(name: str) -> Any:
    """Достать имя из его модуля по первому обращению (см. docstring пакета)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'gdai.viz' has no attribute {name!r}")
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
    "Viewer",
    "ViewerConfig",
    "run_viewer",
    "record_demo",
    "assemble_animation",
    "HOTKEYS",
    "plot_run",
    "plot_metrics",
    "load_metrics",
    "available_metrics",
    "summarize",
    "PanelSpec",
    "saliency_map",
    "class_saliency",
    "overlay_saliency",
    "saliency_rgb",
    "save_saliency_png",
    "saliency_from_obs",
]
