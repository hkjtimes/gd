"""Зрение GDAI: перевод ЛЮБОГО оформления кадра в каноническую карту.

Пакет состоит из четырёх частей:

* `model.py`   — маленький U-Net (< 500k параметров, GroupNorm);
* `dataset.py` — бесконечный поток пар «кадр со случайным дизайном -> карта»,
  с честным разбиением тем на обучающие и отложенные;
* `augment.py` — аугментации КАДРА (разметку они не трогают никогда);
* `train.py`   — CrossEntropy с весами классов + Dice, AdamW, косинусный lr.

Зачем ленивый импорт: `import gdai.perception` не должен тянуть torch и pygame.
Физику, генерацию уровней и тесты среды нужно уметь запускать в самом лёгком
окружении, поэтому имена достаются из своих модулей только при обращении
(PEP 562) — ровно тем же приёмом, что и в `gdai.env`.
"""

from __future__ import annotations

from typing import Any

_LAZY: dict[str, str] = {
    "PerceptionNet": "gdai.perception.model",
    "build_perception_net": "gdai.perception.model",
    "load_perception_net": "gdai.perception.model",
    "resolve_device": "gdai.perception.model",
    "SyntheticSegDataset": "gdai.perception.dataset",
    "make_loaders": "gdai.perception.dataset",
    "DatasetConfig": "gdai.perception.dataset",
    "frame_to_tensor": "gdai.perception.dataset",
    "frames_to_tensor": "gdai.perception.dataset",
    "train_themes": "gdai.perception.dataset",
    "held_out_themes": "gdai.perception.dataset",
    "HELD_OUT_THEME_NAMES": "gdai.perception.dataset",
    "TRAIN_THEME_NAMES": "gdai.perception.dataset",
    "AugmentConfig": "gdai.perception.augment",
    "augment_frame": "gdai.perception.augment",
    "augment_batch": "gdai.perception.augment",
    "train_perception": "gdai.perception.train",
    "evaluate_perception": "gdai.perception.train",
    "SegmentationLoss": "gdai.perception.train",
}


def __getattr__(name: str) -> Any:
    """Достать имя из его модуля по первому обращению (см. docstring пакета)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'gdai.perception' has no attribute {name!r}")
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
    "PerceptionNet",
    "build_perception_net",
    "load_perception_net",
    "resolve_device",
    "SyntheticSegDataset",
    "make_loaders",
    "DatasetConfig",
    "frame_to_tensor",
    "frames_to_tensor",
    "train_themes",
    "held_out_themes",
    "HELD_OUT_THEME_NAMES",
    "TRAIN_THEME_NAMES",
    "AugmentConfig",
    "augment_frame",
    "augment_batch",
    "train_perception",
    "evaluate_perception",
    "SegmentationLoss",
]
