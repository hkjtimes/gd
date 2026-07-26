"""Игра в настоящую Geometry Dash — НЕОБЯЗАТЕЛЬНЫЙ сценарий (SPEC §14).

Предупреждение (оно же в docstring обоих модулей и в README): этот пакет не
является основным путём проекта. Он требует

* ручной калибровки прямоугольника игрового поля на экране,
* пакетов `mss` (захват) и `pynput` (нажатия), которых нет в зависимостях,
* и запущенного окна игры на переднем плане.

Смысл пакета — показать, что зрение, обученное на синтетике с доменной
рандомизацией, переносится на чужие пиксели: та же сеть превращает кадр
настоящей игры в ту же каноническую карту. Проверять и развивать архитектуру
следует на собственном симуляторе (`python -m gdai watch`), где есть эталонная
разметка и воспроизводимость.

Импорт ленивый: `import gdai.realgame` не должен требовать ни mss, ни pynput —
ошибка появляется только при реальной попытке захватить экран или нажать
клавишу, и объясняет, что именно поставить.
"""

from __future__ import annotations

from typing import Any

_LAZY: dict[str, str] = {
    "CaptureRegion": "gdai.realgame.capture",
    "ScreenCapture": "gdai.realgame.capture",
    "require_mss": "gdai.realgame.capture",
    "resize_frame": "gdai.realgame.capture",
    "bgra_to_rgb": "gdai.realgame.capture",
    "monitor_region": "gdai.realgame.capture",
    "region_from_values": "gdai.realgame.capture",
    "calibrate_region": "gdai.realgame.capture",
    "load_region": "gdai.realgame.capture",
    "DEFAULT_REGION_PATH": "gdai.realgame.capture",
    "RealGameConfig": "gdai.realgame.play",
    "KeyHolder": "gdai.realgame.play",
    "EmergencyStop": "gdai.realgame.play",
    "play_real": "gdai.realgame.play",
    "load_agent": "gdai.realgame.play",
    "estimate_features": "gdai.realgame.play",
    "require_pynput": "gdai.realgame.play",
}


def __getattr__(name: str) -> Any:
    """Достать имя из его модуля по первому обращению (см. docstring пакета)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'gdai.realgame' has no attribute {name!r}")
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
    "CaptureRegion",
    "ScreenCapture",
    "require_mss",
    "resize_frame",
    "bgra_to_rgb",
    "monitor_region",
    "region_from_values",
    "calibrate_region",
    "load_region",
    "DEFAULT_REGION_PATH",
    "RealGameConfig",
    "KeyHolder",
    "EmergencyStop",
    "play_real",
    "load_agent",
    "estimate_features",
    "require_pynput",
]
