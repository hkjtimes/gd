"""Сохранение и загрузка чекпойнтов.

Зачем отдельный модуль: обучение зрения и политики падает/прерывается, и
единственная защита — атомарная запись. Если писать прямо в best.pt и умереть
на середине, потеряются и новые, и старые веса. Здесь запись всегда идёт во
временный файл рядом, а затем os.replace (атомарен в пределах ФС).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_TORCH_HINT = (
    "Для работы с чекпойнтами нужен torch. Установите его: pip install torch"
)


def _require_torch():
    """Ленивый импорт torch с понятным сообщением.

    Зачем лениво: физика, генератор уровней и семантическая карта обязаны
    импортироваться в окружении вообще без torch (например, в лёгком CI).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - в проекте torch есть
        raise ImportError(_TORCH_HINT) from exc
    return torch


def _normalize_config(config: Any) -> Any:
    """Превратить dataclass-конфиг в обычный dict.

    Зачем: чекпойнт должен читаться и без импорта классов проекта, иначе
    переименование поля сделает старые файлы нечитаемыми.
    """
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    return config


def save_checkpoint(
    path: str | os.PathLike[str],
    state_dict: dict[str, Any],
    config: Any = None,
    meta: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Атомарно сохранить чекпойнт `{"state_dict", "model_state", "config", "meta"}`.

    Зачем два ключа под веса: SPEC описывает файлы зрения как
    `{"model_state": ...}`, а универсальный контракт — как `state_dict`.
    Оба ключа указывают на один и тот же объект, поэтому файл читается любым
    из потребителей без конвертации.

    `extra` попадает в корень словаря как есть (например, состояние оптимизатора
    или curriculum), чтобы обучение можно было продолжить с того же места.
    """
    torch = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    full_meta: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_ts": time.time(),
        "torch_version": torch.__version__,
    }
    try:
        from gdai import __version__ as gdai_version

        full_meta["gdai_version"] = gdai_version
    except Exception:  # pragma: no cover - защита от циклов импорта
        pass
    if meta:
        full_meta.update(meta)

    payload: dict[str, Any] = {
        "state_dict": state_dict,
        "model_state": state_dict,
        "config": _normalize_config(config),
        "meta": full_meta,
    }
    if extra:
        payload.update(extra)

    # Временный файл кладём рядом с целью: os.replace атомарен только внутри
    # одной файловой системы, /tmp может быть на другой.
    tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Загрузить чекпойнт и привести его к единому виду.

    Зачем нормализация: файлы могли быть записаны как `{"model_state": ...}`
    или вовсе быть «голым» state_dict — потребителю не должно быть до этого
    дела, он всегда получает словарь с ключами state_dict/model_state/config/meta.
    """
    torch = _require_torch()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Чекпойнт не найден: {target}")

    # weights_only=False: внутри лежат не только тензоры, но и конфиг/мета.
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - очень старые версии torch
        payload = torch.load(target, map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError(f"Неожиданный формат чекпойнта {target}: {type(payload)!r}")

    state = payload.get("state_dict", payload.get("model_state"))
    if state is None:
        # «Голый» state_dict модели — оборачиваем, чтобы контракт не ломался.
        state = payload
        payload = {}

    result: dict[str, Any] = dict(payload)
    result["state_dict"] = state
    result["model_state"] = state
    result.setdefault("config", None)
    result.setdefault("meta", {})
    return result


__all__ = ["save_checkpoint", "load_checkpoint"]
