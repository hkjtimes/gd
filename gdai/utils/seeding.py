"""Управление случайностью.

Зачем: обучение RL воспроизводимо только тогда, когда КАЖДЫЙ источник шума
привязан к seed. Здесь единственная точка, где вообще разрешено трогать
глобальные генераторы; во всём остальном коде случайность передаётся явно
как `np.random.Generator`.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

_SEED_MODULUS = 2 ** 31 - 1


def set_global_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Засеять все глобальные ГПСЧ процесса (random, numpy, torch, PYTHONHASHSEED).

    Зачем: нужен один вызов в начале скрипта, чтобы прогон повторился бит-в-бит.
    torch импортируется лениво — модуль должен работать и без него (например,
    в чистой симуляции физики). Возвращает применённый seed для логов.
    """
    seed = int(seed) % _SEED_MODULUS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # мягкий импорт: физике и генератору уровней torch не нужен
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - на CPU-машине не выполняется
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Детерминизм важнее пары процентов скорости: без него нельзя
        # сравнивать два прогона обучения между собой.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Создать локальный генератор.

    Зачем именно `Generator`, а не глобальный `np.random`: среды и датасеты
    живут параллельно, и общий глобальный поток случайности сделал бы каждый
    из них зависимым от порядка вызовов соседей.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) % _SEED_MODULUS)


def seed_from(*parts: object) -> int:
    """Стабильный seed из произвольных частей (имя уровня, номер среды, эпоха).

    Зачем не встроенный `hash()`: он рандомизируется между запусками процесса,
    поэтому «тот же самый» эксперимент дал бы другие числа. blake2b даёт
    одинаковый результат всегда.
    """
    payload = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


__all__ = ["set_global_seed", "make_rng", "seed_from"]
