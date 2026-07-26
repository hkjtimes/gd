"""Служебные инструменты: seed, логи, чекпойнты.

Зачем пакет: это единственные «глобальные» вещи в проекте (случайность,
вывод, файлы весов), и держать их вместе проще, чем искать по модулям.
"""

from __future__ import annotations

from gdai.utils.checkpoint import load_checkpoint, save_checkpoint
from gdai.utils.logging import JsonlLogger, get_logger, iter_jsonl, read_jsonl
from gdai.utils.seeding import make_rng, seed_from, set_global_seed

__all__ = [
    "get_logger",
    "JsonlLogger",
    "read_jsonl",
    "iter_jsonl",
    "set_global_seed",
    "make_rng",
    "seed_from",
    "save_checkpoint",
    "load_checkpoint",
]
