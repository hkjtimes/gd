"""Логирование: единственный разрешённый способ что-либо сообщить наружу.

Зачем: `print` в библиотеке ломает и CLI (мусор в stdout, который может быть
перенаправлен в файл), и обучение (миллионы строк). Здесь два инструмента:
человекочитаемый `get_logger` (в stderr) и машиночитаемый `JsonlLogger`
(metrics.jsonl), из которого потом строятся графики.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

_LOGGER_NAME = "gdai"
_configured = False


def _configure_root() -> None:
    """Один раз навесить обработчик на корневой логгер пакета.

    Зачем отдельная функция: logging глобален, и повторная настройка приводит
    к дублированию каждой строки. Уровень берётся из GDAI_LOG_LEVEL, чтобы
    можно было включить DEBUG без правки кода.
    """
    global _configured
    if _configured:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    level_name = os.environ.get("GDAI_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    # Не отдаём записи наверх: иначе чужой basicConfig напечатает их второй раз.
    logger.propagate = False
    _configured = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Логгер для модуля; имя без префикса автоматически попадает в 'gdai.<name>'."""
    _configure_root()
    if name == _LOGGER_NAME or name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def _json_default(value: Any) -> Any:
    """Привести numpy/torch-скаляры к обычным числам.

    Зачем: метрики почти всегда приходят как np.float32/torch.Tensor, а
    json.dump на них падает — терять из-за этого весь прогон недопустимо.
    """
    for attr in ("item", "tolist"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover - экзотические объекты
                pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class JsonlLogger:
    """Построчный JSON-лог метрик (`metrics.jsonl`).

    Зачем формат JSONL: его можно дописывать во время обучения и читать
    одновременно другим процессом (графики строятся на живом прогоне), а при
    падении процесса всё уже записанное остаётся валидным.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        filename: str = "metrics.jsonl",
        append: bool = True,
    ) -> None:
        out = Path(out_dir)
        # Разрешаем передать как каталог прогона, так и сразу путь к файлу.
        if out.suffix == ".jsonl":
            self._path = out
        else:
            self._path = out / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a" if append else "w", encoding="utf-8")
        self._t0 = time.time()

    @property
    def path(self) -> Path:
        """Путь к metrics.jsonl — его же читают plots.py и тесты."""
        return self._path

    def log(self, record: dict[str, Any]) -> None:
        """Записать одну строку метрик; поля 'time' и 'wall' добавляются сами.

        Зачем flush на каждой записи: обучение идёт часами, и мы хотим видеть
        кривые в реальном времени, а не после завершения.
        """
        if self._file.closed:
            raise ValueError("JsonlLogger уже закрыт — запись невозможна")
        payload = dict(record)
        payload.setdefault("time", round(time.time() - self._t0, 3))
        payload.setdefault("wall", time.time())
        self._file.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        """Закрыть файл; повторный вызов безопасен."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Прочитать metrics.jsonl, молча пропуская битую последнюю строку.

    Зачем терпимость к мусору: файл могли читать в момент записи или прогон
    убили посреди строки — графики из-за этого падать не должны.
    """
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Ленивое чтение metrics.jsonl — для очень длинных прогонов."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


__all__ = ["get_logger", "JsonlLogger", "read_jsonl", "iter_jsonl"]
