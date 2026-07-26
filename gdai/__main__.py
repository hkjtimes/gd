"""Точка входа `python -m gdai` (SPEC §15).

Зачем отдельный файл, а не код в `cli.py`: модуль `gdai.cli` обязан
импортироваться как обычная библиотека (его парсером пользуются тесты и
Makefile), а `python -m gdai` должен выполнять команду и возвращать её код
процессу. Разделение убирает вечную проблему `__main__` — двойной импорт
одного и того же модуля под двумя именами.
"""

from __future__ import annotations

from gdai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
