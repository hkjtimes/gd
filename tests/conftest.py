"""Общая обвязка тестов: headless-графика, детерминизм и типовые фикстуры.

Зачем этот файл
---------------
Три вещи обязаны произойти ДО того, как любой тест что-либо импортирует:

1. **SDL в режиме dummy.** `gdai.env.render` открывает поверхность pygame прямо
   на импорте модуля. На машине без дисплея (CI, контейнер) это падает, если
   переменная окружения выставлена позже, — SDL читает её один раз, в момент
   инициализации.
2. **Путь к пакету.** Тесты должны запускаться в неустановленном репозитории
   (`pytest -q` из корня), поэтому корень проекта добавляется в `sys.path`
   явно, а не «повезёт с rootdir».
3. **Ограничение потоков torch.** На CI ядер мало, а тестов, дергающих BLAS,
   много; без ограничения потоки дерутся между собой и прогон становится в
   разы медленнее, чем однопоточный.

Фикстуры здесь — только «дешёвые» объекты (уровни, состояния, генератор
случайности). Всё, что стоит секунды (уровни от процедурного генератора,
обученные сети), создаётся в самих тестах — так видно цену каждого теста.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- 1. headless ДО любого импорта pygame -----------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Логи библиотеки в stderr во время тестов только мешают читать отчёт.
os.environ.setdefault("GDAI_LOG_LEVEL", "ERROR")

# --- 2. корень репозитория в sys.path ---------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gdai.env.level import Level, LevelObject  # noqa: E402
from gdai.env.physics import PlayerState, make_initial_state  # noqa: E402

# --- 3. потоки torch ---------------------------------------------------------
try:  # torch может отсутствовать в самом лёгком окружении — тесты физики живут и без него
    import torch

    torch.set_num_threads(min(2, os.cpu_count() or 1))
except ImportError:  # pragma: no cover - в проекте torch есть
    torch = None  # type: ignore[assignment]


# Базовый seed тестов. Все генераторы выводятся из него, чтобы падение можно
# было воспроизвести одной строкой, а не «пересобрать всю случайность».
TEST_SEED: int = 20240726


def pytest_configure(config: pytest.Config) -> None:
    """Зарегистрировать маркер `slow`, чтобы CI мог его отфильтровать.

    Регистрация обязательна: с `--strict-markers` (см. pytest.ini) незнакомый
    маркер — ошибка, и это правильно: опечатка в имени маркера иначе тихо
    отключала бы фильтрацию.
    """
    config.addinivalue_line(
        "markers", "slow: тяжёлый тест (в CI отключается через -m 'not slow')"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Локальный генератор с фиксированным seed — воспроизводимость прогона."""
    return np.random.default_rng(TEST_SEED)


@pytest.fixture
def flat_level() -> Level:
    """Ровная дорожка без препятствий: эталон «ничего не мешает».

    Зачем: на нём проверяется чистая физика (высота прыжка, детерминизм) без
    единого постороннего взаимодействия.
    """
    return Level(
        name="flat",
        length=200.0,
        objects=[LevelObject("goal", 200.0, 6.0)],
        ceiling_y=12.0,
    )


@pytest.fixture
def demo_level() -> Level:
    """Небольшой уровень со всеми ключевыми классами объектов.

    Зачем именно такой набор: карта, рендер и приоритеты классов проверяются
    только тогда, когда в кадре одновременно есть блок, шип, пила, кольцо, пад,
    портал и финиш — иначе половина веток растеризатора не выполняется.
    """
    objects = [
        LevelObject("block", 8.0, 0.5),
        LevelObject("platform", 9.0, 2.75),
        LevelObject("spike", 12.0, 0.5),
        LevelObject("spike_down", 12.0, 4.0),
        LevelObject("saw", 15.0, 1.0),
        LevelObject("orb_yellow", 13.5, 2.0),
        LevelObject("pad_pink", 16.0, 0.25),
        LevelObject("portal_ship", 18.0, 1.25),
        LevelObject("portal_gravity_up", 20.0, 1.25),
        LevelObject("portal_speed_2", 22.0, 1.25),
        LevelObject("goal", 30.0, 6.0),
    ]
    return Level(name="demo", length=30.0, objects=objects, ceiling_y=12.0)


@pytest.fixture
def demo_state(demo_level: Level) -> PlayerState:
    """Игрок в воздухе посреди `demo_level` — типовой кадр для карты и рендера."""
    return PlayerState(
        x=11.0,
        y=1.9,
        vy=-4.0,
        mode="cube",
        gravity=1,
        speed_index=1,
        on_ground=False,
        alive=True,
        finished=False,
        hold_prev=False,
    )


@pytest.fixture
def start_state(flat_level: Level) -> PlayerState:
    """Стартовое состояние на ровной дорожке."""
    return make_initial_state(flat_level)


@pytest.fixture
def level_file(tmp_path: Path, demo_level: Level) -> Path:
    """Файл уровня на диске — для сред, которым нужен фиксированный уровень.

    Зачем фиксированный уровень в тестах среды и PPO: процедурная генерация с
    проверкой проходимости стоит около секунды на уровень, и smoke-тест
    обучения превратился бы в тест генератора.
    """
    path = tmp_path / "demo.json"
    demo_level.save(path)
    return path
