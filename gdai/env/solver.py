"""Проверка проходимости уровня: поиск в ширину по кадрам физики.

Зачем этот модуль вообще существует
-----------------------------------
Процедурный генератор без проверки — это генератор красивых непроходимых
уровней. Агент, обучающийся на таких уровнях, получает шум вместо сигнала:
он умирает не потому, что ошибся, а потому, что пройти было нельзя.
Поэтому «проходимость» здесь не эвристика, а факт: если существует
последовательность действий из {ACTION_NONE, ACTION_HOLD}, приводящая игрока
к финишу, поиск её найдёт (в пределах выделенного бюджета).

Как это может быть быстро
-------------------------
Наивный перебор двух действий на кадр даёт 2^N веток. Спасают три вещи:

1. **Синхронность по кадрам.** Ищем волной: множество достижимых состояний
   после кадра t порождает множество после t+1. Это ровно BFS, но без
   очереди — фронт хранится списком.
2. **Дедупликация округлённых состояний.** Физически неразличимые состояния
   (разница в сотые доли тайла) склеиваются: ключ — округлённые
   x/y/vy плюс дискретные флаги. Фронт из-за этого не растёт экспоненциально,
   а держится десятками состояний: на земле все ветки схлопываются в одну,
   в воздухе их столько, сколько было моментов начала прыжка.
3. **Ограничение фронта.** Если состояний всё-таки слишком много (корабль:
   vy непрерывна), фронт прореживается с сохранением разнообразия.

Направление ошибки выбрано осознанно: округление и прореживание могут
«потерять» решение (ложное «непроходим»), но НИКОГДА не выдумать его —
каждое состояние во фронте реально достижимо, потому что получено настоящим
`step_physics`. Генератору это подходит: ложный отказ стоит перегенерации
участка, ложное «да» стоило бы непроходимого уровня.

`hold_prev` тоже входит в ключ, хотя SPEC его не перечисляет: кольца в GD
срабатывают по фронту нажатия, и без этого бита поиск терял бы единственный
способ пройти секцию с цепочкой колец.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from gdai.constants import ACTION_HOLD, ACTION_NONE, ORB, PORTAL_MODE
from gdai.env.level import Level
from gdai.env.physics import PlayerState, make_initial_state, step_physics

# --- кванты дедупликации (см. SPEC §6) --------------------------------------
X_QUANT: float = 0.25
Y_QUANT: float = 0.25
VY_QUANT: float = 0.5

_INV_X: float = 1.0 / X_QUANT
_INV_Y: float = 1.0 / Y_QUANT
_INV_VY: float = 1.0 / VY_QUANT

# --- бюджеты по умолчанию ---------------------------------------------------
# Кадров хватает на уровень в ~300 тайлов даже на самой медленной скорости.
DEFAULT_MAX_FRAMES: int = 20000
# Раскрытых узлов. При ~5 мкс на шаг физики это пара секунд в худшем случае, но
# запас важнее: слишком тесный бюджет превращается в ложное «непроходим» на
# длинном уровне, а это худшая из возможных ошибок — она молча выбрасывает
# нормальные уровни. Генератор задаёт свой, меньший бюджет явно.
DEFAULT_MAX_NODES: int = 500_000
# Ширина фронта. 128 с запасом покрывает «решётку моментов прыжка» куба
# (прыжок длится 30 кадров) и разумную выборку состояний корабля.
DEFAULT_MAX_FRONTIER: int = 128

# На сколько тайлов вокруг кольца важно помнить `hold_prev`.
ORB_ZONE_MARGIN: int = 3

StateKey = tuple[int, int, int, bool, str, int, int, bool]


def state_key(state: PlayerState, track_hold: bool = True) -> StateKey:
    """Ключ склейки состояний: округлённая физика + дискретные флаги.

    Зачем округление, а не точное сравнение: два состояния, отличающиеся на
    сотую тайла, ведут к одинаковым исходам, но как разные узлы взрывают
    перебор. Кванты (0.25 / 0.25 / 0.5) — из SPEC §6.

    `track_hold` включает в ключ `hold_prev`. Он нужен только рядом с
    кольцами (они срабатывают по фронту нажатия), а стоит ровно вдвое больше
    состояний, поэтому вдали от колец бит намеренно обнуляется: уже через
    кадр обе ветки нажатия рождаются заново, так что склейка ничего не теряет.
    """
    return (
        int(state.x * _INV_X),
        int(state.y * _INV_Y),
        int(state.vy * _INV_VY),
        state.on_ground,
        state.mode,
        state.gravity,
        state.speed_index,
        state.hold_prev and track_hold,
    )


def orb_zone(level: Level, margin: int = ORB_ZONE_MARGIN) -> frozenset[int]:
    """Целые x-клетки, рядом с которыми есть кольцо (см. `state_key`)."""
    return _zone(level, (ORB,), margin)


def _zone(level: Level, classes: tuple[int, ...], margin: int) -> frozenset[int]:
    """Целые x-клетки в окрестности объектов перечисленных классов."""
    cells: set[int] = set()
    for obj in level.objects:
        if obj.semantic_class() in classes:
            base = int(obj.x)
            cells.update(range(base - margin, base + margin + 1))
    return frozenset(cells)


@dataclass
class SearchResult:
    """Итог одного прогона поиска.

    Зачем не просто bool: генератор строит уровень участками и продолжает
    поиск с сохранённого фронта, поэтому ему нужны сами состояния, а не
    только вердикт.
    """

    reached: list[PlayerState] = field(default_factory=list)
    finished: bool = False
    nodes: int = 0
    frames: int = 0
    dead_end: bool = False      # фронт опустел — дальше пройти нельзя
    budget_exceeded: bool = False


_BOTH: tuple[bool, ...] = (False, True)
_ONLY_RELEASE: tuple[bool, ...] = (False,)


def _useful_holds(state: PlayerState, sensitive: frozenset[int]) -> tuple[bool, ...]:
    """Какие действия имеет смысл пробовать из этого состояния.

    Зачем: куб в воздухе не управляем вообще — прыжок разрешён только с земли,
    пад срабатывает сам. Значит обе ветки дают физически ОДНО И ТО ЖЕ
    состояние, отличаясь лишь битом `hold_prev`, который вдали от колец в
    ключе не участвует. Проверять обе — удваивать работу впустую, а именно на
    воздушные фазы приходится большая часть фронта.

    Исключения (`sensitive`) — окрестности колец (ловятся нажатием) и порталов
    режима: пройдя портал корабля или волны, игрок В ТОТ ЖЕ КАДР считается уже
    по новой физике, где удержание — это тяга.

    Остаётся один теоретический зазор: «подшагивание» на блок при касании в
    0.02 тайла может сделать куб опорным прямо посреди кадра. Ветка прыжка
    при этом теряется ровно на один кадр (на следующем она снова доступна),
    так что худшее последствие — лишний отказ участка, но не ложное «проходим».
    """
    if state.mode == "cube" and not state.on_ground and int(state.x) not in sensitive:
        return _ONLY_RELEASE
    return _BOTH


def _thin_out(states: list[PlayerState], limit: int) -> list[PlayerState]:
    """Проредить фронт до `limit` состояний, сохраняя разнообразие.

    Зачем равномерный шаг, а не «первые N»: состояния лежат в порядке
    порождения, то есть соседние почти одинаковы. Взяв первые N, мы оставили
    бы один узкий пучок траекторий; равномерная выборка сохраняет и «низкие»,
    и «высокие» ветки.
    """
    n = len(states)
    if n <= limit:
        return states
    stride = n / float(limit)
    return [states[int(i * stride)] for i in range(limit)]


def search_forward(
    level: Level,
    frontier: Sequence[PlayerState],
    target_x: float | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_frontier: int = DEFAULT_MAX_FRONTIER,
) -> SearchResult:
    """Продвинуть множество состояний вперёд волной по кадрам.

    `target_x is None` — искать финиш (режим `is_solvable`).
    `target_x` задан — довести все ветки до `x >= target_x` и вернуть их;
    так генератор проверяет очередной участок, не пересчитывая весь префикс
    заново (иначе построение уровня было бы квадратичным).

    Вход не мутируется: `step_physics` чистая, состояния только порождаются.
    """
    active: list[PlayerState] = [s for s in frontier if s.alive and not s.finished]
    reached: dict[StateKey, PlayerState] = {}
    result = SearchResult()
    if not active:
        result.dead_end = True
        return result

    # Склеиваем состояния ТОЛЬКО внутри одного кадра. Это принципиально:
    # поиск фронтальный, и все ветки на кадре t находятся в одной и той же
    # фазе игры, поэтому совпадение ключей означает физическую эквивалентность.
    # Общая таблица посещённых здесь была бы ошибкой: за кадр игрок проезжает
    # 0.139-0.32 тайла, а x квантуется по 0.25, так что бегущий по земле игрок
    # регулярно попадает в тот же ключ, что и кадром раньше, — и «уже видели»
    # вырезало бы единственную живую траекторию.
    nodes = 0
    step = step_physics
    key_of = state_key
    orbs = orb_zone(level)
    sensitive = orbs | _zone(level, (PORTAL_MODE,), ORB_ZONE_MARGIN)

    for frame in range(max_frames):
        result.frames = frame + 1
        nxt: dict[StateKey, PlayerState] = {}
        for st in active:
            for hold in _useful_holds(st, sensitive):
                nst = step(st, level, hold)[0]
                nodes += 1
                if nst.finished:
                    result.finished = True
                    result.nodes = nodes
                    result.reached = list(reached.values())
                    return result
                if not nst.alive:
                    continue
                k = key_of(nst, int(nst.x) in orbs)
                if target_x is not None and nst.x >= target_x:
                    if k not in reached:
                        reached[k] = nst
                elif k not in nxt:
                    nxt[k] = nst

        result.nodes = nodes
        if nodes >= max_nodes:
            result.budget_exceeded = True
            break
        active = list(nxt.values())
        if len(active) > max_frontier:
            active = _thin_out(active, max_frontier)
        if not active:
            break

    result.reached = _thin_out(list(reached.values()), max_frontier)
    result.dead_end = not result.reached and not result.finished
    return result


def is_solvable(
    level: Level,
    max_frames: int = DEFAULT_MAX_FRAMES,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_frontier: int = DEFAULT_MAX_FRONTIER,
    start: PlayerState | None = None,
) -> bool:
    """Существует ли последовательность действий, доводящая игрока до финиша.

    Зачем именно поиск, а не «прогнать простого бота»: любой бот — это
    предположение о том, как надо играть, и уровень, который бот не осилил,
    вполне может быть проходимым. Здесь же ответ честный в пределах бюджета:
    «нет» означает «в пределах max_frames/max_nodes решение не найдено».
    """
    start_state = start if start is not None else make_initial_state(level, 0.0)
    res = search_forward(
        level,
        [start_state],
        None,
        max_frames=max_frames,
        max_nodes=max_nodes,
        max_frontier=max_frontier,
    )
    return res.finished


def solve_actions(
    level: Level,
    max_frames: int = DEFAULT_MAX_FRAMES,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_frontier: int = DEFAULT_MAX_FRONTIER,
    start: PlayerState | None = None,
) -> list[int] | None:
    """Найти сам путь: список действий 0/1 по кадрам до финиша (или None).

    Зачем: (а) тесты могут проиграть найденный путь через `step_physics` и
    убедиться, что уровень действительно проходится, (б) это готовая
    «демонстрация эксперта» для отладки среды и визуализатора.

    Хранит дерево предков, поэтому дороже по памяти, чем `is_solvable`, — в
    горячем пути генерации не используется.
    """
    start_state = start if start is not None else make_initial_state(level, 0.0)
    if not start_state.alive:
        return None

    # Плоское дерево: parents[i] = (индекс родителя, действие).
    parents: list[tuple[int, int]] = [(-1, ACTION_NONE)]
    states: list[PlayerState] = [start_state]
    active: list[int] = [0]
    nodes = 0
    orbs = orb_zone(level)
    sensitive = orbs | _zone(level, (PORTAL_MODE,), ORB_ZONE_MARGIN)

    for _ in range(max_frames):
        nxt: dict[StateKey, int] = {}
        for idx in active:
            st = states[idx]
            for hold in _useful_holds(st, sensitive):
                action = ACTION_HOLD if hold else ACTION_NONE
                nst = step_physics(st, level, hold)[0]
                nodes += 1
                states.append(nst)
                parents.append((idx, action))
                child = len(states) - 1
                if nst.finished:
                    return _unwind(parents, child)
                if not nst.alive:
                    states.pop()
                    parents.pop()
                    continue
                k = state_key(nst, int(nst.x) in orbs)
                if k in nxt:
                    states.pop()
                    parents.pop()
                    continue
                nxt[k] = child
        if nodes >= max_nodes or not nxt:
            return None
        active = list(nxt.values())
        if len(active) > max_frontier:
            stride = len(active) / float(max_frontier)
            active = [active[int(i * stride)] for i in range(max_frontier)]
    return None


def _unwind(parents: list[tuple[int, int]], node: int) -> list[int]:
    """Собрать последовательность действий от корня к найденному узлу."""
    actions: list[int] = []
    while node > 0:
        parent, action = parents[node]
        actions.append(action)
        node = parent
    actions.reverse()
    return actions


__all__ = [
    "X_QUANT",
    "Y_QUANT",
    "VY_QUANT",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_FRONTIER",
    "ORB_ZONE_MARGIN",
    "StateKey",
    "SearchResult",
    "state_key",
    "orb_zone",
    "search_forward",
    "is_solvable",
    "solve_actions",
]
