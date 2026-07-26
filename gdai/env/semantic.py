"""Каноническая семантическая карта — ground truth всего проекта.

Зачем этот модуль существует
----------------------------
Здесь рождается тот самый «общий язык», ради которого затевалась вся
архитектура: любой уровень с любым оформлением превращается в матрицу из
десяти классов. На этих матрицах учится политика, и ими же размечается
датасет для зрения. Отсюда два жёстких требования:

1. **Точность.** Карта — это разметка. Если растеризатор ошибётся на пиксель,
   U-Net будет учиться на систематически кривой цели, а политика — считать
   шип на полтайла левее, чем он есть.
2. **Скорость.** Карта строится на каждом кадре в каждой из восьми сред и на
   каждом сэмпле датасета зрения. Поэтому здесь нет pygame и нет попиксельных
   циклов по всему кадру: всё сводится к горизонтальным отрезкам (span-ам),
   которые numpy заполняет одним присваиванием среза. Цель — > 2000 карт/с
   на одном ядре CPU.

Камера — общая с «красивым» рендером
------------------------------------
`camera_origin` — единственный источник правды о том, какой кусок мира виден.
`render.py` обязан звать ту же функцию, иначе пиксель кадра перестанет
соответствовать пикселю разметки и обучение зрения развалится. Поэтому
функция ЧИСТАЯ (зависит только от `state`) и ДЕТЕРМИНИРОВАННАЯ — никакой
истории, никакого сглаживания по времени, никакой случайности.

Система координат
-----------------
Мир: тайлы, x вправо, y ВВЕРХ, пол на `GROUND_Y`. Карта: строка 0 — верх кадра
(ось Y перевёрнута), столбец 0 — левый край. Пиксель считается закрашенным,
если его ЦЕНТР попал внутрь фигуры — это даёт симметричную растеризацию и
исключает «расползание» объектов на полпикселя вправо-вниз.

Размеры фигур берутся из `LevelObject.half_extent()`, то есть карта показывает
ровно хитбокс. Так политика видит именно то, что её убивает, а не декоративный
размер спрайта (в Geometry Dash шип нарисован заметно больше своей хитбокс-зоны).
"""

from __future__ import annotations

import math

import numpy as np

from gdai.constants import (
    CLASS_COLORS,
    GROUND_Y,
    NUM_CLASSES,
    OBS_H,
    OBS_W,
    PLAYER,
    PLAYER_X_IN_VIEW,
    PX_PER_TILE,
    SOLID,
    VIEW_TILES_H,
)
from gdai.env.level import (
    Level,
    LevelObject,
    OBJECT_TYPES,
    ORB_TYPES,
    PAD_TYPES,
    PORTAL_TYPES,
    type_half_extent,
    type_semantic_class,
)
from gdai.env.physics import PlayerState, player_half_extent

# --- параметры камеры -------------------------------------------------------
# Доля высоты кадра (снизу), на которой камера держит игрока в «свободном»
# режиме. 0.6 подобрано так, чтобы обычный прыжок с земли (2.4 тайла) целиком
# помещался в кадр и НЕ сдвигал камеру: пол остаётся неподвижным ориентиром.
CAM_Y_ANCHOR: float = 0.6
# Сколько тайлов мира ниже пола остаётся в кадре. Ненулевой отступ нужен, чтобы
# полоса пола была видна как полоса (у неё есть толщина), а не как край экрана.
CAM_GROUND_MARGIN: float = 1.0
# Ширина зоны плавного перехода между «камера прижата к полу» и «камера следит
# за игроком», в тайлах. Обычный max() дал бы излом: в момент, когда игрок
# пересекает порог, камера мгновенно меняет режим и разметка начинает дёргаться
# на пиксель туда-обратно. Мягкий максимум делает зависимость гладкой (C1).
CAM_SOFT_K: float = 0.5

# --- параметры фигур --------------------------------------------------------
# Доля внешнего радиуса, приходящаяся на дырку кольца. Кольцо обязано читаться
# именно как кольцо: это единственный объект, который требует НАЖАТИЯ, и путать
# его с пилой (сплошной круг) политике нельзя.
RING_INNER_RATIO: float = 0.5

# --- приоритет классов ------------------------------------------------------
# SPEC §7: EMPTY < SOLID < PORTAL_* < PAD < ORB < GOAL < HAZARD < PLAYER.
# Порталы разведены по соседним рангам, чтобы таблица была БИЕКТИВНОЙ: тогда
# тот же порядок годится для `downsample_semantic` (сжатие обязано уметь
# однозначно восстановить класс по рангу).
_PRIORITY: tuple[int, ...] = (
    0,   # EMPTY
    1,   # SOLID
    8,   # HAZARD
    9,   # PLAYER
    5,   # PAD
    6,   # ORB
    4,   # PORTAL_GRAVITY
    3,   # PORTAL_MODE
    2,   # PORTAL_SPEED
    7,   # GOAL
)
_PRIORITY_LUT: np.ndarray = np.asarray(_PRIORITY, dtype=np.uint8)
_PRIORITY_INV_LUT: np.ndarray = np.zeros(NUM_CLASSES, dtype=np.uint8)
for _cls, _rank in enumerate(_PRIORITY):
    _PRIORITY_INV_LUT[_rank] = _cls

_CLASS_RGB_LUT: np.ndarray = np.asarray(CLASS_COLORS, dtype=np.uint8)

# --- виды фигур -------------------------------------------------------------
# Целые числа, а не строки: диспетчер фигур вызывается десятки раз на кадр,
# и сравнение int дешевле сравнения str.
_KIND_RECT: int = 0
_KIND_TRI_UP: int = 1
_KIND_TRI_DOWN: int = 2
_KIND_TRI_LEFT: int = 3
_KIND_TRI_RIGHT: int = 4
_KIND_CIRCLE: int = 5
_KIND_RING: int = 6
_KIND_ELLIPSE: int = 7

_KIND_NAMES: tuple[str, ...] = (
    "rect", "tri_up", "tri_down", "tri_left", "tri_right",
    "circle", "ring", "ellipse",
)

# Явное соответствие «тип объекта -> фигура». Треугольник шипа смотрит туда,
# куда указывает имя типа: `spike` — вверх, `spike_down` — вниз и т.д.
_SPIKE_KINDS: dict[str, int] = {
    "spike": _KIND_TRI_UP,
    "spike_down": _KIND_TRI_DOWN,
    "spike_left": _KIND_TRI_LEFT,
    "spike_right": _KIND_TRI_RIGHT,
}


def _kind_for_type(obj_type: str) -> int:
    """Какой фигурой рисуется объект данного типа."""
    if obj_type in _SPIKE_KINDS:
        return _SPIKE_KINDS[obj_type]
    if obj_type == "saw":
        return _KIND_CIRCLE
    if obj_type in ORB_TYPES:
        return _KIND_RING
    if obj_type in PORTAL_TYPES:
        return _KIND_ELLIPSE
    if obj_type in PAD_TYPES:
        return _KIND_RECT
    return _KIND_RECT


# Предрасчёт всего, что нужно для отрисовки одного объекта: ранг приоритета,
# фигура, полуразмеры, класс. Зачем: иначе на каждый объект каждого кадра
# пришлось бы звать методы и лазить по трём словарям.
_DRAW: dict[str, tuple[int, int, float, float, int]] = {}
for _t in OBJECT_TYPES:
    _c = type_semantic_class(_t)
    _hx, _hy = type_half_extent(_t)
    _DRAW[_t] = (_PRIORITY[_c], _kind_for_type(_t), _hx, _hy, _c)

# Публичная карта фигур: `render.py` рисует «красивую» версию тех же объектов и
# обязан согласовать форму со своим ground truth (треугольник — треугольником,
# кольцо — кольцом), иначе зрение будет учить противоречивую цель.
SHAPE_BY_TYPE: dict[str, str] = {t: _KIND_NAMES[_DRAW[t][1]] for t in OBJECT_TYPES}

_P: float = float(PX_PER_TILE)
_INV_P: float = 1.0 / float(PX_PER_TILE)


# ---------------------------------------------------------------------------
# камера
# ---------------------------------------------------------------------------
def _smooth_max(a: float, b: float, k: float) -> float:
    """Гладкий максимум: `max(a, b)`, но без излома в точке равенства.

    Зачем: жёсткий `max` даёт разрыв производной — камера, доехав до
    ограничителя, «щёлкает» между режимами, и разметка дёргается на пиксель
    туда-обратно при почти неподвижном игроке.

    Как: вне полосы шириной k результат в точности равен `max`, внутри полосы
    склейка идёт параболой `b + k*h^2`, где `h = 0.5 + 0.5*(a-b)/k`. Функция
    непрерывна вместе с производной, строго не убывает по `a` и превышает
    честный максимум не более чем на k/4 (при k -> 0 вырождается в `max`).
    """
    t = a - b
    if k <= 0.0 or t >= k:
        return a if t >= 0.0 else b
    if t <= -k:
        return b
    h = 0.5 + 0.5 * t / k
    return b + k * h * h


def camera_origin(state: PlayerState) -> tuple[float, float]:
    """Левый-нижний угол камеры в мировых координатах (тайлы).

    Правило (детерминированное, чистая функция от `state` — это критично,
    потому что рендерер декораций обязан использовать ТУ ЖЕ камеру):

    * по X: `cam_x = state.x - PLAYER_X_IN_VIEW` — игрок всегда стоит на
      четвёртом тайле от левого края, впереди видно 12 тайлов будущего;
    * по Y: камера хочет держать игрока на высоте `CAM_Y_ANCHOR` доли кадра
      (`y - VIEW_TILES_H * CAM_Y_ANCHOR`), но не имеет права опуститься ниже
      `GROUND_Y - CAM_GROUND_MARGIN`. Пока игрок скачет у земли — камера
      прижата к полу и не двигается вовсе (обычный прыжок в 2.4 тайла не
      достаёт до порога), поэтому пол работает неподвижным ориентиром.
      Как только игрок поднимается примерно выше 4 тайлов (корабль, волна,
      высокие постройки) — камера плавно уезжает за ним. Стык двух режимов
      размазан по полосе `CAM_SOFT_K` через `_smooth_max`, чтобы на границе
      разметка не дёргалась на пиксель туда-обратно.

    Верхнего ограничителя нет намеренно: потолок уровня у каждого свой, а
    сигнатура (SPEC §7) даёт только состояние игрока. Потолок как объект мира
    всё равно попадает в карту классом SOLID, когда въезжает в кадр.
    """
    cam_x = float(state.x) - PLAYER_X_IN_VIEW
    follow_y = float(state.y) - VIEW_TILES_H * CAM_Y_ANCHOR
    floor_y = GROUND_Y - CAM_GROUND_MARGIN
    cam_y = _smooth_max(follow_y, floor_y, CAM_SOFT_K)
    return (cam_x, cam_y)


def world_to_pixel(
    wx: float,
    wy: float,
    cam: tuple[float, float],
    view_h: int = OBS_H,
) -> tuple[int, int]:
    """Мировая точка -> пиксель карты как `(столбец, строка)`.

    Зачем отдельная функция: этой же формулой пользуются рендер декораций,
    визуализатор и тесты; продублированная «на месте» арифметика рано или
    поздно разъедется на полпикселя. Значения НЕ обрезаются по кадру —
    вызывающий сам решает, что делать с точкой за экраном.
    """
    cam_x, cam_y = cam
    px = int(math.floor((wx - cam_x) * _P))
    py = int(view_h) - 1 - int(math.floor((wy - cam_y) * _P))
    return (px, py)


# ---------------------------------------------------------------------------
# растеризация: отрезки
# ---------------------------------------------------------------------------
def _col_span(x0: float, x1: float, cam_x: float, view_w: int) -> tuple[int, int]:
    """Полуинтервал столбцов [c0, c1), чьи центры лежат в мировом [x0, x1]."""
    c0 = int(math.ceil((x0 - cam_x) * _P - 0.5))
    c1 = int(math.floor((x1 - cam_x) * _P - 0.5)) + 1
    if c0 < 0:
        c0 = 0
    if c1 > view_w:
        c1 = view_w
    return c0, c1


def _row_span(y0: float, y1: float, cam_y: float, view_h: int) -> tuple[int, int]:
    """Полуинтервал строк [r0, r1) для мирового [y0, y1] (ось Y перевёрнута)."""
    top = float(view_h) - 0.5
    r0 = int(math.ceil(top - (y1 - cam_y) * _P))
    r1 = int(math.floor(top - (y0 - cam_y) * _P)) + 1
    if r0 < 0:
        r0 = 0
    if r1 > view_h:
        r1 = view_h
    return r0, r1


def _fill_bytes(view_w: int) -> tuple[bytes, ...]:
    """Заранее нарезанные строки «класс, повторённый view_w раз».

    Зачем: закраска отрезка идёт записью в память кадра напрямую
    (`memoryview`), а для этого нужен готовый источник байтов. Кэш по ширине
    кадра убирает любые аллокации из горячего цикла.
    """
    rows = _FILL_CACHE.get(view_w)
    if rows is None:
        rows = tuple(bytes((c,)) * view_w for c in range(NUM_CLASSES))
        _FILL_CACHE[view_w] = rows
    return rows


_FILL_CACHE: dict[int, tuple[bytes, ...]] = {}


def _draw_shape(
    sem: np.ndarray,
    buf: memoryview,
    fill: bytes,
    cam_x: float,
    cam_y: float,
    kind: int,
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    cls: int,
    view_w: int,
    view_h: int,
) -> None:
    """Закрасить одну фигуру классом `cls` (перекрывая всё, что было).

    Зачем span-ы, а не булевы маски: фигуры крошечные (шип — около 4x4 px),
    и создание временных numpy-массивов на каждый объект стоило бы в разы
    дороже самой заливки. Здесь на строку приходится ровно одна запись
    непрерывного куска памяти.

    Вся арифметика ведётся в «индексах пикселей»: центр столбца `c` имеет
    координату `c`, поэтому условие попадания центра в фигуру — это просто
    `|c - fx| <= полуширина_в_пикселях`. Такой перевод делается один раз на
    объект, а не на каждую строку.
    """
    # Центр фигуры в индексах столбцов/строк (ось Y перевёрнута) и её
    # полуразмеры в пикселях.
    fx = (cx - cam_x) * _P - 0.5
    fy = (view_h - 0.5) - (cy - cam_y) * _P
    rhx = hx * _P
    rhy = hy * _P

    r0 = int(math.ceil(fy - rhy))
    r1 = int(math.floor(fy + rhy)) + 1
    if r0 < 0:
        r0 = 0
    if r1 > view_h:
        r1 = view_h
    if r0 >= r1:
        return

    if kind == _KIND_RECT:
        c0 = int(math.ceil(fx - rhx))
        c1 = int(math.floor(fx + rhx)) + 1
        if c0 < 0:
            c0 = 0
        if c1 > view_w:
            c1 = view_w
        if c0 < c1:
            sem[r0:r1, c0:c1] = cls
        return

    inv_rhy = 1.0 / rhy if rhy > 0.0 else 0.0
    r_in = rhx * RING_INNER_RATIO
    r_in2 = r_in * r_in
    rhx2 = rhx * rhx

    for r in range(r0, r1):
        dy = r - fy          # смещение строки от центра фигуры, в пикселях
        left: float
        right: float
        if kind == _KIND_TRI_UP:
            # Основание внизу (dy = +rhy), вершина вверху (dy = -rhy):
            # полуширина линейно тает кверху.
            w = rhx * (dy + rhy) * inv_rhy * 0.5
            left, right = fx - w, fx + w
        elif kind == _KIND_TRI_DOWN:
            w = rhx * (rhy - dy) * inv_rhy * 0.5
            left, right = fx - w, fx + w
        elif kind == _KIND_TRI_LEFT:
            # Вершина слева, основание справа: левый край отъезжает вправо.
            left = fx - rhx + 2.0 * rhx * abs(dy) * inv_rhy
            right = fx + rhx
        elif kind == _KIND_TRI_RIGHT:
            left = fx - rhx
            right = fx + rhx - 2.0 * rhx * abs(dy) * inv_rhy
        elif kind == _KIND_CIRCLE:
            d2 = rhx2 - dy * dy
            if d2 <= 0.0:
                continue
            w = math.sqrt(d2)
            left, right = fx - w, fx + w
        elif kind == _KIND_ELLIPSE:
            t = 1.0 - (dy * inv_rhy) ** 2
            if t <= 0.0:
                continue
            w = rhx * math.sqrt(t)
            left, right = fx - w, fx + w
        elif kind == _KIND_RING:
            d2 = rhx2 - dy * dy
            if d2 <= 0.0:
                continue
            w_out = math.sqrt(d2)
            d2_in = r_in2 - dy * dy
            if d2_in > 0.0:
                # Строка проходит через дырку: две дуги, слева и справа.
                w_in = math.sqrt(d2_in)
                c0 = int(math.ceil(fx - w_out))
                c1 = int(math.floor(fx - w_in)) + 1
                if c0 < 0:
                    c0 = 0
                if c1 > view_w:
                    c1 = view_w
                if c0 < c1:
                    base = r * view_w
                    buf[base + c0:base + c1] = fill[:c1 - c0]
                left, right = fx + w_in, fx + w_out
            else:
                left, right = fx - w_out, fx + w_out
        else:  # pragma: no cover - неизвестных фигур в таблице нет
            left, right = fx - rhx, fx + rhx

        c0 = int(math.ceil(left))
        c1 = int(math.floor(right)) + 1
        if c0 < 0:
            c0 = 0
        if c1 > view_w:
            c1 = view_w
        if c0 < c1:
            base = r * view_w
            buf[base + c0:base + c1] = fill[:c1 - c0]


def _draw_player(
    sem: np.ndarray,
    cam_x: float,
    cam_y: float,
    state: PlayerState,
    view_w: int,
    view_h: int,
) -> None:
    """Нарисовать игрока прямоугольником его хитбокса, поверх всего остального.

    Игрок рисуется одинаково во всех режимах (куб/корабль/волна) — форму
    политика всё равно узнаёт из `features`, а зрению важно найти именно
    коробку столкновений. Если камера почему-то оставила игрока за кадром
    (возможно только при нестандартных `view_w/view_h`), рисуем хотя бы один
    прижатый к краю пиксель: карта без игрока для политики бессмысленна.
    """
    hx, hy = player_half_extent(state.mode)
    r0, r1 = _row_span(state.y - hy, state.y + hy, cam_y, view_h)
    c0, c1 = _col_span(state.x - hx, state.x + hx, cam_x, view_w)
    if r0 < r1 and c0 < c1:
        sem[r0:r1, c0:c1] = PLAYER
        return
    px, py = world_to_pixel(state.x, state.y, (cam_x, cam_y), view_h)
    px = min(max(px, 0), view_w - 1)
    py = min(max(py, 0), view_h - 1)
    sem[py, px] = PLAYER


def _draw_rank(obj: LevelObject) -> int:
    """Ключ сортировки: чем больше ранг, тем позже (и значит поверх) рисуем."""
    return _DRAW[obj.type][0]


# ---------------------------------------------------------------------------
# публичное API
# ---------------------------------------------------------------------------
def render_semantic(
    level: Level,
    state: PlayerState,
    view_w: int = OBS_W,
    view_h: int = OBS_H,
) -> np.ndarray:
    """Каноническая карта кадра: uint8 (view_h, view_w) с классами 0..9.

    Камера ставит игрока на `PLAYER_X_IN_VIEW`, ось Y перевёрнута (верх мира —
    меньший индекс строки). Порядок заливки повторяет приоритет из SPEC §7
    (EMPTY < SOLID < PORTAL_* < PAD < ORB < GOAL < HAZARD < PLAYER): объекты
    сортируются по рангу и рисуются от младшего к старшему, поэтому шип поверх
    блока виден как шип, а игрок — всегда виден.

    Пол (всё ниже `GROUND_Y`) и потолок (всё выше `level.ceiling_y`) заливаются
    SOLID до края кадра: для политики это такие же твёрдые поверхности, как
    блоки, и рисовать их «линией» значило бы врать про геометрию мира.
    """
    view_w = int(view_w)
    view_h = int(view_h)
    if view_w <= 0 or view_h <= 0:
        raise ValueError(f"Размер кадра должен быть положительным, получено {view_w}x{view_h}")

    cam_x, cam_y = camera_origin(state)
    sem = np.zeros((view_h, view_w), dtype=np.uint8)   # EMPTY == 0

    # Пол: от нижнего края кадра до GROUND_Y.
    r0, r1 = _row_span(cam_y - 1.0, GROUND_Y, cam_y, view_h)
    if r0 < r1:
        sem[r0:r1, :] = SOLID

    # Потолок: от ceiling_y до верхнего края кадра.
    top_world = cam_y + view_h * _INV_P + 1.0
    ceiling_y = float(level.ceiling_y)
    if ceiling_y < top_world:
        r0, r1 = _row_span(ceiling_y, top_world, cam_y, view_h)
        if r0 < r1:
            sem[r0:r1, :] = SOLID

    # Объекты в полосе кадра (+ запас на объект, чей центр чуть за краем).
    objs = level.objects_in_range(cam_x - 1.0, cam_x + view_w * _INV_P + 1.0)
    if objs:
        objs.sort(key=_draw_rank)
        # Плоский вид на тот же буфер: заливка отрезка — одна запись в память.
        buf = memoryview(sem).cast("B")
        fills = _fill_bytes(view_w)
        draw = _DRAW
        for obj in objs:
            _, kind, hx, hy, cls = draw[obj.type]
            _draw_shape(sem, buf, fills[cls], cam_x, cam_y, kind,
                        obj.x, obj.y, hx, hy, cls, view_w, view_h)
        buf.release()

    _draw_player(sem, cam_x, cam_y, state, view_w, view_h)
    return sem


def semantic_to_rgb(sem: np.ndarray) -> np.ndarray:
    """Раскрасить карту по `CLASS_COLORS` -> (H, W, 3) uint8.

    Зачем: только для человека — окно «что видит нейросеть», отладка датасета
    и картинки в README. В обучении используется сама карта классов.
    """
    arr = np.asarray(sem)
    if arr.ndim != 2:
        raise ValueError(f"Ожидалась карта (H, W), получено {arr.shape}")
    if arr.size and int(arr.max()) >= NUM_CLASSES:
        raise ValueError(
            f"В карте есть класс {int(arr.max())}, а классов всего {NUM_CLASSES}"
        )
    return _CLASS_RGB_LUT[arr.astype(np.intp, copy=False)]


def downsample_semantic(sem: np.ndarray, factor: int = 2) -> np.ndarray:
    """Сжать карту в `factor` раз, отдавая блок самому приоритетному классу.

    Зачем не «самый частый» и не «ближайший сосед»: политика работает на
    сжатой карте (36x64), а шип занимает всего несколько пикселей. Усреднение
    или мода стёрли бы его — агент узнал бы о шипе, только влетев в него.
    Поэтому побеждает опасность: PLAYER > HAZARD > GOAL > ORB > PAD >
    PORTAL_* > SOLID > EMPTY (та же шкала, что и при отрисовке).
    """
    arr = np.asarray(sem)
    if arr.ndim != 2:
        raise ValueError(f"Ожидалась карта (H, W), получено {arr.shape}")
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor должен быть >= 1, получено {factor}")
    if factor == 1:
        return arr.astype(np.uint8, copy=True)
    h, w = arr.shape
    if h % factor or w % factor:
        raise ValueError(
            f"Размер карты {h}x{w} не делится на factor={factor} — "
            "сжатие исказило бы геометрию"
        )
    ranks = _PRIORITY_LUT[arr.astype(np.intp, copy=False)]
    blocks = ranks.reshape(h // factor, factor, w // factor, factor)
    best = blocks.max(axis=3).max(axis=1)
    return _PRIORITY_INV_LUT[best]


def class_priority(cls: int) -> int:
    """Ранг класса в шкале приоритетов (больше — важнее, перекрывает остальных).

    Зачем публично: тесты и `render.py` должны сверяться с тем же порядком, а
    не переписывать его константами у себя.
    """
    c = int(cls)
    if not 0 <= c < NUM_CLASSES:
        raise ValueError(f"Класс {c} вне диапазона 0..{NUM_CLASSES - 1}")
    return _PRIORITY[c]


def view_bounds(cam: tuple[float, float], view_w: int = OBS_W, view_h: int = OBS_H) -> tuple[float, float, float, float]:
    """Видимый прямоугольник мира `(x0, y0, x1, y1)` в тайлах для данной камеры.

    Зачем: рендерер декораций и генератор партиклов должны знать, что вообще
    попадает в кадр, и обязаны считать это из той же камеры, что и разметка.
    """
    cam_x, cam_y = cam
    return (cam_x, cam_y, cam_x + int(view_w) * _INV_P, cam_y + int(view_h) * _INV_P)


__all__ = [
    "render_semantic",
    "camera_origin",
    "world_to_pixel",
    "semantic_to_rgb",
    "downsample_semantic",
    "class_priority",
    "view_bounds",
    "SHAPE_BY_TYPE",
    "CAM_Y_ANCHOR",
    "CAM_GROUND_MARGIN",
    "CAM_SOFT_K",
    "RING_INNER_RATIO",
]
