"""Физика игрока: куб, корабль, волна.

Это сердце проекта: политика учится не «на картинке», а на последствиях своих
действий, и любое расхождение с оригинальной Geometry Dash сразу превращает
обучение в бессмыслицу. Поэтому здесь важны три вещи:

1. **Чистота.** `step_physics` не мутирует вход и не имеет глобального
   состояния — иначе нельзя ни откатывать состояние в поиске проходимости
   (`is_solvable`), ни воспроизводить эпизод по seed.
2. **Детерминизм.** Никакого pygame, никакой случайности, только float.
3. **Точность параболы.** Вертикаль интегрируется трапецией
   (`dy = (vy_до + vy_после)/2 * dt`), что для постоянного ускорения совпадает
   с аналитическим решением. Наивное `y += vy*dt` завысило бы прыжок до 2.56
   тайла вместо канонических 2.4 — а от высоты прыжка зависит вся геометрия
   уровней.

Гравитационная система координат
--------------------------------
Порталы переворачивают гравитацию, и писать каждую проверку дважды («если
вниз… если вверх…») — прямой путь к ошибкам. Поэтому вертикаль считается в
«гравитационной» системе: `Y = y * gravity`, `VY = vy * gravity`. В ней
гравитация ВСЕГДА тянет к меньшему Y, «пол» всегда снизу, а обратное
преобразование — то же умножение на gravity (±1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from gdai.constants import (
    DEFAULT_SPEED_INDEX,
    DT,
    GRAVITY,
    GROUND_Y,
    HAZARD,
    JUMP_V,
    MAX_FALL_V,
    ORB,
    ORB_PINK_V,
    ORB_RED_V,
    ORB_YELLOW_V,
    PAD,
    PAD_PINK_V,
    PAD_RED_V,
    PAD_YELLOW_V,
    PLAYER_HALF,
    PLAYER_HALF_SHIP,
    PLAYER_HALF_WAVE,
    PORTAL_GRAVITY,
    PORTAL_MODE,
    PORTAL_SPEED,
    SHIP_GRAVITY,
    SHIP_MAX_V,
    SHIP_THRUST,
    SOLID,
    SPEED_TILES_PER_SEC,
    WAVE_SPEED_RATIO,
    GOAL,
)
from gdai.env.level import Level, LevelObject

# Насколько глубоким должно быть перекрытие, чтобы считаться контактом.
# Зачем: после «прилипания» к блоку координаты вида 1.45-0.45 дают ошибку
# порядка 1e-16, и без допуска игрок умирал бы от собственного пола.
CONTACT_EPS: float = 1e-9
# Допуск «шага вверх» при боковом контакте. Если куб зацепил блок сбоку на
# доли миллитайла, честнее поставить его сверху, чем убить: в оригинале
# приземление тоже прощает микроскопические перекрытия.
STEP_UP_TOLERANCE: float = 0.02
# Ширина полосы поиска объектов вокруг игрока (тайлы). За кадр игрок проходит
# максимум 0.32 тайла, так что 2 тайла — с большим запасом.
_SEARCH_MARGIN: float = 2.0
# На какой высоте над полом стартует волна (тайлы): касание пола для неё смертельно.
WAVE_START_HEIGHT: float = 2.0

_PAD_VELOCITY: dict[str, float] = {
    "pad_yellow": PAD_YELLOW_V,
    "pad_pink": PAD_PINK_V,
    "pad_red": PAD_RED_V,
}
_ORB_VELOCITY: dict[str, float] = {
    "orb_yellow": ORB_YELLOW_V,
    "orb_pink": ORB_PINK_V,
    "orb_red": ORB_RED_V,
}
# Таблицы порталов публичны: генератор и среда обязаны уметь ответить, с каким
# режимом и какой гравитацией игрок доезжает до середины уровня (старт с
# practice-чекпойнта). Дублировать это соответствие у себя они не имеют права —
# разъехавшиеся копии означали бы, что чекпойнт восстанавливает не ту физику.
PORTAL_MODE_TARGET: dict[str, str] = {
    "portal_cube": "cube",
    "portal_ship": "ship",
    "portal_wave": "wave",
}
PORTAL_GRAVITY_TARGET: dict[str, int] = {
    "portal_gravity_down": 1,
    "portal_gravity_up": -1,
}


@dataclass
class PlayerState:
    """Полное состояние игрока — всё, что нужно для следующего кадра.

    Зачем `hold_prev`: кольцо в Geometry Dash срабатывает по ФРОНТУ нажатия,
    а не по факту удержания. Без памяти о прошлом кадре агент, зажавший
    кнопку, использовал бы одно кольцо шестьдесят раз в секунду.
    """

    x: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    mode: str = "cube"                        # "cube"|"ship"|"wave"
    gravity: int = 1                          # +1 вниз, -1 вверх
    speed_index: int = DEFAULT_SPEED_INDEX
    on_ground: bool = False
    alive: bool = True
    finished: bool = False
    hold_prev: bool = False                   # было ли удержание на прошлом кадре


def player_half_extent(mode: str) -> tuple[float, float]:
    """Полуразмеры хитбокса игрока для режима.

    Зачем волна меньше: её коридоры узкие, и с кубическим хитбоксом они были
    бы физически непроходимы.
    """
    if mode == "wave":
        return (PLAYER_HALF_WAVE, PLAYER_HALF_WAVE)
    if mode == "ship":
        return (PLAYER_HALF_SHIP, PLAYER_HALF_SHIP)
    return (PLAYER_HALF, PLAYER_HALF)


def speed_of(state: PlayerState) -> float:
    """Горизонтальная скорость в тайлах в секунду для текущего портала скорости."""
    idx = min(max(int(state.speed_index), 0), len(SPEED_TILES_PER_SEC) - 1)
    return SPEED_TILES_PER_SEC[idx]


def make_initial_state(
    level: Level,
    start_x: float = 0.0,
    *,
    mode: str | None = None,
    gravity: int | None = None,
    speed_index: int | None = None,
) -> PlayerState:
    """Стартовое состояние на уровне: игрок стоит на «полу» своей гравитации.

    Зачем в физике, а не в среде: этим же состоянием пользуется проверка
    проходимости уровня в генераторе, которая про среду ничего не знает.

    Зачем необязательные `mode`/`gravity`/`speed_index`: старт с середины уровня
    (practice-чекпойнт, отладка участка) обязан учитывать порталы, пройденные
    до этой точки. Поставить игрока «как на старте» посреди секции корабля или
    за портáлом перевёрнутой гравитации значит уронить его в потолок и
    засчитать смерть, которой в честной игре не было бы. По умолчанию (None)
    берутся настройки уровня — обычный старт с нуля.
    """
    mode = level.start_mode if mode is None else str(mode)
    gravity = int(level.start_gravity if gravity is None else gravity)
    _, half_y = player_half_extent(mode)
    # В гравитационной системе «пол» — это меньшая из границ мира.
    floor_y = GROUND_Y if gravity > 0 else level.ceiling_y
    # Волна умирает от касания пола, поэтому её нельзя ставить вплотную:
    # без отступа эпизод заканчивался бы на первом же кадре без удержания.
    offset = WAVE_START_HEIGHT if mode == "wave" else half_y
    y = floor_y + offset * gravity
    return PlayerState(
        x=float(start_x),
        y=float(y),
        vy=0.0,
        mode=mode,
        gravity=gravity,
        speed_index=int(
            level.start_speed_index if speed_index is None else speed_index
        ),
        on_ground=(mode != "wave"),
        alive=True,
        finished=False,
        hold_prev=False,
    )


def _overlaps(
    px: float, py: float, phx: float, phy: float, obj: LevelObject, eps: float
) -> bool:
    """Пересечение AABB игрока с AABB объекта с допуском eps."""
    ohx, ohy = obj.half_extent()
    return (
        abs(px - obj.x) < phx + ohx - eps
        and abs(py - obj.y) < phy + ohy - eps
    )


def step_physics(
    state: PlayerState,
    level: Level,
    hold: bool,
    dt: float = DT,
) -> tuple[PlayerState, dict[str, Any]]:
    """Один кадр физики. Возвращает новое состояние и словарь событий.

    События: {"died": bool, "finished": bool, "jumped": bool, "used_orb": bool,
    "used_pad": bool, "portal": str|None}. Функция ЧИСТАЯ — не мутирует вход.

    Порядок операций воспроизводит поведение Geometry Dash:
    1) горизонталь и боковой удар о блок (= смерть);
    2) порталы, пады (сами) и кольца (только по фронту нажатия);
    3) вертикальная скорость по режиму;
    4) вертикальное смещение и разбор столкновений с блоками;
    5) пол и потолок мира;
    6) шипы и финиш.
    """
    events: dict[str, Any] = {
        "died": False,
        "finished": False,
        "jumped": False,
        "used_orb": False,
        "used_pad": False,
        "portal": None,
    }
    hold = bool(hold)

    # Мёртвый/финишировавший игрок замирает: среда сама решит, когда сбросить
    # эпизод, а нам нельзя менять его состояние задним числом.
    if not state.alive or state.finished:
        return replace(state, hold_prev=hold), events

    mode = state.mode
    gravity = int(state.gravity)
    speed_index = int(state.speed_index)
    y = float(state.y)
    vy = float(state.vy)
    on_ground = bool(state.on_ground)
    half_x, half_y = player_half_extent(mode)
    died = False
    finished = False

    # --- 1. горизонталь ----------------------------------------------------
    x = float(state.x) + speed_of(state) * dt
    nearby = level.objects_in_range(x - _SEARCH_MARGIN, x + _SEARCH_MARGIN)

    for obj in nearby:
        if obj.semantic_class() != SOLID:
            continue
        ohx, ohy = obj.half_extent()
        if abs(x - obj.x) >= half_x + ohx - CONTACT_EPS:
            continue
        pen_y = (half_y + ohy) - abs(y - obj.y)
        if pen_y <= CONTACT_EPS:
            continue
        # Микроскопическое зацепление сверху при движении вниз — это не удар
        # в стену, а неточность float: ставим куб на блок.
        if (
            mode == "cube"
            and vy * gravity <= 0.0
            and y * gravity > obj.y * gravity
            and pen_y <= STEP_UP_TOLERANCE
        ):
            y = (obj.y * gravity + ohy + half_y) * gravity
            vy = 0.0
            on_ground = True
            continue
        died = True
        break

    if died:
        events["died"] = True
        return (
            replace(
                state, x=x, y=y, vy=vy, mode=mode, gravity=gravity,
                speed_index=speed_index, on_ground=on_ground,
                alive=False, finished=False, hold_prev=hold,
            ),
            events,
        )

    # --- 2. порталы, пады, кольца ------------------------------------------
    rising_edge = hold and not state.hold_prev
    pad_hit: LevelObject | None = None
    orb_hit: LevelObject | None = None

    for obj in nearby:
        cls = obj.semantic_class()
        if cls in (PORTAL_GRAVITY, PORTAL_MODE, PORTAL_SPEED):
            if not _overlaps(x, y, half_x, half_y, obj, 0.0):
                continue
            if cls == PORTAL_GRAVITY:
                target = PORTAL_GRAVITY_TARGET[obj.type]
                if target != gravity:
                    gravity = target
                    on_ground = False
                    events["portal"] = obj.type
            elif cls == PORTAL_MODE:
                target_mode = PORTAL_MODE_TARGET[obj.type]
                if target_mode != mode:
                    mode = target_mode
                    half_x, half_y = player_half_extent(mode)
                    on_ground = False
                    events["portal"] = obj.type
            else:  # PORTAL_SPEED
                target_idx = int(obj.type.rsplit("_", 1)[1])
                if target_idx != speed_index:
                    speed_index = target_idx
                    events["portal"] = obj.type
        elif cls == PAD and pad_hit is None:
            # Условие vy*gravity <= 0 заменяет собой флаг «пад уже использован»:
            # после срабатывания игрок летит вверх, поэтому на следующих кадрах
            # тот же пад не подбросит его повторно. Без этого один пад разгонял
            # бы куба всё время, пока он его касается.
            if state.vy * gravity <= 0.0 and _overlaps(x, y, half_x, half_y, obj, 0.0):
                pad_hit = obj
        elif cls == ORB and orb_hit is None and rising_edge:
            if _overlaps(x, y, half_x, half_y, obj, 0.0):
                orb_hit = obj
        elif cls == GOAL:
            if _overlaps(x, y, half_x, half_y, obj, 0.0):
                finished = True

    if pad_hit is not None:
        # Пад срабатывает сам, без нажатия, и гасит инерцию — как в оригинале.
        vy = _PAD_VELOCITY[pad_hit.type] * gravity
        on_ground = False
        events["used_pad"] = True
    if orb_hit is not None:
        # Кольцо перебивает пад: игрок нажал осознанно.
        vy = _ORB_VELOCITY[orb_hit.type] * gravity
        on_ground = False
        events["used_orb"] = True

    # --- 3. вертикальная скорость ------------------------------------------
    # Считаем в гравитационной системе: VY > 0 — «вверх» относительно игрока.
    VY = vy * gravity
    if mode == "cube":
        if on_ground and hold:
            # Куб при удержании прыгает снова сразу после приземления.
            VY = JUMP_V
            on_ground = False
            events["jumped"] = True
        VY_next = VY - GRAVITY * dt
        if VY_next < -MAX_FALL_V:
            VY_next = -MAX_FALL_V
        dY = 0.5 * (VY + VY_next) * dt      # трапеция == точная парабола
    elif mode == "ship":
        accel = SHIP_THRUST if hold else -SHIP_GRAVITY
        VY_next = VY + accel * dt
        if VY_next > SHIP_MAX_V:
            VY_next = SHIP_MAX_V
        elif VY_next < -SHIP_MAX_V:
            VY_next = -SHIP_MAX_V
        dY = 0.5 * (VY + VY_next) * dt
    else:  # wave — мгновенная диагональ, без инерции
        wave_v = speed_of(state) * WAVE_SPEED_RATIO
        VY_next = wave_v if hold else -wave_v
        dY = VY_next * dt
        on_ground = False

    Y = y * gravity + dY
    VY = VY_next

    # Опора обязана подтверждаться КАЖДЫЙ кадр. Без этого сброса игрок, сошедший
    # с края блока, навсегда оставался бы «на земле»: он мог бы прыгать в
    # воздухе, признак `on_ground` врал бы политике, а поиск проходимости считал
    # бы проходимыми уровни, которые пройти нельзя. Стоящий на поверхности игрок
    # ничего не теряет — гравитация тут же вдавливает его обратно, и разбор
    # столкновений (шаги 4 и 5) возвращает флаг на место в этом же кадре.
    on_ground = False

    # --- 4. столкновения с блоками по вертикали -----------------------------
    land_Y: float | None = None
    for obj in nearby:
        if obj.semantic_class() != SOLID:
            continue
        ohx, ohy = obj.half_extent()
        if abs(x - obj.x) >= half_x + ohx - CONTACT_EPS:
            continue
        obj_Y = obj.y * gravity
        if (half_y + ohy) - abs(Y - obj_Y) <= CONTACT_EPS:
            continue
        if mode != "cube":
            # Для корабля и волны любое касание блока — смерть (SPEC §5.4).
            died = True
            break
        if VY <= 0.0 and Y > obj_Y:
            top = obj_Y + ohy + half_y
            land_Y = top if land_Y is None else max(land_Y, top)
        else:
            # Удар «в лоб» снизу или встречный — смерть.
            died = True
            break

    if not died and land_Y is not None:
        Y = land_Y
        VY = 0.0
        on_ground = True

    # --- 5. пол и потолок мира ---------------------------------------------
    if not died:
        # В гравитационной системе нижняя граница — та, к которой тянет.
        floor_Y = min(GROUND_Y * gravity, level.ceiling_y * gravity)
        roof_Y = max(GROUND_Y * gravity, level.ceiling_y * gravity)
        if Y - half_y <= floor_Y + CONTACT_EPS:
            if mode == "wave":
                died = True
            else:
                Y = floor_Y + half_y
                VY = 0.0
                on_ground = True
        elif Y + half_y >= roof_Y - CONTACT_EPS:
            if mode == "wave":
                died = True
            else:
                # В потолок мира упираемся, но «землёй» он не становится.
                Y = roof_Y - half_y
                VY = 0.0

    y = Y * gravity
    vy = VY * gravity
    if mode == "wave":
        on_ground = False

    # --- 6. шипы и финиш ----------------------------------------------------
    if not died:
        for obj in nearby:
            if obj.semantic_class() != HAZARD:
                continue
            if _overlaps(x, y, half_x, half_y, obj, 0.0):
                died = True
                break

    if died:
        events["died"] = True
        finished = False
    else:
        if x >= level.length:
            finished = True
        if finished:
            events["finished"] = True

    return (
        replace(
            state,
            x=x,
            y=y,
            vy=vy,
            mode=mode,
            gravity=gravity,
            speed_index=speed_index,
            on_ground=on_ground,
            alive=not died,
            finished=finished,
            hold_prev=hold,
        ),
        events,
    )


__all__ = [
    "PlayerState",
    "step_physics",
    "player_half_extent",
    "speed_of",
    "make_initial_state",
    "PORTAL_MODE_TARGET",
    "PORTAL_GRAVITY_TARGET",
    "CONTACT_EPS",
    "STEP_UP_TOLERANCE",
    "WAVE_START_HEIGHT",
]
