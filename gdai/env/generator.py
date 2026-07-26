"""Процедурная генерация уровней, которые гарантированно проходимы.

Идея
----
Уровень собирается не «россыпью объектов по шуму Перлина», а **участками**
(паттернами): одиночный шип, ряд шипов, лестница блоков, остров среди шипов,
низкий коридор, цепочка колец, пад-прыжок, секция корабля, секция волны,
смена скорости, смена гравитации, пила, зубчатый пол, отдых. Каждый паттерн —
маленький, самодостаточный кусок дизайна, который умеет масштабироваться по
сложности: чем выше `difficulty`, тем плотнее, шире и злее.

Почему уровни проходимы
-----------------------
Геометрия паттернов считается по настоящим формулам прыжка (высота 2.4 тайла,
полёт 0.5 с), но одной аккуратности мало: скорость меняется порталами, участки
стыкуются в воздухе, гравитация переворачивается. Поэтому после КАЖДОГО
участка префикс уровня проверяется поиском по кадрам (`gdai.env.solver`).
Непроходимый участок переигрывается другими случайными параметрами, а если и
это не помогло — заменяется на «отдых», который проходим всегда. Проверка
инкрементальная: поиск продолжается с сохранённого фронта, поэтому построение
линейно по длине уровня, а не квадратично.

Система координат паттернов
---------------------------
Паттерны пишутся в **локальной системе**, где y = 0 — это «пол под ногами»,
а рост y — «вверх для игрока». При обычной гравитации это мировые координаты,
при перевёрнутой — зеркало относительно потолка (`y_world = ceiling_y - y`),
а шипы меняют ориентацию. Благодаря этому один и тот же код паттерна работает
и на полу, и на потолке — иначе каждый паттерн пришлось бы писать дважды.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from gdai.constants import (
    DEFAULT_SPEED_INDEX,
    DT,
    GRAVITY,
    HAZARD,
    HAZARD_HALF,
    JUMP_V,
    PAD_PINK_V,
    PAD_RED_V,
    PAD_YELLOW_V,
    PLAYER_HALF,
    SHIP_THRUST,
    SOLID,
    SPEED_TILES_PER_SEC,
)
from gdai.env.level import Level, LevelObject
from gdai.env.physics import PlayerState, make_initial_state
from gdai.env.solver import (
    SearchResult,
    is_solvable,
    search_forward,
    solve_actions,
    state_key,
)
from gdai.utils.logging import get_logger

_log = get_logger("env.generator")

# --- геометрические константы паттернов -------------------------------------
CEILING_Y: float = 12.0          # высота мира; совпадает с Level.ceiling_y
SPIKE_Y: float = 0.5             # центр шипа, стоящего на полу
# Высота центра игрока, при которой он гарантированно выше шипа.
SPIKE_CLEAR_Y: float = SPIKE_Y + HAZARD_HALF + PLAYER_HALF          # 1.23
SAW_HALF: float = 0.40           # см. gdai/env/level.py::_SAW_HALF
SAW_CLEAR_Y: float = SPIKE_Y + SAW_HALF + PLAYER_HALF               # 1.35
PORTAL_STACK_STEP: float = 2.5   # порталы ставим стопкой, шаг = их высота

START_RUNWAY: float = 4.0        # безопасный разгон перед первым препятствием
TAIL_RUNWAY: float = 7.0         # безопасный участок перед финишем
_GRAVITY_RETURN_RESERVE: float = 14.0   # место на возврат гравитации перед финишем
_SENTINEL_LENGTH: float = 1.0e9  # «финиша ещё нет» на время постройки

# --- бюджеты генерации ------------------------------------------------------
SEGMENT_ATTEMPTS: int = 3        # столько раз переигрываем непроходимый участок
LEVEL_ATTEMPTS: int = 3          # столько раз пересобираем уровень целиком
# Фронт при сборке уже, чем по умолчанию в solver: генерация запускается
# тысячами (учебный план RL), и лишняя ширина здесь стоит секунд на уровень,
# а теряет считанные проценты участков — их всё равно перегенерируют.
_BUILD_MAX_FRONTIER: int = 64
_BUILD_MAX_NODES: int = 60_000   # на один участок
_FINAL_MAX_NODES: int = 400_000  # на финальную проверку целого уровня

# Зеркало типов при перевёрнутой гравитации: хитбокс симметричен, но
# «смотреть» шип обязан в сторону игрока, иначе картинка врёт разметке.
_MIRROR_TYPE: dict[str, str] = {
    "spike": "spike_down",
    "spike_down": "spike",
}


# ---------------------------------------------------------------------------
# Контекст и низкоуровневые помощники
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatternContext:
    """Состояние «мира» на входе в участок.

    Зачем неизменяемый: паттерн возвращает НОВЫЙ контекст, а старый остаётся
    у генератора. Если участок оказался непроходимым и откатывается, откатить
    нужно и режим/скорость/гравитацию — с mutable-объектом это была бы вечная
    охота за побочными эффектами.
    """

    mode: str = "cube"                       # режим игрока на входе
    gravity: int = 1                         # +1 вниз, -1 вверх
    speed_index: int = DEFAULT_SPEED_INDEX
    ceiling_y: float = CEILING_Y
    since_special: int = 99                  # участков с последней спец-секции


PatternResult = tuple[list[LevelObject], float, PatternContext]
PatternFn = Callable[[np.random.Generator, float, float, PatternContext], PatternResult]


def _world_y(y_local: float, ctx: PatternContext) -> float:
    """Локальная высота («над полом игрока») -> мировая координата y."""
    return float(y_local) if ctx.gravity > 0 else ctx.ceiling_y - float(y_local)


def _put(
    out: list[LevelObject], ctx: PatternContext, obj_type: str, x: float, y_local: float
) -> None:
    """Положить объект, пересчитав локальную высоту и ориентацию по гравитации."""
    obj_type = obj_type if ctx.gravity > 0 else _MIRROR_TYPE.get(obj_type, obj_type)
    out.append(LevelObject(obj_type, float(x), _world_y(y_local, ctx)))


def _portal_stack(
    out: list[LevelObject], ctx: PatternContext, portal_type: str, x: float, count: int = 2
) -> None:
    """Портал стопкой в 2 клетки высотой.

    Зачем стопка: портал ловит игрока полосой ±1.25 тайла, и куб на вершине
    прыжка (центр 2.85) в одиночный портал у пола ещё попадает, а вылетевший
    с пада — уже нет. Пропущенный портал раздваивает состояние мира (часть
    веток с новой скоростью, часть со старой), что и путает поиск, и делает
    уровень нечестным. Второй экземпляр — no-op, если первый уже сработал.
    """
    for i in range(count):
        y_local = 1.25 + i * PORTAL_STACK_STEP
        out.append(LevelObject(portal_type, float(x), _world_y(y_local, ctx)))


def _column(
    out: list[LevelObject], ctx: PatternContext, x: float, height: float, base: float = 0.0
) -> None:
    """Столбик блоков высотой `height` тайлов, стоящий на локальной высоте `base`."""
    for i in range(int(round(height))):
        _put(out, ctx, "block", x, base + i + 0.5)


def _pillars(
    out: list[LevelObject],
    ctx: PatternContext,
    x: float,
    gap_lo: float,
    gap_hi: float,
    width: int = 1,
) -> None:
    """Пара колонн от пола и от потолка, оставляющая коридор (gap_lo, gap_hi)."""
    for w in range(width):
        cx = x + w
        y = 0.5
        while y + 0.5 <= gap_lo + 1e-9:
            _put(out, ctx, "block", cx, y)
            y += 1.0
        y = ctx.ceiling_y - 0.5
        while y - 0.5 >= gap_hi - 1e-9:
            _put(out, ctx, "block", cx, y)
            y -= 1.0


def _speed(ctx: PatternContext) -> float:
    """Горизонтальная скорость в тайлах/с для текущего портала скорости."""
    return SPEED_TILES_PER_SEC[ctx.speed_index]


def _jump_span(ctx: PatternContext) -> float:
    """Сколько тайлов проезжает игрок за один полный прыжок.

    Полёт длится 2*JUMP_V/GRAVITY = 0.5 с — это естественная «единица длины»
    для всех паттернов: любой зазор измеряется в долях прыжка.
    """
    return (2.0 * JUMP_V / GRAVITY) * _speed(ctx)


def _airtime_above(
    y_threshold: float, v0: float = JUMP_V, y_from: float = PLAYER_HALF
) -> float:
    """Сколько секунд центр игрока держится выше `y_threshold` после толчка `v0`.

    Зачем: ширина любого «непроходимого пешком» препятствия обязана
    укладываться в это время, умноженное на скорость. Формула — обычная
    парабола, старт с высоты стояния `y_from`.
    """
    disc = v0 * v0 - 2.0 * GRAVITY * (y_threshold - y_from)
    if disc <= 0.0:
        return 0.0
    return 2.0 * math.sqrt(disc) / GRAVITY


def _clear_span(
    ctx: PatternContext, y_threshold: float, v0: float = JUMP_V, y_from: float = PLAYER_HALF
) -> float:
    """Ширина в тайлах, которую игрок пролетает выше `y_threshold`."""
    return _airtime_above(y_threshold, v0, y_from) * _speed(ctx)


def _rise_span(
    ctx: PatternContext, y_target: float, v0: float = JUMP_V, y_from: float = PLAYER_HALF
) -> float:
    """Сколько тайлов проезжает игрок, ПОДНИМАЯСЬ от `y_from` до `y_target`.

    Зачем именно подъём, а не весь полёт: препятствие нужно миновать не
    «когда-нибудь», а к моменту, когда игрок до него доедет. Толчок не
    мгновенный — эта величина и есть та фора, которую обязан давать разбег.
    """
    disc = v0 * v0 - 2.0 * GRAVITY * (y_target - y_from)
    if disc <= 0.0:
        return float("inf")
    return (v0 - math.sqrt(disc)) / GRAVITY * _speed(ctx)


def _slack(ctx: PatternContext, difficulty: float) -> float:
    """Запас по x: сколько кадров игроку прощается при выборе момента прыжка.

    Зачем это главная ручка сложности: физически препятствие проходимо, если
    существует хотя бы один правильный кадр. Уровень, где такой кадр ровно
    один, честен, но невыносим — поэтому окно сжимается со сложностью
    плавно, от четверти секунды до десятой.
    """
    frames = 15.0 - 9.0 * difficulty
    return frames * _speed(ctx) * DT


def _lead_in(
    ctx: PatternContext,
    difficulty: float,
    clear_y: float = SPIKE_CLEAR_Y,
    obstacle_half: float = HAZARD_HALF,
    v0: float = JUMP_V,
) -> float:
    """Свободный разбег перед препятствием.

    Складывается из трёх честных слагаемых: половина игрока + половина
    препятствия (когда начинается опасное перекрытие), путь на подъём до
    безопасной высоты и запас на выбор кадра. Без него участок формально
    проходим «в один кадр из шестидесяти», а на стыке участков — вообще никак.
    """
    return (
        PLAYER_HALF
        + obstacle_half
        + _rise_span(ctx, clear_y, v0)
        + _slack(ctx, difficulty)
    )


def _trail(ctx: PatternContext, difficulty: float) -> float:
    """Место после препятствия, чтобы игрок успел приземлиться и снова прыгнуть.

    Самая ранняя из проходящих траекторий приземляется примерно через 0.45
    длины прыжка после препятствия — на неё и рассчитываем, плюс запас.
    """
    return 0.45 * _jump_span(ctx) + _slack(ctx, difficulty)


def _hop_gap(ctx: PatternContext, difficulty: float) -> float:
    """Расстояние между двумя препятствиями, которые прыгаются по очереди."""
    return _trail(ctx, difficulty) + PLAYER_HALF + HAZARD_HALF + _rise_span(ctx, SPIKE_CLEAR_Y)


def _max_spike_row(ctx: PatternContext, v0: float = JUMP_V, safety: float = 0.85) -> int:
    """Сколько шипов подряд ещё реально перепрыгнуть на текущей скорости.

    Ряд из n шипов «занимает» (n-1) + 2*(PLAYER_HALF+HAZARD_HALF) тайлов, и
    всё это время игрок обязан быть выше шипа. `safety` оставляет запас на
    неидеальный момент прыжка — без него ряд был бы проходим ровно одним
    кадром из шестидесяти.
    """
    span = _clear_span(ctx, SPIKE_CLEAR_Y, v0) * safety
    n = int(math.floor(span - 2.0 * (PLAYER_HALF + HAZARD_HALF))) + 1
    return max(1, n)


def _pick_count(rng: np.random.Generator, difficulty: float, lo: int, hi: int) -> int:
    """Число «штук» в участке: тянется к hi с ростом сложности, но с разбросом."""
    if hi <= lo:
        return lo
    center = lo + (hi - lo) * difficulty
    value = int(round(center + float(rng.normal(0.0, 0.5))))
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Паттерны-участки
# ---------------------------------------------------------------------------
def pattern_rest(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Пустой участок отдыха.

    Зачем нужен: во-первых, ритм — сплошной поток препятствий нечитаем и для
    человека, и для агента; во-вторых, это гарантированно проходимый запасной
    вариант, которым заменяется участок, не поддавшийся перегенерации.
    """
    length = 3.0 + float(rng.uniform(1.0, 5.0)) * (1.0 - 0.5 * difficulty)
    return [], x_start + length, ctx


def pattern_single_spike(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Одиночный шип — базовый «прыгни здесь».

    Разбег слегка «дышит»: без случайности повторная попытка после отказа
    построила бы ровно тот же участок и отказ повторился бы трижды подряд.
    """
    objs: list[LevelObject] = []
    x = x_start + _lead_in(ctx, difficulty) * float(rng.uniform(1.0, 1.5))
    _put(objs, ctx, "spike", x, SPIKE_Y)
    return objs, x + _trail(ctx, difficulty), ctx


def pattern_spike_row(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Ряд шипов вплотную: один прыжок, но времени в воздухе нужно больше."""
    objs: list[LevelObject] = []
    n = _pick_count(rng, difficulty, 2, _max_spike_row(ctx))
    x = x_start + _lead_in(ctx, difficulty)
    for i in range(n):
        _put(objs, ctx, "spike", x + i, SPIKE_Y)
    return objs, x + (n - 1) + _trail(ctx, difficulty), ctx


def pattern_spike_stagger(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Несколько групп шипов подряд — серия прыжков с приземлением между ними."""
    objs: list[LevelObject] = []
    groups = _pick_count(rng, difficulty, 2, 4)
    row_max = max(1, _max_spike_row(ctx) - 1)
    gap = _hop_gap(ctx, difficulty)
    x = x_start + _lead_in(ctx, difficulty)
    for g in range(groups):
        n = _pick_count(rng, difficulty, 1, row_max)
        for i in range(n):
            _put(objs, ctx, "spike", x + i, SPIKE_Y)
        x += n - 1
        if g < groups - 1:
            x += gap
    return objs, x + _trail(ctx, difficulty), ctx


def pattern_block_stairs(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Лестница блоков вверх со спуском обрывом.

    Ступень выше 2 тайлов физически недостижима (прыжок даёт 2.4 тайла),
    поэтому подъём идёт шагами 1-2. Ширина ступени — это время, за которое
    игрок обязан успеть оттолкнуться перед следующей стенкой.
    """
    objs: list[LevelObject] = []
    steps = _pick_count(rng, difficulty, 2, 3)
    x = x_start
    height = 0
    for _ in range(steps):
        dh = 2 if (height <= 2 and rng.random() < 0.25 + 0.35 * difficulty) else 1
        # Разбег до стенки ступени: подняться на dh, стоя на предыдущей.
        x += (
            PLAYER_HALF
            + 0.5
            + _rise_span(ctx, height + dh + PLAYER_HALF, y_from=height + PLAYER_HALF)
            + _slack(ctx, difficulty)
        )
        height += dh
        width = 2 if difficulty > 0.6 else 3
        for w in range(width):
            _column(objs, ctx, x + w, height)
        x += width - 1
    # Спуск — свободное падение с высоты `height`.
    fall = math.sqrt(2.0 * max(float(height), 0.5) / GRAVITY) * _speed(ctx)
    return objs, x + 1.0 + fall + _slack(ctx, difficulty), ctx


def pattern_platform_gap(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Пропасть из шипов с островом-платформой посередине.

    Зачем: заставляет разбить перелёт на два прыжка и приземлиться точно —
    самый частый мотив настоящих уровней GD. Остров ставится там, куда
    приходит нисходящая ветвь траектории, перелетевшей первую группу шипов.
    """
    objs: list[LevelObject] = []
    row_max = max(1, min(3, _max_spike_row(ctx) - 1))
    islands = _pick_count(rng, difficulty, 1, 2)
    plat_h = 1
    plat_w = 3 if difficulty < 0.5 else 2
    # Тонкая платформа ведёт себя как блок (пролезть под ней нельзя), но
    # выглядит и размечается иначе — пусть встречаются оба варианта.
    thin = bool(rng.random() < 0.4)
    x = x_start + _lead_in(ctx, difficulty)
    for i in range(islands + 1):
        n = _pick_count(rng, difficulty, 1, row_max)
        for k in range(n):
            _put(objs, ctx, "spike", x + k, SPIKE_Y)
        x += n - 1
        if i < islands:
            # Между шипами и стенкой острова нужен тот же разбег, что и
            # перед любой стенкой: иначе в неё врезаются лбом.
            x += _trail(ctx, difficulty)
            for w in range(plat_w):
                if thin:
                    _put(objs, ctx, "platform", x + w, plat_h - 0.25)
                else:
                    _column(objs, ctx, x + w, plat_h)
            x += (plat_w - 1) + PLAYER_HALF + HAZARD_HALF + _rise_span(
                ctx, SPIKE_CLEAR_Y, y_from=plat_h + PLAYER_HALF
            ) + _slack(ctx, difficulty)
    fall = math.sqrt(2.0 * max(plat_h, 0.5) / GRAVITY) * _speed(ctx)
    return objs, x + _trail(ctx, difficulty) + fall, ctx


def pattern_corridor(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Низкий потолок: прыгать нельзя, на выходе — шип.

    Зачем: единственный паттерн, который наказывает за лишнее нажатие. Без
    него агент быстро выучивает «держать всегда» и живёт неплохо.
    """
    objs: list[LevelObject] = []
    length = int(round(4 + 5 * difficulty + rng.integers(0, 3)))
    # Игрок обязан войти в коридор по земле, поэтому даём ему приземлиться.
    x = x_start + _trail(ctx, difficulty)
    for i in range(length):
        _put(objs, ctx, "block", x + i, 2.5)
    x_exit = x + length - 1
    if difficulty > 0.35:
        # Свисающие с потолка шипы: пешком безопасны (макушка на 0.9), но
        # окончательно запрещают прыжок. Нужны и для разнообразия разметки —
        # сеть зрения обязана видеть шипы во всех четырёх ориентациях.
        for i in range(1, length - 1, 2):
            _put(objs, ctx, "spike_down", x + i, 2.5 - 0.5 - HAZARD_HALF)
    # Шип за коридором: голова должна успеть миновать последний блок (высота
    # 1.55) РАНЬШЕ, чем начнётся подъём над шипом — отсюда обязательный зазор.
    spike_x = x_exit + 0.5 + PLAYER_HALF + _rise_span(ctx, 2.0 - PLAYER_HALF) + _slack(
        ctx, difficulty
    )
    _put(objs, ctx, "spike", spike_x, SPIKE_Y)
    return objs, spike_x + _trail(ctx, difficulty), ctx


def pattern_jagged_floor(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Зубчатый пол: тумбы высотой в блок, между ними — шип в яме."""
    objs: list[LevelObject] = []
    teeth = _pick_count(rng, difficulty, 2, 4)
    gap = 4.0 - 1.5 * difficulty
    x = x_start + PLAYER_HALF + 0.5 + _rise_span(ctx, 1.0 + PLAYER_HALF) + _slack(
        ctx, difficulty
    )
    for i in range(teeth):
        _column(objs, ctx, x, 1)
        _column(objs, ctx, x + 1, 1)
        x += 1.0
        if i < teeth - 1:
            _put(objs, ctx, "spike", x + gap * 0.5, SPIKE_Y)
            x += gap
    return objs, x + _trail(ctx, difficulty), ctx


def pattern_saw(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Пилы: на полу (перепрыгнуть) или в воздухе (не прыгать)."""
    objs: list[LevelObject] = []
    roll = float(rng.random())
    if roll < 0.45:
        x = x_start + _lead_in(ctx, difficulty, SAW_CLEAR_Y, SAW_HALF)
        n = 1 + int(difficulty * 1.5)
        step = 1.0 + 0.6 * (1.0 - difficulty)
        for i in range(n):
            _put(objs, ctx, "saw", x + i * step, SPIKE_Y)
        x += (n - 1) * step
        return objs, x + _trail(ctx, difficulty), ctx
    # Висящая преграда: пешком безопасно, любой прыжок — смерть.
    x = x_start + _trail(ctx, difficulty)
    n = _pick_count(rng, difficulty, 2, 4)
    if roll < 0.8:
        for i in range(n):
            _put(objs, ctx, "saw", x + i * 1.4, 2.2)
        width = (n - 1) * 1.4
    else:
        # Блок с шипами по бокам: те же «не прыгай», но другая разметка —
        # боковые шипы обязаны появляться в датасете зрения.
        for i in range(n):
            bx = x + i * 3.0
            _put(objs, ctx, "block", bx, 2.5)
            _put(objs, ctx, "spike_left", bx - 0.5 - HAZARD_HALF, 2.5)
            _put(objs, ctx, "spike_right", bx + 0.5 + HAZARD_HALF, 2.5)
        width = (n - 1) * 3.0 + 1.0
    return objs, x + width + 2.0 + _slack(ctx, difficulty), ctx


def pattern_orb_chain(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Поле шипов шире одного прыжка + цепочка колец над ним.

    Кольцо срабатывает по фронту нажатия и задаёт vy заново, поэтому перелёт
    строится как «прыжок + N подхватов». Кольца ставим чаще, чем требует
    физика: их хитбокс щедрый (0.6), и запас по x прощает ошибку в полкадра.
    """
    objs: list[LevelObject] = []
    orbs = _pick_count(rng, difficulty, 1, 3)
    step = _jump_span(ctx) * (0.60 - 0.10 * difficulty)
    x = x_start + _lead_in(ctx, difficulty)
    n_spikes = max(2, int(step * (orbs + 1) - 1.5))
    for i in range(n_spikes):
        _put(objs, ctx, "spike", x + i, SPIKE_Y)
    orb_type = "orb_yellow"
    if difficulty >= 0.6:
        orb_type = str(rng.choice(["orb_yellow", "orb_pink", "orb_red"], p=[0.5, 0.25, 0.25]))
    for k in range(orbs):
        _put(objs, ctx, orb_type, x + step * (k + 1) - 0.5, 1.8)
    return objs, x + (n_spikes - 1) + _trail(ctx, difficulty), ctx


def pattern_pad_jump(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Трамплин перед широким полем шипов.

    Пад срабатывает сам, без нажатия, поэтому важна не реакция, а понимание:
    прыгать ПЕРЕД падом нельзя — в воздухе с vy > 0 он не сработает, а поле
    шипов шире обычного прыжка.
    """
    objs: list[LevelObject] = []
    if difficulty < 0.3 and rng.random() < 0.5:
        pad_type, v0 = "pad_pink", PAD_PINK_V
    elif difficulty >= 0.55 and rng.random() < 0.35:
        pad_type, v0 = "pad_red", PAD_RED_V
    else:
        pad_type, v0 = "pad_yellow", PAD_YELLOW_V
    # Игрок обязан подъехать к паду по земле — даём приземлиться.
    x = x_start + _trail(ctx, difficulty)
    _put(objs, ctx, pad_type, x, 0.25)
    span = _clear_span(ctx, SPIKE_CLEAR_Y, v0) * 0.75
    n_max = max(1, int(math.floor(span - 2.0 * (PLAYER_HALF + HAZARD_HALF))) + 1)
    n = _pick_count(rng, difficulty, 1, n_max)
    first = x + PLAYER_HALF + HAZARD_HALF + _rise_span(ctx, SPIKE_CLEAR_Y, v0)
    for i in range(n):
        _put(objs, ctx, "spike", first + i, SPIKE_Y)
    # Приземление после пада: полный полёт длится 2*v0/GRAVITY секунд.
    flight = 2.0 * v0 / GRAVITY * _speed(ctx)
    return objs, x + flight + _slack(ctx, difficulty), ctx


def pattern_speed_change(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Смена скорости порталом.

    Зачем отдельным участком, а не «поверх» препятствия: скорость меняет
    геометрию ВСЕХ последующих паттернов (длина прыжка пропорциональна ей),
    поэтому портал ставится в чистом месте, где его невозможно проскочить.

    Самая медленная скорость (индекс 0, 0.5x) намеренно редка: она и в
    оригинале экзотика, и растягивает остаток уровня на четверть по времени.
    """
    if difficulty < 0.4:
        table = {1: 3.0, 2: 2.0}
    elif difficulty < 0.75:
        table = {0: 0.3, 1: 2.0, 2: 3.0, 3: 1.5}
    else:
        table = {0: 0.3, 1: 1.0, 2: 2.5, 3: 2.5, 4: 1.2}
    table.pop(ctx.speed_index, None)
    if not table:
        table = {DEFAULT_SPEED_INDEX: 1.0}
    options = list(table)
    probs = np.asarray([table[o] for o in options], dtype=np.float64)
    probs /= probs.sum()
    target = int(options[int(rng.choice(len(options), p=probs))])
    objs: list[LevelObject] = []
    x = x_start + 2.0
    _portal_stack(objs, ctx, f"portal_speed_{target}", x)
    new_ctx = replace(ctx, speed_index=target, since_special=0)
    return objs, x + 3.0, new_ctx


def pattern_gravity_flip(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Переворот гравитации: игрок «падает» на потолок и идёт по нему.

    После портала нужен длинный пустой пролёт — перелёт через все 12 тайлов
    занимает больше полусекунды даже с учётом MAX_FALL_V.
    """
    target = -ctx.gravity
    portal_type = "portal_gravity_down" if target > 0 else "portal_gravity_up"
    objs: list[LevelObject] = []
    x = x_start + 2.0
    # Тип портала описан в мировых терминах, поэтому зеркалить его нельзя:
    # кладём напрямую, только пересчитав высоту.
    for i in range(2):
        objs.append(LevelObject(portal_type, float(x), _world_y(1.25 + i * PORTAL_STACK_STEP, ctx)))
    new_ctx = replace(ctx, gravity=target, since_special=0)
    fall = 0.60 * _speed(ctx) + 3.0
    return objs, x + fall, new_ctx


def pattern_ship_section(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Секция корабля: порталы + коридор из колонн с окнами.

    Корабль умирает от любого касания блока, поэтому окна делаются широкими и
    смещаются между собой не сильнее, чем корабль успевает подняться:
    за время dt = dx/v тяга 52 тайл/с^2 даёт примерно (a/4)*dt^2 при разгоне
    и торможении. Отсюда предел на скачок окна.
    """
    objs: list[LevelObject] = []
    v = _speed(ctx)
    x = x_start + 2.0
    _portal_stack(objs, ctx, "portal_ship", x)

    gates = _pick_count(rng, difficulty, 2, 4)
    spacing = 5.5 - 1.5 * difficulty
    gap = 5.2 - 2.0 * difficulty
    lo_limit = gap * 0.5 + 0.8
    hi_limit = ctx.ceiling_y - gap * 0.5 - 0.8
    center = min(max(2.4, lo_limit), hi_limit)
    max_jump = 0.65 * (SHIP_THRUST * 0.25) * (spacing / v) ** 2

    x += 4.0
    for _ in range(gates):
        _pillars(objs, ctx, x, center - gap * 0.5, center + gap * 0.5, width=1)
        x += spacing
        delta = float(rng.uniform(-max_jump, max_jump))
        center = min(max(center + delta, lo_limit), hi_limit)

    x += 1.5
    _portal_stack(objs, ctx, "portal_cube", x)
    # После возврата в куб игрок падает с любой высоты — пролёт до пола.
    return objs, x + 0.62 * v + 3.0, replace(ctx, mode="cube", since_special=0)


def pattern_wave_section(
    rng: np.random.Generator, difficulty: float, x_start: float, ctx: PatternContext
) -> PatternResult:
    """Секция волны: узкий коридор, движение строго по диагонали 45 градусов.

    Волна умирает и от блока, и от пола/потолка мира, поэтому первое окно
    ставится далеко и высоко: сразу после портала игрок находится в 0.2 тайла
    от смертельного пола и обязан немедленно набирать высоту.
    """
    objs: list[LevelObject] = []
    v = _speed(ctx)
    x = x_start + 2.0
    _portal_stack(objs, ctx, "portal_wave", x)

    gates = _pick_count(rng, difficulty, 2, 4)
    spacing = 5.0 - 1.0 * difficulty
    gap = 3.6 - 1.4 * difficulty
    lo_limit = gap * 0.5 + 0.9
    hi_limit = ctx.ceiling_y - gap * 0.5 - 0.9
    center = min(max(3.5, lo_limit), hi_limit)
    max_jump = 0.55 * spacing        # волна идёт под 45 градусов, не круче

    x += 5.0
    for _ in range(gates):
        _pillars(objs, ctx, x, center - gap * 0.5, center + gap * 0.5, width=1)
        x += spacing
        delta = float(rng.uniform(-max_jump, max_jump))
        center = min(max(center + delta, lo_limit), hi_limit)

    x += 1.5
    _portal_stack(objs, ctx, "portal_cube", x)
    return objs, x + 0.62 * v + 3.0, replace(ctx, mode="cube", since_special=0)


# ---------------------------------------------------------------------------
# Реестр паттернов
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatternSpec:
    """Паттерн + правила его применения.

    `est_length` нужен только для планирования: не начинать двадцатитайловую
    секцию корабля за пять тайлов до финиша. `special` помечает участки с
    порталами — их не ставят подряд, иначе уровень превращается в мигалку.
    """

    func: PatternFn
    name: str
    est_length: float
    min_difficulty: float = 0.0
    weight_low: float = 1.0
    weight_high: float = 1.0
    special: bool = False


# est_length рассчитан на базовую скорость; на быстрых порталах участки
# длиннее, но бюджет всё равно ограничен сверху длиной уровня.
PATTERN_SPECS: tuple[PatternSpec, ...] = (
    PatternSpec(pattern_rest, "rest", 8.0, 0.0, 2.0, 0.5),
    PatternSpec(pattern_single_spike, "single_spike", 10.0, 0.0, 3.0, 1.0),
    PatternSpec(pattern_block_stairs, "block_stairs", 28.0, 0.0, 1.5, 1.2),
    PatternSpec(pattern_spike_row, "spike_row", 12.0, 0.08, 1.0, 2.5),
    PatternSpec(pattern_corridor, "corridor", 24.0, 0.08, 1.0, 1.5),
    PatternSpec(pattern_saw, "saw", 12.0, 0.10, 0.8, 1.6),
    PatternSpec(pattern_spike_stagger, "spike_stagger", 34.0, 0.15, 0.8, 2.5),
    PatternSpec(pattern_pad_jump, "pad_jump", 16.0, 0.15, 0.7, 1.6),
    PatternSpec(pattern_platform_gap, "platform_gap", 30.0, 0.20, 0.6, 2.0),
    PatternSpec(pattern_speed_change, "speed_change", 6.0, 0.20, 0.5, 1.4, special=True),
    PatternSpec(pattern_jagged_floor, "jagged_floor", 26.0, 0.25, 0.5, 2.0),
    PatternSpec(pattern_ship_section, "ship_section", 40.0, 0.25, 0.5, 1.6, special=True),
    PatternSpec(pattern_orb_chain, "orb_chain", 24.0, 0.30, 0.4, 2.0),
    PatternSpec(pattern_gravity_flip, "gravity_flip", 14.0, 0.30, 0.3, 1.4, special=True),
    PatternSpec(pattern_wave_section, "wave_section", 38.0, 0.45, 0.3, 1.4, special=True),
)

PATTERNS: tuple[PatternFn, ...] = tuple(spec.func for spec in PATTERN_SPECS)

_SPECIAL_COOLDOWN: int = 2   # столько обычных участков между спец-секциями


def _choose_spec(
    rng: np.random.Generator, difficulty: float, ctx: PatternContext, budget: float
) -> PatternSpec:
    """Выбрать следующий участок: по сложности, бюджету длины и кулдауну."""
    candidates: list[PatternSpec] = []
    weights: list[float] = []
    for spec in PATTERN_SPECS:
        if difficulty < spec.min_difficulty:
            continue
        if spec.special and ctx.since_special < _SPECIAL_COOLDOWN:
            continue
        if spec.est_length > budget:
            continue
        candidates.append(spec)
        weights.append(max(1e-6, spec.weight_low + (spec.weight_high - spec.weight_low) * difficulty))
    if not candidates:
        return PATTERN_SPECS[0]      # «отдых» помещается всегда
    probs = np.asarray(weights, dtype=np.float64)
    probs /= probs.sum()
    return candidates[int(rng.choice(len(candidates), p=probs))]


# ---------------------------------------------------------------------------
# Сборка уровня
# ---------------------------------------------------------------------------
def _default_length(difficulty: float) -> float:
    """Длина уровня по сложности: сложнее — длиннее, но не бесконечно.

    Зачем ограничение: длина линейно определяет и стоимость проверки
    проходимости, и длину эпизода в RL; уровень на тысячу тайлов не помогает
    учиться, зато делает и генерацию, и обучение неприлично медленными.
    """
    return round(68.0 + 42.0 * difficulty, 1)


def _advance(level: Level, frontier: list[PlayerState], target_x: float) -> SearchResult:
    """Довести фронт поиска до `x >= target_x` по только что добавленному участку."""
    return search_forward(
        level,
        frontier,
        target_x,
        max_nodes=_BUILD_MAX_NODES,
        max_frontier=_BUILD_MAX_FRONTIER,
    )


def _truncate(level: Level, count: int) -> None:
    """Откатить уровень до `count` объектов (участок не прошёл проверку)."""
    del level.objects[count:]
    level.rebuild_index()


def _build_once(
    difficulty: float, rng: np.random.Generator, name: str, target_length: float
) -> Level | None:
    """Одна попытка собрать уровень; None — если фронт поиска где-то умер."""
    start_speed = DEFAULT_SPEED_INDEX
    if difficulty >= 0.6 and rng.random() < 0.3:
        start_speed = 2
    level = Level(
        name=name,
        length=_SENTINEL_LENGTH,
        objects=[],
        start_mode="cube",
        start_speed_index=start_speed,
        start_gravity=1,
        ceiling_y=CEILING_Y,
    )
    ctx = PatternContext(speed_index=start_speed, ceiling_y=CEILING_Y)
    frontier: list[PlayerState] = [make_initial_state(level, 0.0)]
    x = START_RUNWAY
    body_end = max(target_length - TAIL_RUNWAY, START_RUNWAY + 5.0)

    # Возврат гравитации в конце тоже занимает место — резервируем его заранее,
    # иначе уровень с переворотом в финале вылезает на два десятка тайлов.
    while x < body_end - (_GRAVITY_RETURN_RESERVE if ctx.gravity != 1 else 0.0):
        spec = _choose_spec(rng, difficulty, ctx, budget=body_end - x)
        accepted = False
        for attempt in range(SEGMENT_ATTEMPTS):
            base = len(level.objects)
            objs, x_end, new_ctx = spec.func(rng, difficulty, x, ctx)
            level.extend(objs)
            res = _advance(level, frontier, x_end)
            if res.reached:
                frontier = res.reached
                x = x_end
                ctx = replace(
                    new_ctx,
                    since_special=0 if spec.special else new_ctx.since_special + 1,
                )
                accepted = True
                break
            _truncate(level, base)
            _log.debug(
                "участок %s непроходим (попытка %d), x=%.1f", spec.name, attempt + 1, x
            )
        if accepted:
            continue
        # Ни одна попытка не прошла — ставим гарантированно безопасный отдых.
        _, x_end, ctx = pattern_rest(rng, difficulty, x, ctx)
        res = _advance(level, frontier, x_end)
        if not res.reached:
            _log.debug("фронт поиска умер на отдыхе при x=%.1f", x)
            return None
        frontier = res.reached
        x = x_end
        ctx = replace(ctx, since_special=ctx.since_special + 1)

    # Финиш обязан быть в обычной гравитации и в режиме куба: иначе «дойти до
    # конца» превращается в лотерею, а среда не сможет восстановить чекпойнт.
    if ctx.gravity != 1:
        objs, x_end, ctx = pattern_gravity_flip(rng, difficulty, x, ctx)
        level.extend(objs)
        res = _advance(level, frontier, x_end)
        if not res.reached:
            return None
        frontier = res.reached
        x = x_end

    goal_x = round(x + TAIL_RUNWAY, 2)
    level.length = goal_x
    level.add(LevelObject("goal", goal_x, CEILING_Y * 0.5))

    res = search_forward(
        level,
        frontier,
        None,
        max_nodes=_BUILD_MAX_NODES,
        max_frontier=_BUILD_MAX_FRONTIER,
    )
    if not res.finished:
        _log.debug("хвост уровня непроходим (goal_x=%.1f)", goal_x)
        return None

    level.checkpoints = make_checkpoints(level)
    return level


def _flat_level(name: str, length: float) -> Level:
    """Аварийный уровень: ровный пол и редкие одиночные шипы.

    Зачем: `generate_level` обязана вернуть проходимый уровень всегда, даже
    если случайность трижды подряд сложилась неудачно. Такой уровень проходим
    по построению — шипы стоят реже, чем длина прыжка.
    """
    objs = [LevelObject("spike", x, SPIKE_Y) for x in np.arange(12.0, length - 10.0, 12.0)]
    level = Level(
        name=name,
        length=float(length),
        objects=list(objs) + [LevelObject("goal", float(length), CEILING_Y * 0.5)],
        ceiling_y=CEILING_Y,
    )
    level.checkpoints = make_checkpoints(level)
    return level


def generate_level(
    difficulty: float,
    rng: np.random.Generator,
    name: str = "procedural",
    length: float | None = None,
) -> Level:
    """Сгенерировать проходимый уровень заданной сложности.

    `difficulty ∈ [0, 1]` управляет плотностью препятствий, шириной окон,
    длиной рядов шипов, наличием колец и падов, долей секций корабля и волны,
    сменами скорости и гравитации. При 0 — почти пустая дорожка с одиночными
    шипами, при 1 — плотный поток со сменами режима.

    Возвращённый уровень гарантированно проходим: каждый участок проверен
    поиском по кадрам во время сборки, а результат — целиком через
    `is_solvable`. Если случайность трижды подряд дала тупик, возвращается
    аварийный ровный уровень — но не непроходимый.

    `length` — ЦЕЛЕВАЯ длина: последний участок не режется посередине, поэтому
    итог может оказаться на несколько тайлов длиннее (резать участок значило
    бы получить обрубок, проходимость которого никто не проверял).
    """
    d = float(min(max(float(difficulty), 0.0), 1.0))
    target = float(length) if length is not None else _default_length(d)
    target = max(target, START_RUNWAY + TAIL_RUNWAY + 8.0)

    for attempt in range(LEVEL_ATTEMPTS):
        level = _build_once(d, rng, name, target)
        if level is None:
            continue
        if is_solvable(level, max_nodes=_FINAL_MAX_NODES, max_frontier=_BUILD_MAX_FRONTIER):
            return level
        _log.debug("уровень не прошёл финальную проверку (попытка %d)", attempt + 1)

    _log.warning(
        "не удалось собрать уровень сложности %.2f за %d попыток — отдаю ровный",
        d, LEVEL_ATTEMPTS,
    )
    return _flat_level(name, target)


def make_checkpoints(level: Level, every: float = 25.0) -> list[float]:
    """x-координаты practice-чекпойнтов примерно через каждые `every` тайлов.

    Зачем не просто арифметическая прогрессия: чекпойнт посреди поля шипов или
    внутри блока бесполезен — с него нельзя стартовать. Поэтому кандидат
    сдвигается к ближайшему «чистому» месту, где рядом нет ни опасности, ни
    блока, и игрок может спокойно стоять на полу.
    """
    result: list[float] = []
    if every <= 0.0:
        return result
    x = float(every)
    limit = level.length - 6.0
    while x < limit:
        safe = _safe_checkpoint_x(level, x, limit)
        if safe is not None and (not result or safe - result[-1] >= every * 0.5):
            result.append(round(safe, 3))
        x += every
    return result


def _safe_checkpoint_x(level: Level, x: float, limit: float) -> float | None:
    """Ближайшее к `x` место, где можно безопасно возродиться."""
    for offset in np.arange(0.0, 8.0, 0.5):
        for candidate in ((x - offset), (x + offset)) if offset else (x,):
            if candidate < 5.0 or candidate > limit:
                continue
            if _is_clear(level, float(candidate)):
                return float(candidate)
    return None


def _is_clear(level: Level, x: float) -> bool:
    """Нет ли рядом с `x` опасностей и блоков, мешающих старту с чекпойнта.

    Расстояние считается до КРАЯ объекта, а не до его центра: широкий блок,
    центр которого «достаточно далеко», всё равно может стоять стеной прямо
    перед возрождённым игроком.
    """
    for obj in level.objects_in_range(x - 4.0, x + 4.0):
        cls = obj.semantic_class()
        if cls not in (HAZARD, SOLID):
            continue
        distance = abs(obj.x - x) - obj.half_extent()[0]
        if distance < (3.0 if cls == HAZARD else 2.0):
            return False
    return True


__all__ = [
    "CEILING_Y",
    "PatternContext",
    "PatternSpec",
    "PATTERNS",
    "PATTERN_SPECS",
    "generate_level",
    "make_checkpoints",
    "is_solvable",
    "solve_actions",
    "search_forward",
    "state_key",
    "pattern_rest",
    "pattern_single_spike",
    "pattern_spike_row",
    "pattern_spike_stagger",
    "pattern_block_stairs",
    "pattern_platform_gap",
    "pattern_corridor",
    "pattern_jagged_floor",
    "pattern_saw",
    "pattern_orb_chain",
    "pattern_pad_jump",
    "pattern_speed_change",
    "pattern_gravity_flip",
    "pattern_ship_section",
    "pattern_wave_section",
]
