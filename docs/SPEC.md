# SPEC — GDAI: нейросеть, которая учится проходить Geometry Dash

> Этот файл — **обязательный контракт** для всех модулей проекта.
> Любой код в репозитории обязан соответствовать именам, сигнатурам и константам ниже.
> Если что-то не описано — выбирай простое решение и документируй его в docstring на русском.

---

## 0. Главная идея (почему это работает с декорациями)

Обычный пиксельный RL-агент выучивает **картинку**, а не **игру**. Стоит поменять тему
уровня, фон, свечение, партиклы, цвет блоков — и агент разваливается.

GDAI разделяет задачу на два независимых уровня:

```
  сырой кадр (любой дизайн)        каноническая карта             действие
   RGB 3x72x128 ───────────►  10 классов 72x128  ───────────►  {0: ничего, 1: держать}
        [ ЗРЕНИЕ: U-Net ]           [ ПОЛИТИКА: PPO ]
     учится supervised            учится в RL
     на синтетике с рандомизацией  на канонических картах
```

* **Зрение (perception)** переводит ЛЮБОЙ дизайн в одну и ту же каноническую
  семантическую карту: где пол, где шип, где игрок, где портал. Обучается
  с учителем, потому что симулятор одновременно рисует и «красивый» кадр,
  и идеальную разметку. Инвариантность к дизайну достигается **доменной
  рандомизацией**: случайные палитры, фоны, партиклы, свечение, текстуры,
  параллакс, тряска камеры + аугментации.
* **Политика (policy)** никогда не видит декораций — только каноническую карту.
  Поэтому она не может переобучиться на дизайн в принципе.

Побочный бонус: архитектура **интуитивно понятная** — можно буквально показать
пользователю «вот что видит нейросеть» (`python -m gdai watch`).

---

## 1. Раскладка проекта

```
gd/
├── README.md                 # RU, главный документ: идея, установка, команды, схемы
├── docs/SPEC.md              # этот файл
├── docs/ARCHITECTURE.md      # подробный разбор архитектуры + FAQ (RU)
├── requirements.txt
├── pyproject.toml
├── Makefile
├── levels/                   # готовые уровни *.json
├── gdai/
│   ├── __init__.py
│   ├── constants.py          # ФАЗА 1
│   ├── config.py             # ФАЗА 1
│   ├── env/
│   │   ├── __init__.py
│   │   ├── level.py          # ФАЗА 1  формат уровня, объекты, IO
│   │   ├── physics.py        # ФАЗА 1  физика куба/корабля/волны
│   │   ├── generator.py      # ФАЗА 1  процедурная генерация по сложности
│   │   ├── semantic.py       # ФАЗА 2  растеризация канонической карты
│   │   ├── gd_env.py         # ФАЗА 2  среда (gym-подобная)
│   │   ├── themes.py         # ФАЗА 3  темы/палитры/декорации
│   │   └── render.py         # ФАЗА 3  «красивый» рендер с рандомизацией
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── augment.py        # ФАЗА 3
│   │   ├── model.py          # ФАЗА 3  U-Net
│   │   ├── dataset.py        # ФАЗА 3  генерация пар (кадр, разметка)
│   │   └── train.py          # ФАЗА 3
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── networks.py       # ФАЗА 3  actor-critic
│   │   ├── buffer.py         # ФАЗА 3  rollout + GAE
│   │   ├── ppo.py            # ФАЗА 3  обучение
│   │   ├── curriculum.py     # ФАЗА 3  учебный план + practice-чекпойнты
│   │   └── vecenv.py         # ФАЗА 3  синхронный векторный враппер
│   ├── pipeline.py           # ФАЗА 4  зрение+политика вместе
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── viewer.py         # ФАЗА 4  окно «что видит ИИ»
│   │   ├── plots.py          # ФАЗА 4  графики обучения
│   │   └── saliency.py       # ФАЗА 4  карты внимания
│   ├── realgame/
│   │   ├── __init__.py
│   │   ├── capture.py        # ФАЗА 4  захват экрана настоящей GD (опционально)
│   │   └── play.py           # ФАЗА 4
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seeding.py, logging.py, checkpoint.py
│   └── cli.py                # ФАЗА 4  python -m gdai <команда>
└── tests/                    # pytest
```

Правило: **один файл — один владелец**. Не редактируй чужие файлы; если нужна
чужая функция — импортируй её по контракту из этого SPEC.

Язык: идентификаторы — английские, docstring и комментарии — **русские**.
Тип-аннотации обязательны для публичных функций.

---

## 2. `gdai/constants.py` — канонический словарь

```python
# --- семантические классы (порядок фиксирован, менять нельзя) ---
EMPTY          = 0   # пустота / фон / любая декорация
SOLID          = 1   # блок, платформа, пол — на него можно приземлиться
HAZARD         = 2   # шип, пила, любой мгновенно убивающий объект
PLAYER         = 3   # сам игрок
PAD            = 4   # жёлтый/розовый/красный трамплин (срабатывает сам)
ORB            = 5   # кольцо (срабатывает по нажатию)
PORTAL_GRAVITY = 6   # портал смены гравитации
PORTAL_MODE    = 7   # портал смены режима (куб/корабль/волна)
PORTAL_SPEED   = 8   # портал смены скорости
GOAL           = 9   # финиш

NUM_CLASSES = 10
CLASS_NAMES: tuple[str, ...]     # ("empty","solid","hazard",...)
CLASS_COLORS: tuple[tuple[int,int,int], ...]   # RGB для визуализации карты, len == NUM_CLASSES

# --- геометрия мира ---
TILE = 1.0            # 1 тайл мира == 1 блок GD (в оригинале 30 px)
VIEW_TILES_W = 16     # ширина камеры в тайлах
VIEW_TILES_H = 9      # высота камеры в тайлах
PX_PER_TILE  = 8      # пикселей на тайл в наблюдении
OBS_W = VIEW_TILES_W * PX_PER_TILE   # 128
OBS_H = VIEW_TILES_H * PX_PER_TILE   # 72
PLAYER_X_IN_VIEW = 4.0   # игрок стоит на 4-м тайле от левого края камеры
GROUND_Y = 0.0           # уровень пола (низ игрока при y=0)

# --- физика (тайлы и секунды), dt = 1/60 ---
DT = 1.0 / 60.0
GRAVITY      = 76.8    # тайл/с^2, куб
JUMP_V       = 19.2    # тайл/с, старт прыжка (высота ~2.4 тайла, полёт ~0.5 с)
MAX_FALL_V   = 30.0
SPEEDS       = (0.5, 1.0, 2.0, 3.0, 4.0)          # индексы 0..4
SPEED_TILES_PER_SEC = (8.36, 10.386, 12.914, 15.6, 19.2)
DEFAULT_SPEED_INDEX = 1

PAD_YELLOW_V = 24.0
PAD_PINK_V   = 15.0
PAD_RED_V    = 30.0
ORB_YELLOW_V = 19.2
ORB_PINK_V   = 13.0
ORB_RED_V    = 26.0

SHIP_THRUST  = 52.0    # тайл/с^2 вверх при удержании
SHIP_GRAVITY = 52.0
SHIP_MAX_V   = 16.0
WAVE_SPEED_RATIO = 1.0   # волна движется по диагонали 45°

# --- хитбоксы (полуразмеры относительно центра) ---
PLAYER_HALF        = 0.45   # куб 0.9x0.9
PLAYER_HALF_SHIP   = 0.45
PLAYER_HALF_WAVE   = 0.25
HAZARD_HALF        = 0.28   # хитбокс шипа сильно меньше картинки — как в GD
ORB_HALF           = 0.60   # кольцо ловит щедро
PAD_HALF_X, PAD_HALF_Y = 0.5, 0.25
PORTAL_HALF_X, PORTAL_HALF_Y = 0.35, 1.25

# --- действия ---
ACTION_NONE = 0
ACTION_HOLD = 1
NUM_ACTIONS = 2
```

---

## 3. `gdai/config.py`

Только `@dataclass`, никакой логики. Все поля с дефолтами и русскими комментариями.

```python
@dataclass
class EnvConfig:
    obs_mode: str = "semantic"      # "semantic" | "pixels" | "both"
    max_steps: int = 6000
    difficulty: float = 0.3         # 0..1, для процедурной генерации
    level_path: str | None = None   # если задан — фиксированный уровень из файла
    seed: int | None = None
    randomize_theme: bool = True    # менять тему на каждом эпизоде (для obs_mode с pixels)
    decoration_level: float = 1.0   # 0 = голый уровень, 1 = максимум декора
    practice_checkpoints: bool = True
    checkpoint_prob: float = 0.5    # вероятность старта с чекпойнта
    semantic_noise: float = 0.0     # вероятность порчи пикселя карты (робастность политики)
    reward_progress: float = 1.0
    reward_death: float = -1.0
    reward_finish: float = 10.0
    reward_alive: float = 0.0

@dataclass
class PerceptionConfig:
    base_channels: int = 24
    depth: int = 3
    steps: int = 4000
    batch_size: int = 16
    lr: float = 3e-4
    val_every: int = 250
    device: str = "auto"
    out_dir: str = "runs/perception"
    augment: bool = True

@dataclass
class PPOConfig:
    num_envs: int = 8
    rollout_steps: int = 256
    total_steps: int = 2_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    out_dir: str = "runs/agent"

@dataclass
class CurriculumConfig:
    start_difficulty: float = 0.05
    max_difficulty: float = 1.0
    step: float = 0.05
    promote_success_rate: float = 0.7   # доля пройденных уровней для повышения
    window: int = 50
```

---

## 4. `gdai/env/level.py` — формат уровня

```python
OBJECT_TYPES = (
  "block", "platform",            # SOLID
  "spike", "spike_down", "spike_left", "spike_right", "saw",   # HAZARD
  "pad_yellow", "pad_pink", "pad_red",                          # PAD
  "orb_yellow", "orb_pink", "orb_red",                          # ORB
  "portal_gravity_down", "portal_gravity_up",                   # PORTAL_GRAVITY
  "portal_cube", "portal_ship", "portal_wave",                  # PORTAL_MODE
  "portal_speed_0", ... "portal_speed_4",                       # PORTAL_SPEED
  "goal",                                                       # GOAL
)

@dataclass
class LevelObject:
    type: str
    x: float          # центр объекта в тайлах
    y: float          # центр объекта в тайлах, y растёт вверх, пол при y=0
    def semantic_class(self) -> int: ...
    def half_extent(self) -> tuple[float, float]:  # для хитбокса и растеризации

@dataclass
class Level:
    name: str
    length: float                     # длина в тайлах (x финиша)
    objects: list[LevelObject]
    start_mode: str = "cube"          # cube|ship|wave
    start_speed_index: int = 1
    start_gravity: int = 1            # 1 вниз, -1 вверх
    ceiling_y: float = 12.0
    theme_hint: str | None = None
    checkpoints: list[float] = field(default_factory=list)  # x-координаты для practice

    def to_dict(self)/from_dict(cls,d)/save(path)/ classmethod load(path)
    def objects_in_range(self, x0: float, x1: float) -> list[LevelObject]   # быстрый bucket-индекс
```

Хранение JSON — человекочитаемое, ключи как в `to_dict`. Версия формата `"version": 1`.
`objects_in_range` обязан быть O(k) — построй бакеты по int(x) при создании уровня
(перестраивать при изменении `objects` через метод `rebuild_index()`).

---

## 5. `gdai/env/physics.py` — состояние и шаг физики

```python
@dataclass
class PlayerState:
    x: float; y: float           # центр игрока
    vy: float
    mode: str                    # "cube"|"ship"|"wave"
    gravity: int                 # +1 вниз, -1 вверх
    speed_index: int
    on_ground: bool
    alive: bool
    finished: bool
    hold_prev: bool              # было ли удержание на прошлом кадре (для колец)

def step_physics(state: PlayerState, level: Level, hold: bool,
                 dt: float = DT) -> tuple[PlayerState, dict]:
    """Один кадр физики. Возвращает новое состояние и словарь событий
    {"died": bool, "finished": bool, "jumped": bool, "used_orb": bool,
     "used_pad": bool, "portal": str|None}. Функция ЧИСТАЯ — не мутирует вход."""
```

Порядок внутри шага (важен, воспроизводит поведение GD):
1. Горизонтальное движение `x += SPEED_TILES_PER_SEC[speed_index] * dt`.
   Проверка бокового столкновения с SOLID → смерть.
2. Порталы/пады/кольца по пересечению хитбоксов (кольцо — только по фронту нажатия
   `hold and not hold_prev`, пад — автоматически).
3. Вертикаль:
   - cube: если `on_ground and hold` → `vy = JUMP_V * gravity_sign_up`; иначе `vy -= GRAVITY*gravity*dt`.
     Куб при удержании прыгает снова сразу после приземления (как в оригинале).
   - ship: `vy += (SHIP_THRUST if hold else -SHIP_GRAVITY) * ... * gravity`, клип по `SHIP_MAX_V`.
   - wave: `vy = ±speed * WAVE_SPEED_RATIO` мгновенно, без инерции.
4. `y += vy*dt`, разрешение столкновений с SOLID: приземление сверху (по направлению
   гравитации) → snap + `on_ground=True`; удар «в лоб» → смерть; для ship/wave удар
   в любую сторону → смерть.
5. Пол `GROUND_Y` и потолок `ceiling_y` — твёрдые; в режиме wave касание = смерть,
   в cube/ship — приземление.
6. HAZARD пересечение → смерть. `x >= level.length` → `finished=True`.

Функция обязана быть детерминированной и быстрой (чистый Python/numpy, без pygame).

---

## 6. `gdai/env/generator.py`

```python
def generate_level(difficulty: float, rng: np.random.Generator,
                   name: str = "procedural", length: float | None = None) -> Level
```

`difficulty ∈ [0,1]` управляет: плотностью препятствий, шириной окон для прыжка,
числом подряд идущих шипов, наличием кольца/пада-цепочек, долей ship/wave секций,
сменами скорости и гравитации. При `difficulty=0` — почти пустой уровень с
одиночными шипами; при `1.0` — плотный поток.

**Критично:** генератор обязан выдавать **проходимые** уровни. Строй уровень
паттернами (`PATTERNS` — список функций «участок»), каждый паттерн знает, как его
проходить. Обязательно есть функция

```python
def is_solvable(level: Level, max_frames: int = 20000) -> bool
```
— поиск в ширину/greedy по кадрам (действие ∈ {0,1}) с ограничением по времени;
используется в тестах и внутри генератора (перегенерировать участок, если непроходим).
Дедупликация состояний: округляй (x до 0.25 тайла, y до 0.25, vy до 0.5, on_ground, mode, gravity).

Также: `def make_checkpoints(level: Level, every: float = 25.0) -> list[float]`.

---

## 7. `gdai/env/semantic.py` — каноническая карта (это ground truth)

```python
def render_semantic(level: Level, state: PlayerState,
                    view_w: int = OBS_W, view_h: int = OBS_H) -> np.ndarray:
    """uint8 (view_h, view_w) с классами 0..9. Камера: игрок на PLAYER_X_IN_VIEW,
    ось Y перевёрнута (мир вверх = меньший индекс строки)."""

def camera_origin(state: PlayerState) -> tuple[float, float]:
    """Левый-нижний угол камеры в мировых координатах. Камера следует за x игрока,
    по Y — плавно центрируется, но всегда показывает пол, если он близко."""

def world_to_pixel(wx, wy, cam) -> tuple[int, int]
def semantic_to_rgb(sem: np.ndarray) -> np.ndarray   # (H,W,3) uint8 по CLASS_COLORS
def downsample_semantic(sem: np.ndarray, factor: int = 2) -> np.ndarray  # приоритет опасных классов
```

Приоритет отрисовки (что поверх чего): EMPTY < SOLID < PORTAL_* < PAD < ORB < GOAL < HAZARD < PLAYER.
Шипы рисуются треугольником с учётом ориентации; пилы — кругом; порталы — вертикальным овалом.

Растеризация — numpy, без pygame, быстрая (векторно по прямоугольникам).

---

## 8. `gdai/env/gd_env.py` — среда

```python
class GeometryDashEnv:
    metadata = {"render_modes": ["rgb_array", "human"]}
    action_space_n = NUM_ACTIONS

    def __init__(self, config: EnvConfig | None = None, renderer=None): ...
    def reset(self, *, seed: int | None = None,
              level: Level | None = None,
              start_x: float | None = None) -> tuple[dict, dict]:
        """Возвращает (obs, info). obs — dict:
           "semantic": uint8 (OBS_H, OBS_W)          если obs_mode in ("semantic","both")
           "pixels":   uint8 (OBS_H, OBS_W, 3)       если obs_mode in ("pixels","both")
           "features": float32 (FEATURE_DIM,)        всегда
        """
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """(obs, reward, terminated, truncated, info).
        info: {"x": float, "progress": 0..1, "died": bool, "finished": bool,
               "level_name": str, "difficulty": float}"""
    def render(self, mode="rgb_array") -> np.ndarray   # всегда «красивый» кадр
    def set_difficulty(self, d: float) -> None
    def close(self) -> None
```

`FEATURE_DIM = 8`: `[vy_norm, on_ground, gravity, mode_is_cube, mode_is_ship,
mode_is_wave, speed_norm, progress]`.

Награда: `reward_progress * dx_в_тайлах / 10` каждый кадр (dx — пройденное за кадр),
`+ reward_alive`, при смерти `+ reward_death`, при финише `+ reward_finish`.
Terminated при смерти/финише, truncated при `max_steps`.

Practice-чекпойнты: если `config.practice_checkpoints` и агент умер дальше первого
чекпойнта, то с вероятностью `checkpoint_prob` следующий `reset` стартует с
последнего пройденного чекпойнта (восстанавливая mode/gravity/speed на этот x —
для этого симулируй уровень до чекпойнта или храни снимок состояния при проходе).

`semantic_noise > 0` → случайно портить пиксели карты (симуляция ошибок зрения).

---

## 9. `gdai/env/themes.py` + `gdai/env/render.py` — доменная рандомизация

Это **сердце устойчивости к декорациям**. Рендерер обязан уметь генерировать
визуально ОЧЕНЬ разные кадры для одного и того же уровня.

```python
@dataclass
class Theme:
    name: str
    bg_top: RGB; bg_bottom: RGB          # градиент фона
    block_fill: RGB; block_edge: RGB
    hazard_fill: RGB; hazard_edge: RGB
    player_fill: RGB; player_edge: RGB
    ground_fill: RGB; ground_line: RGB
    glow: float            # 0..1 сила свечения
    block_style: str       # "flat"|"outline"|"bevel"|"striped"|"dotted"|"gradient"|"noise"
    hazard_style: str      # "solid"|"outline"|"gradient"|"double"
    bg_style: str          # "plain"|"grid"|"stars"|"stripes"|"circles"|"clouds"|"noise"
    parallax_layers: int   # 0..3
    particles: float       # 0..1 плотность партиклов
    pulse: float           # 0..1 «биение» под музыку

BUILTIN_THEMES: tuple[Theme, ...]        # >= 10 штук, стилистика реальных уровней GD
def random_theme(rng) -> Theme           # полностью случайная тема (все поля рандом)
def theme_by_name(name: str) -> Theme
```

```python
class Renderer:
    def __init__(self, width=OBS_W, height=OBS_H, theme: Theme | None = None,
                 decoration_level: float = 1.0, seed: int | None = None): ...
    def set_theme(self, theme: Theme) -> None
    def randomize(self, rng) -> None     # новая тема + новые декорации
    def render(self, level: Level, state: PlayerState, t: int) -> np.ndarray  # (H,W,3) uint8
```

Обязательные источники вариативности (все включаются `decoration_level`):
1. случайные палитры (в т.ч. такие, где блок и шип похожи по цвету — чтобы сеть
   училась на форму, а не на цвет);
2. фоновые узоры + 0..3 слоя параллакса с случайными фигурами;
3. декоративные (неигровые!) объекты: полосы, «трубы», плавающие фигуры, глиф-паттерны —
   они НЕ попадают в семантическую карту, сеть обязана научиться их игнорировать;
4. партиклы/следы за игроком, вспышки, «биение» яркости;
5. свечение/bloom, виньетка, лёгкая тряска камеры (не более 1 px, чтобы не ломать разметку),
   шум, изменение контраста/гаммы;
6. случайная толщина обводки, скругления, текстуры блоков;
7. иногда — инвертированная цветовая схема или монохром.

**Жёсткое требование:** декорации не имеют права смещать игровые объекты. Рендер и
`render_semantic` обязаны использовать одну и ту же камеру (`camera_origin`),
чтобы пиксель кадра и пиксель разметки соответствовали друг другу.
Тряска камеры применяется К ОБОИМ или ни к одному (проще — не применять к геометрии,
а делать её пост-эффектом сдвига обоих массивов; но по умолчанию shake=0 для датасета).

Рендер через pygame с `SDL_VIDEODRIVER=dummy` (headless). Модуль обязан выставлять
переменную окружения ДО `import pygame`, если дисплея нет.

---

## 10. `gdai/perception/` — зрение

`model.py`:
```python
class PerceptionNet(nn.Module):
    """Маленький U-Net: вход (B,3,72,128) float32 в [0,1], выход (B,10,72,128) логиты."""
    def __init__(self, num_classes=NUM_CLASSES, base_channels=24, depth=3): ...
    def forward(self, x) -> Tensor
    @torch.no_grad()
    def predict(self, frame_uint8: np.ndarray) -> np.ndarray   # (H,W,3)->(H,W) uint8
    @torch.no_grad()
    def predict_batch(self, frames_uint8: np.ndarray) -> np.ndarray
```
Требования: < 500k параметров, работает на CPU быстрее 100 кадров/с батчами.
Instance/GroupNorm (не BatchNorm) — устойчивее к смене домена при батче 1.

`dataset.py`:
```python
class SyntheticSegDataset(torch.utils.data.Dataset | IterableDataset):
    """Бесконечный поток пар (кадр с рандомным дизайном, каноническая карта).
    Каждый сэмпл: случайный уровень (или из levels/), случайная тема, случайное
    состояние игрока (в т.ч. в воздухе, разные режимы/гравитация)."""
def make_loaders(cfg: PerceptionConfig) -> tuple[DataLoader, DataLoader]
```
Валидация — на **отложенных темах**, которых не было в обучении (held-out themes),
чтобы честно измерять обобщение на новый дизайн. Функция должна это гарантировать.

`augment.py`: цветовой джиттер, гамма, шум, блюр, JPEG-подобные артефакты, cutout,
случайная яркость/контраст, лёгкий сдвиг цветовых каналов. Аугментации применяются
ТОЛЬКО к кадру, не к разметке (кроме геометрических — их не используем вовсе).

`train.py`:
```python
def train_perception(cfg: PerceptionConfig) -> dict   # метрики
```
Лосс: CrossEntropy с весами классов (редкие HAZARD/ORB/PORTAL важнее) + Dice.
Метрики: pixel accuracy, mIoU, **IoU по HAZARD и SOLID отдельно** (они решают всё).
Сохранять `best.pt` (+ `last.pt`) c `{"model_state": ..., "config": asdict(cfg)}`.
Логи в `out_dir/metrics.jsonl`. Поддержать `--steps` для быстрых прогонов.

---

## 11. `gdai/agent/` — политика

`networks.py`:
```python
class ActorCritic(nn.Module):
    """Вход: semantic one-hot (B,10,36,64) + features (B,8). Выход: logits (B,2), value (B,)."""
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=8, hidden=256): ...
    def forward(self, sem, feat) -> tuple[Tensor, Tensor]
    def act(self, sem, feat, deterministic=False) -> tuple[action, logprob, value]
    def evaluate_actions(self, sem, feat, actions) -> tuple[logprob, entropy, value]
def semantic_to_tensor(sem_uint8: np.ndarray) -> Tensor   # one-hot + downsample x2
```

`buffer.py`: `RolloutBuffer` с GAE(λ), нормализацией преимуществ, итератором минибатчей.

`vecenv.py`: `SyncVectorEnv` — список сред, авто-reset, единый батч наблюдений.

`curriculum.py`:
```python
class Curriculum:
    def __init__(self, cfg: CurriculumConfig): ...
    def current_difficulty(self) -> float
    def record_episode(self, finished: bool) -> None
    def maybe_promote(self) -> bool     # True если сложность повысилась
    def state_dict(self)/load_state_dict(...)
```

`ppo.py`:
```python
def train_agent(cfg: PPOConfig, env_cfg: EnvConfig,
                curriculum_cfg: CurriculumConfig | None = None,
                on_iteration: Callable | None = None) -> dict
```
Классический PPO: clipped surrogate, value clipping, entropy bonus, ортогональная
инициализация, annealing lr. Логи в `out_dir/metrics.jsonl` (шаги, reward, длина эпизода,
доля прохождений, сложность, KL, entropy). Чекпойнты `best.pt`/`last.pt`.
Должен уметь честно отработать `total_steps=2000` за секунды (для smoke-теста).

---

## 12. `gdai/pipeline.py` — всё вместе

```python
class GDAgent:
    """Полный агент: пиксели -> зрение -> политика -> действие.
    Работает и с ground-truth картой (быстро), и с предсказанной (честно)."""
    def __init__(self, policy_path: str | None = None,
                 perception_path: str | None = None,
                 device: str = "auto", use_perception: bool = True): ...
    def see(self, frame_rgb: np.ndarray) -> np.ndarray     # (H,W,3)->(H,W) классы
    def act(self, obs: dict, deterministic: bool = True) -> int
    def reset(self) -> None
def evaluate(agent, env, episodes=20, use_perception=False) -> dict
    # {"success_rate":..,"mean_progress":..,"mean_reward":..,"mean_len":..}
```

---

## 13. `gdai/viz/` — интуитивность

`viewer.py`: окно pygame 3 панели:
`[красивый кадр] | [что видит нейросеть (предсказанная карта)] | [эталон + разница]`,
снизу — полоса прогресса уровня, вероятность действия «держать», value, FPS.
Управление: `Space` — играть самому, `A` — отдать управление ИИ, `T` — сменить тему,
`D` — уровень декора, `P` — вкл/выкл зрение (ground truth vs предсказание),
`R` — рестарт, `N` — новый уровень, `Esc` — выход.
Обязательно работать и в headless (`--record out.mp4|out.gif` или запись PNG-кадров).

`plots.py`: графики из `metrics.jsonl` → PNG (matplotlib, без интерактива).
`saliency.py`: градиент выхода политики по входной карте → тепловая карта «на что смотрит ИИ».

---

## 14. `gdai/realgame/` — опционально, настоящая Geometry Dash

`capture.py`: захват окна через `mss` (мягкий импорт, понятная ошибка если нет),
калибровка: пользователь указывает прямоугольник игрового поля, кадр ресайзится в
`OBS_W x OBS_H`. `play.py`: цикл захват → `GDAgent.see` → `act` → нажатие пробела
(`pynput`, мягкий импорт). Обязательно: предупреждение в docstring/README, что модуль
требует ручной калибровки и не является основным сценарием.

---

## 15. `gdai/cli.py`

```
python -m gdai selfcheck                 # быстрая проверка всей связки (< 60 c)
python -m gdai play                      # играть самому (pygame окно)
python -m gdai watch [--policy ...] [--perception ...]   # смотреть, что видит ИИ
python -m gdai gen-level --difficulty 0.4 --out levels/x.json
python -m gdai train-perception [--steps N] [--out runs/perception]
python -m gdai train-agent [--total-steps N] [--curriculum] [--out runs/agent]
python -m gdai eval --policy runs/agent/best.pt [--perception runs/perception/best.pt]
python -m gdai plot --run runs/agent --out curves.png
python -m gdai play-real --policy ... --perception ...   # настоящая игра
python -m gdai demo                      # запись гифки/видео демонстрации
```
`argparse` с подкомандами, русский `help` у каждой. Точка входа `gdai/__main__.py`.

---

## 16. Тесты (`tests/`, pytest)

Обязательный минимум:
* `test_physics.py` — высота прыжка ≈ 2.4 тайла, детерминизм, смерть о шип, пад/кольцо,
  смена гравитации, чистота функции (вход не мутируется).
* `test_level.py` — round-trip JSON, `objects_in_range`, индекс.
* `test_generator.py` — 20 уровней разной сложности проходимы (`is_solvable`).
* `test_semantic.py` — игрок всегда присутствует на карте; классы в диапазоне;
  приоритет HAZARD над SOLID; размер массива.
* `test_env.py` — reset/step контракт, формы наблюдений, награда за прогресс,
  детерминизм по seed, practice-чекпойнты.
* `test_render_invariance.py` — **ключевой**: для 30 случайных тем семантическая
  карта одного и того же состояния БАЙТ-В-БАЙТ одинакова, а кадры различаются
  (средняя попиксельная разница выше порога).
* `test_perception.py` — forward/предсказание, число параметров < 500k,
  переобучение на 1 сэмпле за 200 шагов даёт accuracy > 0.95.
* `test_ppo.py` — smoke: 2000 шагов обучения не падают, лосс конечен, чекпойнт грузится.
* `test_pipeline.py` — `GDAgent` работает без обученных весов (случайная инициализация).

Все тесты обязаны проходить меньше чем за ~3 минуты суммарно на CPU.

---

## 17. Инженерные правила

* Никаких зависимостей кроме: `numpy`, `torch`, `pygame`, `matplotlib`, `imageio` (опц.),
  `opencv-python-headless` (опц.), `pytest` (dev), `mss`/`pynput` (опц., только realgame).
  Все опциональные — мягкий импорт с понятным сообщением.
* Никакого `from x import *`. Явные `__all__` в пакетах.
* Все случайности через `np.random.Generator` / `torch.Generator` — воспроизводимость по seed.
* `device="auto"` → cuda если есть, иначе cpu.
* Код должен работать на Python 3.10+.
* Никаких `print` в библиотеке — только через `gdai/utils/logging.py` (`get_logger`).
* Публичные функции — с русским docstring, объясняющим «зачем», а не «что».
