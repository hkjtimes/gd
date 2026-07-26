"""Аугментации кадра — вторая линия обороны против «выучивания дизайна».

Зачем они нужны, если есть темы
-------------------------------
`themes.py` меняет ЗАМЫСЕЛ картинки: палитру, узоры, декорации. Аугментации
меняют КАНАЛ, по которому картинка дошла до сети: яркость монитора, гамму
записи, шум сенсора, сжатие видео, замыленность апскейла. Это разные оси
вариативности, и обе нужны: без тем сеть выучит «шип — красный», без
аугментаций — «шип — ровно такой градиент из pygame». Особенно это важно для
`realgame`, где кадр приходит с настоящего экрана через захват и JPEG-подобное
сжатие, а не из нашего рендера.

Жёсткое правило (SPEC §10)
--------------------------
Аугментации применяются ТОЛЬКО к кадру. Разметка — эталон, её трогать нельзя.
Поэтому здесь НЕТ ни поворотов, ни сдвигов, ни отражений, ни масштабирования:
любая такая операция обязана была бы синхронно сдвинуть карту, а это лишний
источник рассинхрона (и заодно ложь: игрок в Geometry Dash всегда в одной
точке кадра, поворот кадра — невозможное для политики событие).
Единственное исключение — сдвиг ОТДЕЛЬНЫХ цветовых каналов на 1 px: это не
геометрия сцены, а модель хроматической аберрации объектива, и она сама по
себе учит сеть не доверять одному каналу.

Все операции работают с float32 (H,W,3) в [0,1] и возвращают его же; публичная
`augment_frame` принимает и отдаёт uint8, как и остальной конвейер кадров.
Вся случайность — через явно переданный `np.random.Generator`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

# Размер блока JPEG-подобного сжатия. Восемь — как в настоящем JPEG; на кадре
# 128x72 это ровно 16x9 блоков, поэтому артефакты ложатся без padding'а.
JPEG_BLOCK: int = 8

# Базовая таблица квантования JPEG (яркостная, качество ~50). Используется как
# форма распределения «что теряется первым»: высокие частоты (правый нижний
# угол) грубеют раньше низких — именно это даёт узнаваемый «мыльный квадратик».
_JPEG_Q: np.ndarray = np.asarray(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def _dct_matrix(n: int = JPEG_BLOCK) -> np.ndarray:
    """Ортонормированная матрица DCT-II размера n x n.

    Зачем своя, а не из scipy/cv2: обе библиотеки — необязательные зависимости,
    а матрица 8x8 считается один раз за импорт и стоит микросекунды.
    """
    k = np.arange(n, dtype=np.float64)
    m = np.cos(np.pi * (2.0 * k[None, :] + 1.0) * k[:, None] / (2.0 * n))
    m *= np.sqrt(2.0 / n)
    m[0, :] *= np.sqrt(0.5)
    return m.astype(np.float32)


_DCT: np.ndarray = _dct_matrix()


@dataclass(frozen=True)
class AugmentConfig:
    """Вероятности и силы отдельных аугментаций.

    Значения подобраны так, чтобы кадр оставался «читаемым человеком»: цель —
    расширить домен, а не превратить обучение в угадывание по шуму. Каждое
    поле `p_*` — вероятность применить операцию к конкретному кадру.
    """

    p_brightness: float = 0.8      # яркость/контраст — самый частый сдвиг домена
    brightness: float = 0.25       # ±доля от диапазона
    contrast: float = 0.45         # ±доля вокруг 1.0

    p_gamma: float = 0.5
    gamma_range: tuple[float, float] = (0.6, 1.7)

    p_color: float = 0.7           # независимые коэффициенты по каналам
    color_gain: float = 0.28
    p_desaturate: float = 0.12     # иногда цвет вообще не несёт информации
    p_channel_shift: float = 0.25  # хроматическая аберрация на 1 px
    # Доля пикселя, на которую реально разъезжаются каналы. Верхняя граница
    # намеренно меньше единицы: на одноцветной теме сдвиг канала «на целый
    # пиксель» равен сдвигу всей картинки относительно разметки.
    channel_shift_amount: tuple[float, float] = (0.2, 0.7)
    p_channel_swap: float = 0.08   # перестановка каналов: «другой профиль цвета»

    p_noise: float = 0.6
    noise_sigma: float = 0.06      # гауссов шум сенсора
    p_salt: float = 0.15
    salt_amount: float = 0.01      # доля «битых» пикселей

    p_blur: float = 0.35           # мягкость апскейла/расфокус
    blur_strength: float = 1.0     # 0..1, доля смешивания с размытием

    p_jpeg: float = 0.35
    jpeg_quality: tuple[float, float] = (0.3, 1.0)  # 1 — почти без потерь

    p_posterize: float = 0.12
    posterize_levels: tuple[int, int] = (5, 24)

    p_cutout: float = 0.25         # окклюзия: часть кадра закрыта «интерфейсом»
    cutout_boxes: tuple[int, int] = (1, 3)
    cutout_size: tuple[float, float] = (0.05, 0.22)   # доля стороны кадра

    def scaled(self, factor: float) -> "AugmentConfig":
        """Ослабить/усилить все вероятности разом (0 — выключить аугментации).

        Зачем: удобно для расписания — начинать обучение на почти чистых кадрах
        и наращивать сложность, а на валидации ставить 0 одним вызовом.
        """
        f = float(max(0.0, factor))
        changes = {
            fld.name: float(min(1.0, getattr(self, fld.name) * f))
            for fld in fields(self)
            if fld.name.startswith("p_")
        }
        return replace(self, **changes)


DEFAULT_AUGMENT: AugmentConfig = AugmentConfig()


# ---------------------------------------------------------------------------
# отдельные операции (все — над float32 (H,W,3) в [0,1])
# ---------------------------------------------------------------------------
def _as_float(frame: np.ndarray) -> np.ndarray:
    """Привести кадр к float32 (H,W,3) в [0,1], не портя оригинал."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Ожидался кадр (H,W,3), получено {arr.shape}")
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    return np.clip(arr.astype(np.float32, copy=True), 0.0, 1.0)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Обратное преобразование float [0,1] -> uint8 с честным округлением."""
    return np.clip(np.asarray(frame) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def brightness_contrast(
    frame: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    """Сдвиг яркости и растяжение контраста вокруг средне-серого.

    Контраст крутится вокруг 0.5, а не вокруг среднего кадра: среднее зависит
    от темы (тёмный неон против светлого пастельного), и «нормировка по кадру»
    незаметно убрала бы часть той самой вариативности, ради которой всё
    затевалось.
    """
    return np.clip((frame - 0.5) * float(contrast) + 0.5 + float(brightness), 0.0, 1.0)


def adjust_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Гамма-коррекция: имитация другого монитора/кодека записи."""
    g = float(max(0.05, gamma))
    return np.power(np.clip(frame, 0.0, 1.0), g, dtype=np.float32)


def color_jitter(
    frame: np.ndarray, gains: tuple[float, float, float]
) -> np.ndarray:
    """Независимые коэффициенты по R/G/B — «другой баланс белого»."""
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(frame * g, 0.0, 1.0)


def desaturate(frame: np.ndarray, amount: float) -> np.ndarray:
    """Подмешать серое: `amount=1` — полный монохром.

    Зачем в аугментациях, если есть монохромные темы: тема обесцвечивает
    ПАЛИТРУ (объекты всё ещё разной яркости по замыслу), а это — обесцвечивание
    готового кадра вместе со свечением, партиклами и пост-эффектами.
    """
    gray = (frame @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    a = float(np.clip(amount, 0.0, 1.0))
    return frame * (1.0 - a) + gray * a


def _shift_edge(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Сдвинуть плоскость на целые пиксели с ПОВТОРОМ края (без заворота).

    Зачем не `np.roll`: заворот приносит к левому краю содержимое правого — на
    кадре 128 px это кусок мира, отстоящий на 16 тайлов, то есть чистая ложь про
    геометрию прямо на границе, где зрению и так тяжелее всего.
    """
    out = plane
    if dy:
        out = np.roll(out, dy, axis=0)
        if dy > 0:
            out[:dy, :] = out[dy:dy + 1, :]
        else:
            out[dy:, :] = out[dy - 1:dy, :]
    if dx:
        out = np.roll(out, dx, axis=1)
        if dx > 0:
            out[:, :dx] = out[:, dx:dx + 1]
        else:
            out[:, dx:] = out[:, dx - 1:dx]
    return out


def channel_shift(
    frame: np.ndarray, dx: int, dy: int, amount: float = 1.0
) -> np.ndarray:
    """Сдвинуть красный и синий каналы на ±1 px в разные стороны.

    Зелёный (несущий большую часть яркости) остаётся на месте — иначе сдвинулся
    бы весь кадр относительно разметки. Так модель хроматической аберрации не
    ломает соответствие «пиксель кадра — пиксель карты».

    `amount < 1` смешивает сдвинутый канал с исходным, то есть даёт сдвиг на
    ДОЛЮ пикселя. Это не косметика: на красной или синей теме зелёный канал
    почти пуст, вся видимая структура живёт в сдвигаемых каналах, и целый
    пиксель сдвига становится сдвигом всей картинки относительно разметки —
    1/8 тайла. Частичное смешивание удерживает смещение заметно ниже пикселя
    при том же эффекте «дешёвой оптики».
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0 or (dx == 0 and dy == 0):
        return frame
    out = frame.copy()
    red = _shift_edge(frame[..., 0].copy(), dy, dx)
    blue = _shift_edge(frame[..., 2].copy(), -dy, -dx)
    out[..., 0] = frame[..., 0] * (1.0 - a) + red * a
    out[..., 2] = frame[..., 2] * (1.0 - a) + blue * a
    return out


def add_noise(frame: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Гауссов шум — модель сенсора/битрейта видео."""
    if sigma <= 0.0:
        return frame
    noise = rng.normal(0.0, float(sigma), size=frame.shape).astype(np.float32)
    return np.clip(frame + noise, 0.0, 1.0)


def salt_pepper(frame: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Битые пиксели: доля `amount` точек становится чёрной или белой."""
    if amount <= 0.0:
        return frame
    h, w = frame.shape[:2]
    count = int(h * w * float(amount))
    if count <= 0:
        return frame
    ys = rng.integers(0, h, size=count)
    xs = rng.integers(0, w, size=count)
    vals = rng.integers(0, 2, size=count).astype(np.float32)
    out = frame.copy()
    out[ys, xs, :] = vals[:, None]
    return out


def blur(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Мягкое размытие 3x3 (сепарабельное) с плавной силой.

    Зачем сепарабельно и вручную: свёртка 1x3 + 3x1 на numpy стоит копейки, а
    тянуть ради этого cv2 (необязательная зависимость) в горячий цикл датасета
    не хочется.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return frame
    pad = np.pad(frame, ((1, 1), (1, 1), (0, 0)), mode="edge")
    horiz = (pad[:, :-2] + 2.0 * pad[:, 1:-1] + pad[:, 2:]) * 0.25
    soft = (horiz[:-2] + 2.0 * horiz[1:-1] + horiz[2:]) * 0.25
    return frame * (1.0 - s) + soft * s


def jpeg_artifacts(frame: np.ndarray, quality: float) -> np.ndarray:
    """Блочные артефакты сжатия: DCT 8x8 -> квантование -> обратно.

    Зачем честный DCT, а не «размытие блоками»: настоящий JPEG портит именно
    высокие частоты, то есть в первую очередь ТОНКИЕ КОНТУРЫ — шипы и кромки
    блоков. Модель, пережившая это, не рассыпется на записи с YouTube или на
    захвате экрана с аппаратным кодеком.
    """
    q = float(np.clip(quality, 0.05, 1.0))
    h, w = frame.shape[:2]
    ph = (-h) % JPEG_BLOCK
    pw = (-w) % JPEG_BLOCK
    work = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge") if (ph or pw) else frame
    hh, ww = work.shape[:2]

    # (H,W,3) -> (blocks_y, blocks_x, 3, 8, 8): DCT считается батчем матричных
    # умножений, без единого питоновского цикла по блокам.
    blocks = work.reshape(hh // JPEG_BLOCK, JPEG_BLOCK, ww // JPEG_BLOCK, JPEG_BLOCK, 3)
    blocks = blocks.transpose(0, 2, 4, 1, 3)

    coeffs = _DCT @ (blocks - 0.5) @ _DCT.T
    scale = (_JPEG_Q / 255.0) * ((1.0 - q) * 2.0 + 0.02)
    coeffs = np.round(coeffs / scale) * scale
    restored = _DCT.T @ coeffs @ _DCT + 0.5

    out = restored.transpose(0, 3, 1, 4, 2).reshape(hh, ww, 3)
    return np.clip(out[:h, :w], 0.0, 1.0)


def posterize(frame: np.ndarray, levels: int) -> np.ndarray:
    """Уменьшить число градаций цвета — «палитра из 16 цветов»."""
    n = max(2, int(levels))
    return np.round(frame * (n - 1)) / (n - 1)


def cutout(
    frame: np.ndarray,
    rng: np.random.Generator,
    boxes: int = 1,
    size_range: tuple[float, float] = (0.05, 0.2),
) -> np.ndarray:
    """Закрыть несколько прямоугольников сплошным цветом (окклюзия).

    Зачем это законно, хотя разметка под заплаткой остаётся прежней: сеть учится
    ДОСТРАИВАТЬ объект по контексту, а не отказываться от ответа, если кусок
    закрыт партиклом, взрывом или элементом интерфейса. Заплатки маленькие
    (до ~20% стороны), поэтому цель остаётся в основном наблюдаемой.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    lo, hi = float(size_range[0]), float(size_range[1])
    for _ in range(max(0, int(boxes))):
        bh = max(2, int(h * rng.uniform(lo, hi)))
        bw = max(2, int(w * rng.uniform(lo, hi)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        x0 = int(rng.integers(0, max(1, w - bw)))
        if rng.random() < 0.5:
            fill = rng.random(3).astype(np.float32)
        else:
            fill = np.full(3, float(rng.random()), dtype=np.float32)
        out[y0:y0 + bh, x0:x0 + bw, :] = fill
    return out


# ---------------------------------------------------------------------------
# сборка
# ---------------------------------------------------------------------------
def augment_frame(
    frame: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Случайная цепочка аугментаций для одного кадра: uint8 -> uint8.

    Порядок операций повторяет физику тракта «экран -> камера -> файл»:
    сначала свет и цвет сцены (яркость, гамма, баланс), затем оптика (сдвиг
    каналов, расфокус), затем сенсор (шум) и только в конце кодек (JPEG,
    постеризация). Перемешивать их местами можно, но так распределение кадров
    ближе к тому, что придёт из реального захвата.
    """
    conf = cfg if cfg is not None else DEFAULT_AUGMENT
    img = _as_float(frame)

    if rng.random() < conf.p_brightness:
        img = brightness_contrast(
            img,
            brightness=float(rng.uniform(-conf.brightness, conf.brightness)),
            contrast=float(1.0 + rng.uniform(-conf.contrast, conf.contrast)),
        )
    if rng.random() < conf.p_gamma:
        img = adjust_gamma(img, float(rng.uniform(*conf.gamma_range)))
    if rng.random() < conf.p_color:
        gains = 1.0 + rng.uniform(-conf.color_gain, conf.color_gain, size=3)
        img = color_jitter(img, tuple(float(g) for g in gains))
    if rng.random() < conf.p_channel_swap:
        img = img[..., rng.permutation(3)]
    if rng.random() < conf.p_desaturate:
        img = desaturate(img, float(rng.uniform(0.5, 1.0)))
    if rng.random() < conf.p_channel_shift:
        img = channel_shift(
            img,
            dx=int(rng.integers(-1, 2)),
            dy=int(rng.integers(-1, 2)),
            amount=float(rng.uniform(*conf.channel_shift_amount)),
        )
    if rng.random() < conf.p_blur:
        img = blur(img, float(rng.uniform(0.3, 1.0)) * conf.blur_strength)
    if rng.random() < conf.p_noise:
        img = add_noise(img, float(rng.uniform(0.01, conf.noise_sigma)), rng)
    if rng.random() < conf.p_salt:
        img = salt_pepper(img, float(rng.uniform(0.001, conf.salt_amount)), rng)
    if rng.random() < conf.p_jpeg:
        img = jpeg_artifacts(img, float(rng.uniform(*conf.jpeg_quality)))
    if rng.random() < conf.p_posterize:
        img = posterize(img, int(rng.integers(*conf.posterize_levels)))
    if rng.random() < conf.p_cutout:
        img = cutout(
            img,
            rng,
            boxes=int(rng.integers(conf.cutout_boxes[0], conf.cutout_boxes[1] + 1)),
            size_range=conf.cutout_size,
        )

    return to_uint8(img)


def augment_batch(
    frames: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig | None = None,
) -> np.ndarray:
    """Аугментировать пачку кадров (N,H,W,3), каждый — своей случайной цепочкой.

    Именно «каждый своей»: одинаковая аугментация на весь батч даёт коррелированный
    градиент и заметно хуже учит инвариантности.
    """
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"Ожидались кадры (N,H,W,3), получено {arr.shape}")
    return np.stack([augment_frame(f, rng, cfg) for f in arr], axis=0)


__all__ = [
    "AugmentConfig",
    "DEFAULT_AUGMENT",
    "augment_frame",
    "augment_batch",
    "brightness_contrast",
    "adjust_gamma",
    "color_jitter",
    "desaturate",
    "channel_shift",
    "add_noise",
    "salt_pepper",
    "blur",
    "jpeg_artifacts",
    "posterize",
    "cutout",
    "to_uint8",
    "JPEG_BLOCK",
]
