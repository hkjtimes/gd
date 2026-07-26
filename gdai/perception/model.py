"""U-Net зрения: «любой дизайн -> каноническая карта».

Зачем именно такая архитектура
------------------------------
Сеть решает узкую задачу: перевести кадр 128x72 в карту из десяти классов той
же величины. Ей не нужно узнавать тысячу объектов — нужно надёжно отделить
ФОРМУ игрового объекта от бесконечно разнообразной декорации. Отсюда три
решения, которые здесь важнее «глубины» и «модности»:

1. **Маленькая (~200k параметров при base_channels=24).** Большая сеть на
   синтетике быстро начинает запоминать частные признаки конкретных тем; узкая
   вынуждена опираться на устойчивое — контур, симметрию, положение. Плюс она
   обязана работать в реальном времени рядом с политикой на одном CPU.
2. **GroupNorm, а не BatchNorm.** В бою кадры приходят по одному
   (`predict`), и статистика батча вырождается: BatchNorm в этом режиме
   считает mean/var по единственному кадру и «плывёт» при смене темы —
   ровно там, где нам нужна стабильность. GroupNorm нормирует внутри кадра,
   поэтому результат для батча 1 и для батча 16 идентичен.
3. **Основная работа — на пониженном разрешении.** Свёртка на полном кадре
   стоит в 4 раза дороже, чем на половинном, а форма шипа читается и там.
   Поэтому полноразмерными остаются только тонкий stem и последний блок
   декодера, который восстанавливает границы: так сеть держит > 200 кадров/с
   на четырёх ядрах CPU при батче 16.

Контракт входа/выхода (SPEC §10): вход (B,3,72,128) float32 в [0,1], выход
(B,10,72,128) — ЛОГИТЫ, без softmax (его берёт на себя функция потерь).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gdai.constants import NUM_CLASSES, OBS_H, OBS_W

# Максимальное число групп в GroupNorm. Восемь — компромисс: групп достаточно,
# чтобы нормировка была «локальной по смыслу», и при этом любое наше число
# каналов (16, 24, 48, 72, 96) на неё делится.
MAX_GROUPS: int = 8

# Во сколько раз каналы первого (полноразмерного) слоя уже базовых. Полное
# разрешение — самое дорогое место сети, и держать там широкий тензор нет
# смысла: там решаются только границы, а не «что это за объект».
STEM_RATIO: float = 2.0 / 3.0

# Потолок роста каналов вглубь: c_i = base * min(i, GROWTH_CAP). Без потолка
# глубина 4-5 мгновенно съедает бюджет в 500k параметров, ничего не давая:
# на карте 9x16 признаков и так «много каналов на пиксель».
GROWTH_CAP: int = 3


def _num_groups(channels: int) -> int:
    """Сколько групп взять для GroupNorm на данном числе каналов.

    Зачем не константа: число каналов задаётся конфигом (`base_channels`), и
    при нестандартном значении фиксированные 8 групп уронили бы конструктор.
    Берём наибольший делитель, не превосходящий `MAX_GROUPS`.
    """
    c = int(channels)
    if c <= 0:
        raise ValueError(f"Число каналов должно быть положительным, получено {channels}")
    g = min(MAX_GROUPS, c)
    while c % g:
        g -= 1
    return g


def conv_block(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    """Свёртка + GroupNorm + ReLU — единственный «кирпич» этой сети.

    Bias у свёртки выключен намеренно: сразу за ней идёт нормировка со своим
    сдвигом, и второй сдвиг был бы бесполезной парой тысяч параметров.
    """
    return nn.Sequential(
        nn.Conv2d(int(in_ch), int(out_ch), int(kernel),
                  stride=int(stride), padding=int(kernel) // 2, bias=False),
        nn.GroupNorm(_num_groups(out_ch), int(out_ch)),
        nn.ReLU(inplace=True),
    )


def resolve_device(device: str = "auto") -> torch.device:
    """Превратить строку конфига в `torch.device` («auto» -> cuda, если есть).

    Зачем публично: тем же правилом обязаны пользоваться обучение зрения,
    политика и `pipeline`, иначе чекпойнт, сохранённый на одном устройстве,
    начнёт грузиться на другое по чуть иным правилам.
    """
    name = str(device).strip().lower()
    if name in ("auto", "", "default"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class PerceptionNet(nn.Module):
    """Маленький U-Net: вход (B,3,72,128) float32 в [0,1], выход (B,10,72,128) логиты.

    Схема (при `base_channels=24, depth=3`):

    ```
    72x128  stem 3->16 ------------------------------skip-----------------+
    36x64   down 16->24, conv 24 ----------------skip---------------+     |
    18x32   down 24->48, conv 48 ------------skip------------+      |     |
    9x16    down 48->72, conv 72, conv 72 (bottleneck)       |      |     |
    18x32   up + concat(72,48) -> 1x1 -> 48, conv 48 --------+      |     |
    36x64   up + concat(48,24) -> 1x1 -> 24, conv 24 ---------------+     |
    72x128  up + concat(24,16) -> 1x1 -> 16, conv 16 ---------------------+
    72x128  head 1x1 16 -> 10 логитов
    ```

    Приём с `1x1` сразу после конкатенации — способ уложиться в бюджет: он
    сжимает объединённый тензор до рабочей ширины ДО дорогой свёртки 3x3, и
    самый большой (полноразмерный) блок декодера обходится втрое дешевле.

    Апсемплинг — `nearest` по размеру соответствующего skip-тензора, а не
    множителем 2. Так сеть корректно работает и на кадрах, чьи стороны не
    делятся на 2^depth (например, при захвате настоящей игры).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 24,
        depth: int = 3,
    ) -> None:
        super().__init__()
        num_classes = int(num_classes)
        base_channels = int(base_channels)
        depth = int(depth)
        if num_classes < 2:
            raise ValueError(f"Классов должно быть >= 2, получено {num_classes}")
        if base_channels < 4:
            raise ValueError(f"base_channels должен быть >= 4, получено {base_channels}")
        if not 1 <= depth <= 5:
            raise ValueError(f"depth должен быть в 1..5, получено {depth}")

        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth

        stem_ch = max(8, int(round(base_channels * STEM_RATIO)))
        channels: list[int] = [stem_ch] + [
            base_channels * min(i, GROWTH_CAP) for i in range(1, depth + 1)
        ]
        self.channels: tuple[int, ...] = tuple(channels)

        # Кодер: каждый уровень — снижение разрешения вдвое (свёртка со stride 2,
        # она дешевле «pool + conv» и заодно учит, ЧТО именно стоит сохранить)
        # плюс одна свёртка на новом масштабе.
        self.stem = conv_block(3, channels[0])
        self.down = nn.ModuleList(
            nn.Sequential(
                conv_block(channels[i - 1], channels[i], stride=2),
                conv_block(channels[i], channels[i]),
            )
            for i in range(1, depth + 1)
        )
        # Дно: ещё одна свёртка на самом дешёвом разрешении. Здесь у сети самый
        # широкий обзор — именно тут решается «это шип на блоке или узор фона».
        self.bottleneck = conv_block(channels[-1], channels[-1])
        # Декодер: 1x1 для сжатия конкатенации, затем 3x3 для восстановления формы.
        self.up = nn.ModuleList(
            nn.Sequential(
                conv_block(channels[i] + channels[i - 1], channels[i - 1], kernel=1),
                conv_block(channels[i - 1], channels[i - 1]),
            )
            for i in range(depth, 0, -1)
        )
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self._init_weights()

    # --- инициализация -------------------------------------------------------
    def _init_weights(self) -> None:
        """He-инициализация под ReLU + нулевой bias у головы.

        Зачем нулевой bias на выходе: в самом начале обучения все классы должны
        быть равновероятны. Если голова стартует со случайным смещением, первые
        сотни шагов уходят на то, чтобы «переспорить» этот перекос, а при
        сильно несбалансированных классах (EMPTY — 80% пикселей) это ещё и
        загоняет сеть в вырожденный ответ.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    # --- прямой проход -------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Логиты классов для батча кадров: (B,3,H,W) -> (B,num_classes,H,W)."""
        if x.dim() != 4:
            raise ValueError(f"Ожидался тензор (B,3,H,W), получено {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Ожидалось 3 канала на входе, получено {x.shape[1]}")

        skips: list[Tensor] = []
        h = self.stem(x)
        skips.append(h)
        for block in self.down:
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for i, block in enumerate(self.up):
            skip = skips[-2 - i]
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat((h, skip), dim=1))

        return self.head(h)

    # --- вывод для внешнего мира --------------------------------------------
    def _prepare_frames(self, frames: np.ndarray) -> Tensor:
        """Кадры (N,H,W,3) uint8/float -> тензор (N,3,H,W) float32 в [0,1].

        Зачем терпимость к float: кадр может прийти уже нормализованным (из
        среды, из захвата экрана), и молчаливое деление такого входа на 255
        сделало бы картинку чёрной — сеть бы «ослепла» без единой ошибки.
        """
        arr = np.asarray(frames)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Ожидались кадры (N,H,W,3), получено {arr.shape}")
        if arr.dtype == np.uint8:
            data = arr.astype(np.float32) / 255.0
        else:
            data = arr.astype(np.float32, copy=False)
            if data.size and float(data.max()) > 1.5:
                data = data / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(data.transpose(0, 3, 1, 2)))
        return tensor.to(self.device)

    @property
    def device(self) -> torch.device:
        """Устройство, на котором лежат веса (нужно, чтобы не гонять данные зря)."""
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self, frame_uint8: np.ndarray) -> np.ndarray:
        """Один кадр (H,W,3) uint8 -> карта классов (H,W) uint8.

        Основной боевой вызов: так зрение работает внутри `GDAgent` и в окне
        «что видит нейросеть».
        """
        arr = np.asarray(frame_uint8)
        if arr.ndim != 3:
            raise ValueError(f"Ожидался кадр (H,W,3), получено {arr.shape}")
        return self.predict_batch(arr[None, ...])[0]

    @torch.no_grad()
    def predict_batch(self, frames_uint8: np.ndarray, chunk: int = 64) -> np.ndarray:
        """Пачка кадров (N,H,W,3) uint8 -> карты классов (N,H,W) uint8.

        Батч режется на куски по `chunk`: при оценке зрения на тысячах кадров
        один большой прогон занял бы сотни мегабайт под промежуточные тензоры,
        а выигрыша по скорости уже не даёт.
        """
        arr = np.asarray(frames_uint8)
        if arr.ndim != 4:
            raise ValueError(f"Ожидались кадры (N,H,W,3), получено {arr.shape}")
        was_training = self.training
        self.eval()
        try:
            outputs: list[np.ndarray] = []
            step = max(1, int(chunk))
            for start in range(0, arr.shape[0], step):
                batch = self._prepare_frames(arr[start:start + step])
                logits = self.forward(batch)
                outputs.append(logits.argmax(dim=1).to(torch.uint8).cpu().numpy())
        finally:
            if was_training:
                self.train()
        return np.concatenate(outputs, axis=0) if outputs else np.zeros(
            (0,) + arr.shape[1:3], dtype=np.uint8
        )

    # --- сервис --------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Число параметров — контракт SPEC §10 требует держать его < 500k."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(int(p.numel()) for p in params)

    def receptive_field(self) -> int:
        """Грубая оценка рецептивного поля в пикселях входа (для отладки).

        Зачем: если поле окажется меньше объекта, сеть физически не сможет
        отличить «шип» от «угла блока» — это первое, что стоит проверить, если
        IoU по HAZARD не растёт.
        """
        rf = 1
        jump = 1
        for _ in range(2):  # stem + первая свёртка ветви
            rf += 2 * jump
        for _ in range(self.depth):
            jump *= 2
            rf += 2 * jump      # свёртка со stride 2
            rf += 2 * jump      # свёртка на новом масштабе
        rf += 2 * jump          # bottleneck
        return int(rf)

    def extra_repr(self) -> str:
        return (
            f"classes={self.num_classes}, base={self.base_channels}, "
            f"depth={self.depth}, channels={self.channels}"
        )


def build_perception_net(config: Any = None) -> PerceptionNet:
    """Создать сеть по `PerceptionConfig` (или по умолчанию, если конфиг None).

    Зачем: и обучение, и загрузка чекпойнта, и `pipeline` должны собирать сеть
    ОДИНАКОВО — иначе `load_state_dict` упадёт на несовпадении форм.
    """
    if config is None:
        return PerceptionNet()
    base = int(getattr(config, "base_channels", 24))
    depth = int(getattr(config, "depth", 3))
    return PerceptionNet(base_channels=base, depth=depth)


def load_perception_net(
    path: str,
    device: str = "auto",
    strict: bool = True,
) -> PerceptionNet:
    """Загрузить обученное зрение из `best.pt`/`last.pt`.

    Архитектура берётся из конфига, сохранённого рядом с весами: так файл
    самодостаточен и его не нужно сопровождать «а какой там был base_channels».
    """
    from gdai.utils.checkpoint import load_checkpoint

    payload = load_checkpoint(path, map_location="cpu")
    cfg = payload.get("config") or {}
    base = int(cfg.get("base_channels", 24)) if isinstance(cfg, dict) else 24
    depth = int(cfg.get("depth", 3)) if isinstance(cfg, dict) else 3
    model = PerceptionNet(base_channels=base, depth=depth)
    model.load_state_dict(payload["state_dict"], strict=strict)
    model.to(resolve_device(device))
    model.eval()
    return model


def input_shape() -> tuple[int, int, int]:
    """Форма входа сети (C,H,W) — единая точка правды для тестов и визуализации."""
    return (3, OBS_H, OBS_W)


__all__ = [
    "PerceptionNet",
    "build_perception_net",
    "load_perception_net",
    "conv_block",
    "resolve_device",
    "input_shape",
    "MAX_GROUPS",
    "STEM_RATIO",
    "GROWTH_CAP",
]
