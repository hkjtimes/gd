"""Обучение зрения: supervised-сегментация на бесконечной синтетике.

Что здесь решается
------------------
Задача выглядит как обычная сегментация, но у неё два перекоса, из-за которых
«просто CrossEntropy» даёт красивую метрику и бесполезную сеть:

1. **Дикая несбалансированность.** Около 80% пикселей — EMPTY, ещё 15% —
   SOLID (пол и потолок). Шипы, кольца и порталы вместе занимают проценты.
   Сеть, предсказывающая «везде пусто, снизу пол», получает pixel accuracy
   около 0.95 и убивает агента на первом же шипе. Лечится весами классов в
   CrossEntropy плюс Dice, который смотрит на ПЕРЕКРЫТИЕ фигур, а не на долю
   правильных пикселей, и потому не даёт редким классам исчезнуть.
2. **Цена ошибок разная.** Спутать порталы между собой — неприятно; не увидеть
   шип — смерть. Поэтому метрики считаются не только средние: `IoU_hazard` и
   `IoU_solid` выведены отдельно и по ним же выбирается лучший чекпойнт.

Валидация идёт на отложенных темах (см. `dataset.py`), то есть измеряет ровно
то, ради чего затевалась вся доменная рандомизация, — перенос на НОВЫЙ дизайн.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gdai.config import PerceptionConfig
from gdai.constants import CLASS_NAMES, HAZARD, NUM_CLASSES, SOLID
from gdai.perception.dataset import infinite_batches, make_loaders
from gdai.perception.model import PerceptionNet, resolve_device
from gdai.utils.checkpoint import save_checkpoint
from gdai.utils.logging import JsonlLogger, get_logger

_LOG = get_logger("perception.train")

# Веса классов в CrossEntropy. Подобраны по порядку величины их частоты, а не
# как 1/частота: строгая обратная частота даёт ORB вес в сотни раз больше
# EMPTY, и обучение начинает галлюцинировать кольца по всему кадру. Здесь
# «сглаженная» шкала: массовые классы придавлены, смертельно важные подняты.
#
# Числа не выдуманы, а подогнаны по замерам датасета. Реальные доли пикселей:
# EMPTY 87.7%, SOLID 10.4%, GOAL 0.96%, PLAYER 0.57%, PORTAL_MODE 0.16%,
# HAZARD 0.12%, остальные — сотые доли процента. Классы выучиваются строго в
# порядке своего размера, и при мягкой шкале (HAZARD 6 против EMPTY 0.3, то
# есть 20:1) очередь до опасностей просто не доходит: замер на 2000 шагах дал
# pixel_acc 0.977, IoU_solid 0.90 и IoU_hazard РОВНО 0 — сеть не предсказала ни
# одного пикселя шипа, пада и игрока даже на обучающем домене.
#
# Поэтому шкала «жёсткая»: отношение HAZARD к EMPTY поднято до 50:1, мелкие
# пады/кольца/порталы — до 20-40:1. Это всё ещё далеко от честного 1/частота
# (там было бы 700:1), и верхнюю границу пришлось искать по замерам с ДВУХ
# сторон:
#   * 20:1 (старая шкала, HAZARD=6): IoU_hazard = 0.000 — шипы не предсказаны ни
#     разу, ни на валидации, ни на обучающем домене;
#   * 100:1 (HAZARD=30): IoU_hazard = 0.11 на валидации, но в игре сеть рисует
#     120 пикселей опасности на кадр при истинных 13.6 — то есть 9-кратное
#     перепредсказание. После приоритетного сжатия карты (шип побеждает в блоке
#     2x2) это превращается в 43 «шипа» вместо 6, и политика, обученная на
#     чистой карте, разваливается;
#   * 50:1 (текущая) — компромисс между «не видит» и «видит везде».
CLASS_WEIGHTS: tuple[float, ...] = (
    0.30,   # EMPTY  — фон, его и так большинство
    0.55,   # SOLID  — пол/блоки, крупные и лёгкие
    15.00,  # HAZARD — самый мелкий класс и единственный, чей пропуск — смерть
    6.00,   # PLAYER — маленький, но всегда ровно один и всегда на месте
    12.00,  # PAD    — сотые доли процента пикселей
    12.00,  # ORB
    8.00,   # PORTAL_GRAVITY
    8.00,   # PORTAL_MODE
    8.00,   # PORTAL_SPEED
    3.00,   # GOAL   — редкий по кадрам, но крупный, когда виден
)

# Степень фокусировки CE на трудных пикселях (focal loss, Lin et al.).
#
# Зачем понадобилась, хотя веса классов уже есть. Веса меняют вклад класса
# ЛИНЕЙНО, а перекос здесь на три порядка: замер по датасету — EMPTY 85%
# пикселей, SOLID 11%, HAZARD 0.15%, PLAYER 0.6%. С нормировкой весов к среднему
# 1 (как того требует масштаб лосса) опасности получают около 2.6% суммарного
# CE, и обучение честно садится в ответ «пусто + пол + финиш»: прогон на 2000
# шагов давал pixel_acc 0.977, IoU_solid 0.90 и IoU_hazard РОВНО 0 — сеть ни
# разу не предсказала ни шип, ни игрока, ни пад (проверено по гистограмме
# предсказанных классов, и на обучающем домене тоже).
#
# Focal домножает вклад пикселя на (1-p_t)^gamma и потому давит именно те
# пиксели, которые УЖЕ уверенно угаданы: фон с p=0.99 при gamma=2 весит в 10^4
# раз меньше, а невыученный шип с p=0.01 сохраняет почти полный вес. Это
# ровно тот дисбаланс, который весами класса не лечится: он не между классами,
# а между «лёгкими» и «трудными» пикселями внутри кадра.
FOCAL_GAMMA: float = 2.0

# Вклад Dice в общий лосс. Dice — главный (а на первых сотнях шагов и
# единственный) источник градиента для крошечных классов: его производная по
# редкому классу пропорциональна 1/площадь, то есть для шипа в сотни раз
# больше, чем для фона. Поднят с 0.75 до 1.5 вместе с включением focal: Dice
# усредняется ПО КЛАССАМ, поэтому шип получает в нём ту же долю, что и фон, —
# это единственное слагаемое, которому всё равно, что шип в шестьсот раз мельче.
DICE_WEIGHT: float = 1.5

# Доля шагов на разогрев lr. Косинус без разогрева на первых шагах вместе с
# весами классов легко выбивает сеть в вырожденный ответ «всё HAZARD».
WARMUP_FRAC: float = 0.03
MIN_LR_FRAC: float = 0.05      # нижняя граница косинуса, доля от базового lr
WEIGHT_DECAY: float = 1e-4
LOG_EVERY: int = 25            # шагов между строками в metrics.jsonl


# ---------------------------------------------------------------------------
# лосс
# ---------------------------------------------------------------------------
def class_weight_tensor(device: torch.device | str = "cpu") -> Tensor:
    """Веса классов как тензор, нормированные к среднему 1.

    Нормировка нужна, чтобы масштаб лосса (а значит и подходящий lr) не зависел
    от того, как именно расставлены относительные веса.
    """
    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    return (w / w.mean()).to(device)


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    """Мягкий Dice по классам, присутствующим в батче.

    Зачем считать только по присутствующим: если в батче нет ни одного портала,
    «идеальный» Dice для него равен 1 при любом предсказании и просто разбавляет
    градиент. А главное — отсутствующий класс не должен штрафоваться как
    промах, иначе сеть учится вообще никогда его не предсказывать.
    """
    probs = logits.softmax(dim=1)
    num_classes = logits.shape[1]
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    cardinality = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    present = onehot.sum(dims) > 0
    if not bool(present.any()):
        return logits.new_zeros(())
    return 1.0 - dice[present].mean()


def focal_cross_entropy(
    logits: Tensor,
    target: Tensor,
    weight: Tensor,
    gamma: float = FOCAL_GAMMA,
) -> Tensor:
    """CrossEntropy с весами классов и focal-модуляцией `(1 - p_t)^gamma`.

    Зачем не `F.cross_entropy`: при gamma=0 это ровно она (с той же нормировкой
    на сумму весов), но при gamma>0 добавляется единственное, что спасает
    крошечные классы, — подавление уже выученных пикселей. Подробности и замеры
    в комментарии к `FOCAL_GAMMA`.
    """
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    sample_w = weight[target]
    if gamma > 0.0:
        # `detach` у модулятора: focal задуман как ПЕРЕВЗВЕШИВАНИЕ примеров, а
        # не как дополнительный путь градиента. Без detach сеть может уменьшать
        # лосс, просто становясь менее уверенной на лёгких пикселях.
        modulator = (1.0 - log_pt.detach().exp()).pow(float(gamma))
        sample_w = sample_w * modulator
    return -(sample_w * log_pt).sum() / weight[target].sum().clamp(min=1e-6)


class SegmentationLoss(nn.Module):
    """CrossEntropy с весами классов + Dice — ровно как требует SPEC §10.

    CE отвечает за попиксельную правоту, Dice — за то, чтобы объект не исчез
    целиком. По отдельности они дают либо «размытые пятна вместо шипов» (CE),
    либо неустойчивое обучение на почти пустых кадрах (Dice).
    """

    def __init__(
        self,
        weights: Tensor | None = None,
        dice_weight: float = DICE_WEIGHT,
        focal_gamma: float = FOCAL_GAMMA,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_gamma = float(focal_gamma)
        self.register_buffer(
            "weights", class_weight_tensor() if weights is None else weights.float()
        )

    def forward(self, logits: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Возвращает (лосс, слагаемые для логов)."""
        ce = focal_cross_entropy(logits, target, self.weights, self.focal_gamma)
        if self.dice_weight > 0.0:
            dice = dice_loss(logits, target)
            total = ce + self.dice_weight * dice
        else:
            dice = logits.new_zeros(())
            total = ce
        return total, {"ce": float(ce.detach()), "dice": float(dice.detach())}


# ---------------------------------------------------------------------------
# метрики
# ---------------------------------------------------------------------------
def confusion_matrix(pred: Tensor, target: Tensor, num_classes: int = NUM_CLASSES) -> Tensor:
    """Матрица ошибок (num_classes x num_classes), строки — истина.

    Зачем матрица, а не сразу IoU: из неё считаются ВСЕ метрики разом, и
    накапливать её по батчам можно простым сложением — значит валидация не
    зависит от того, каким размером батча её прогнали.
    """
    k = int(num_classes)
    idx = target.reshape(-1).to(torch.int64) * k + pred.reshape(-1).to(torch.int64)
    return torch.bincount(idx, minlength=k * k).reshape(k, k)


def metrics_from_confusion(conf: Tensor) -> dict[str, float]:
    """pixel_acc, mIoU и IoU по каждому классу из матрицы ошибок.

    mIoU усредняется только по классам, которые в выборке ЕСТЬ (или которые
    сеть предсказала): иначе редкий портал, не попавший в кадры, обнулял бы
    среднее и метрика перестала бы что-либо значить.
    """
    conf = conf.to(torch.float64)
    total = float(conf.sum())
    correct = float(conf.diag().sum())
    tp = conf.diag()
    union = conf.sum(dim=1) + conf.sum(dim=0) - tp
    valid = union > 0
    iou = torch.where(valid, tp / union.clamp(min=1.0), torch.zeros_like(tp))

    result: dict[str, float] = {
        "pixel_acc": correct / total if total > 0 else 0.0,
        "miou": float(iou[valid].mean()) if bool(valid.any()) else 0.0,
    }
    for cls, name in enumerate(CLASS_NAMES):
        result[f"iou_{name}"] = float(iou[cls]) if bool(valid[cls]) else float("nan")
    result["iou_hazard"] = result[f"iou_{CLASS_NAMES[HAZARD]}"]
    result["iou_solid"] = result[f"iou_{CLASS_NAMES[SOLID]}"]
    return result


@torch.no_grad()
def evaluate_perception(
    model: PerceptionNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Прогнать модель по валидационному потоку и вернуть метрики.

    Публичная функция, потому что ею пользуются и обучение, и тесты, и
    `pipeline`, где нужно проверить чужой чекпойнт на новых темах.
    """
    was_training = model.training
    model.eval()
    dev = torch.device(device)
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    seen = 0
    try:
        for i, (frames, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = model(frames.to(dev))
            pred = logits.argmax(dim=1).cpu()
            conf += confusion_matrix(pred, labels, NUM_CLASSES)
            seen += int(frames.shape[0])
    finally:
        if was_training:
            model.train()
    metrics = metrics_from_confusion(conf)
    metrics["samples"] = float(seen)
    return metrics


# ---------------------------------------------------------------------------
# расписание lr
# ---------------------------------------------------------------------------
def lr_lambda_cosine(step: int, total: int, warmup_frac: float = WARMUP_FRAC) -> float:
    """Множитель lr: линейный разогрев, затем косинус до `MIN_LR_FRAC`.

    Косинус (а не «шаг вниз») выбран потому, что прогон часто короткий: при
    600-1000 шагах ступенчатое расписание успевает сработать один раз и просто
    обрывает обучение, тогда как косинус аккуратно доводит веса до минимума.
    """
    total = max(1, int(total))
    warmup = max(1, int(total * float(warmup_frac)))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cosine


# ---------------------------------------------------------------------------
# обучение
# ---------------------------------------------------------------------------
def train_perception(cfg: PerceptionConfig) -> dict[str, Any]:
    """Обучить зрение и вернуть словарь метрик (SPEC §10).

    Что происходит по шагам: батч синтетики -> логиты -> CE(веса)+Dice ->
    AdamW с косинусным lr. Каждые `cfg.val_every` шагов — валидация на
    ОТЛОЖЕННЫХ темах, строка в `out_dir/metrics.jsonl` и сохранение
    `last.pt`; при улучшении mIoU — ещё и `best.pt`.

    Возвращает финальные метрики, лучшие метрики и служебные поля (пути,
    число параметров, скорость) — этого достаточно, чтобы CLI и тесты не
    лазили в файлы.
    """
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PerceptionNet(base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    n_params = model.count_parameters()
    if n_params >= 500_000:
        # Это контракт из SPEC §10, а не украшение: большая сеть не успеет
        # отработать рядом с политикой в реальном времени.
        raise ValueError(
            f"Модель слишком большая: {n_params} параметров при лимите 500k. "
            "Уменьшите base_channels или depth."
        )

    train_loader, val_loader = make_loaders(cfg)
    criterion = SegmentationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=WEIGHT_DECAY)
    total_steps = max(1, int(cfg.steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda_cosine(s, total_steps)
    )

    logger = JsonlLogger(out_dir)
    _LOG.info(
        "обучение зрения: %d шагов, батч %d, устройство %s, параметров %d",
        total_steps, cfg.batch_size, device, n_params,
    )

    best_score = -1.0
    best_metrics: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    loss_window: list[float] = []
    started = time.time()

    batches = infinite_batches(train_loader)
    model.train()
    try:
        for step in range(total_steps):
            frames, labels = next(batches)
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(frames)
            loss, parts = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Клип нормы: на кадре, где впервые появился редкий класс с большим
            # весом, градиент может быть в разы больше обычного.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            loss_window.append(float(loss.detach()))
            is_last = step + 1 == total_steps

            if (step + 1) % LOG_EVERY == 0 or is_last:
                elapsed = max(1e-6, time.time() - started)
                record = {
                    "split": "train",
                    "step": step + 1,
                    "loss": float(np.mean(loss_window)),
                    "ce": parts["ce"],
                    "dice": parts["dice"],
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "samples_per_sec": (step + 1) * int(cfg.batch_size) / elapsed,
                }
                logger.log(record)
                _LOG.info(
                    "шаг %d/%d loss %.4f (ce %.4f dice %.4f) lr %.2e %.1f сэмпл/с",
                    step + 1, total_steps, record["loss"], record["ce"],
                    record["dice"], record["lr"], record["samples_per_sec"],
                )
                loss_window.clear()

            if (step + 1) % max(1, int(cfg.val_every)) == 0 or is_last:
                metrics = evaluate_perception(model, val_loader, device)
                metrics["step"] = float(step + 1)
                last_metrics = metrics
                history.append({"split": "val", **metrics})
                logger.log(_json_safe({"split": "val", **metrics}))
                _LOG.info(
                    "валидация (held-out темы) шаг %d: acc %.4f mIoU %.4f "
                    "IoU_solid %.4f IoU_hazard %.4f",
                    step + 1, metrics["pixel_acc"], metrics["miou"],
                    metrics["iou_solid"], _nan_to_zero(metrics["iou_hazard"]),
                )

                save_checkpoint(
                    out_dir / "last.pt",
                    model.state_dict(),
                    config=cfg,
                    meta={"step": step + 1, "params": n_params},
                    extra={"metrics": metrics},
                )
                score = _selection_score(metrics)
                if score > best_score:
                    best_score = score
                    best_metrics = dict(metrics)
                    save_checkpoint(
                        out_dir / "best.pt",
                        model.state_dict(),
                        config=cfg,
                        meta={"step": step + 1, "params": n_params},
                        extra={"metrics": metrics},
                    )
    finally:
        logger.close()
        batches.close()
        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)

    duration = time.time() - started
    result: dict[str, Any] = {
        "steps": total_steps,
        "params": n_params,
        "device": str(device),
        "duration_sec": round(duration, 2),
        "steps_per_sec": round(total_steps / max(1e-6, duration), 3),
        "out_dir": str(out_dir),
        "best_path": str(out_dir / "best.pt"),
        "last_path": str(out_dir / "last.pt"),
        "metrics_path": str(out_dir / "metrics.jsonl"),
        "held_out_themes": list(_held_out_names()),
        "final": last_metrics,
        "best": best_metrics,
        "history": history,
    }
    # Плоские копии главных чисел: так их удобно печатать в CLI и сравнивать
    # в тестах, не разбирая вложенные словари.
    for key in ("pixel_acc", "miou", "iou_solid", "iou_hazard"):
        result[key] = float(last_metrics.get(key, float("nan")))
    _LOG.info(
        "готово за %.1f с: mIoU %.4f, IoU_solid %.4f, IoU_hazard %.4f",
        duration, result["miou"], result["iou_solid"], _nan_to_zero(result["iou_hazard"]),
    )
    return result


def _nan_to_zero(value: float) -> float:
    """NaN (класс не встретился) -> 0.0 для аккуратного вывода в лог."""
    return 0.0 if value != value else float(value)


def _shutdown_loader(loader: Any) -> None:
    """Погасить worker-процессы загрузчика сразу после обучения.

    Зачем явно: `persistent_workers=True` держит процессы живыми, пока жив сам
    загрузчик. В долгоживущем процессе (CLI `selfcheck`, тесты, ноутбук) после
    обучения остались бы висеть чужие процессы, каждый со своим рендерером и
    пулом уровней в памяти.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    try:
        if callable(shutdown):
            shutdown()
        loader._iterator = None
    except Exception:  # pragma: no cover - гасим по принципу «лучше, чем ничего»
        _LOG.debug("не удалось явно остановить worker'ов загрузчика", exc_info=True)


def _json_safe(record: dict[str, Any]) -> dict[str, Any]:
    """NaN -> None перед записью в metrics.jsonl.

    Зачем: NaN — валидный Python, но НЕ валидный JSON. Строгие читатели (jq,
    любой не-питоновский инструмент) споткнутся об него, а `null` понимают все
    и он честно означает «класса не было в выборке».
    """
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in record.items()}


def _selection_score(metrics: dict[str, float]) -> float:
    """Оценка чекпойнта: mIoU с добавкой за жизненно важные классы.

    Зачем не чистый mIoU: он усредняет по всем классам, и сеть, отлично
    рисующая пол и портал, но теряющая шипы, может выиграть у осторожной. А
    агента убивают именно шипы, поэтому HAZARD и SOLID входят в оценку ещё раз.
    """
    miou = _nan_to_zero(metrics.get("miou", 0.0))
    hazard = _nan_to_zero(metrics.get("iou_hazard", 0.0))
    solid = _nan_to_zero(metrics.get("iou_solid", 0.0))
    return 0.5 * miou + 0.3 * hazard + 0.2 * solid


def _held_out_names() -> Sequence[str]:
    """Имена отложенных тем — попадают в отчёт, чтобы прогон был воспроизводим."""
    from gdai.perception.dataset import HELD_OUT_THEME_NAMES

    return HELD_OUT_THEME_NAMES


# ---------------------------------------------------------------------------
# запуск из командной строки
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Аргументы для `python -m gdai.perception.train` и для `gdai train-perception`."""
    parser = argparse.ArgumentParser(
        description="Обучение зрения (U-Net) на синтетике с доменной рандомизацией",
    )
    parser.add_argument("--steps", type=int, default=None, help="число шагов обучения")
    parser.add_argument("--batch-size", type=int, default=None, help="размер батча")
    parser.add_argument("--lr", type=float, default=None, help="скорость обучения")
    parser.add_argument("--base-channels", type=int, default=None, help="ширина сети")
    parser.add_argument("--depth", type=int, default=None, help="число уровней U-Net")
    parser.add_argument("--val-every", type=int, default=None, help="шагов между валидациями")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--out", type=str, default=None, help="каталог прогона")
    parser.add_argument(
        "--no-augment", action="store_true", help="отключить аугментации кадра"
    )
    return parser


def config_from_args(args: argparse.Namespace, base: PerceptionConfig | None = None) -> PerceptionConfig:
    """Собрать `PerceptionConfig`, перекрыв только явно указанные аргументы."""
    cfg = base if base is not None else PerceptionConfig()
    changes: dict[str, Any] = {}
    if getattr(args, "steps", None) is not None:
        changes["steps"] = int(args.steps)
    if getattr(args, "batch_size", None) is not None:
        changes["batch_size"] = int(args.batch_size)
    if getattr(args, "lr", None) is not None:
        changes["lr"] = float(args.lr)
    if getattr(args, "base_channels", None) is not None:
        changes["base_channels"] = int(args.base_channels)
    if getattr(args, "depth", None) is not None:
        changes["depth"] = int(args.depth)
    if getattr(args, "val_every", None) is not None:
        changes["val_every"] = int(args.val_every)
    if getattr(args, "device", None) is not None:
        changes["device"] = str(args.device)
    if getattr(args, "out", None) is not None:
        changes["out_dir"] = str(args.out)
    if getattr(args, "no_augment", False):
        changes["augment"] = False
    return replace(cfg, **changes)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Точка входа CLI: разобрать аргументы, обучить, вернуть метрики."""
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    _LOG.info("конфигурация: %s", asdict(cfg))
    return train_perception(cfg)


__all__ = [
    "train_perception",
    "evaluate_perception",
    "SegmentationLoss",
    "dice_loss",
    "focal_cross_entropy",
    "FOCAL_GAMMA",
    "confusion_matrix",
    "metrics_from_confusion",
    "class_weight_tensor",
    "lr_lambda_cosine",
    "build_arg_parser",
    "config_from_args",
    "main",
    "CLASS_WEIGHTS",
    "DICE_WEIGHT",
]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    main()
