"""Графики обучения из `metrics.jsonl` в PNG (SPEC §13).

Зачем отдельный модуль, а не «посмотреть в tensorboard»: единственный формат
логов в проекте — построчный JSON, который пишет `gdai.utils.logging`. Он
читается чем угодно, дописывается на живом прогоне и не требует сервера.
Здесь он превращается в сетку графиков одной командой:

    python -m gdai plot --run runs/agent --out curves.png

Модуль намеренно ничего не знает о том, кто писал лог: набор панелей
собирается из тех ключей, которые реально встретились в файле. Поэтому одна
и та же функция рисует и кривые PPO (награда, прохождения, KL), и кривые
зрения (loss, IoU по шипам и блокам), и любой будущий лог — новые метрики
появятся на графике сами.

Backend всегда Agg: графики строятся на сервере без дисплея, в CI и внутри
обучения, где открывать окна нельзя.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gdai.utils.logging import get_logger, read_jsonl

_LOG = get_logger("viz.plots")

# Имя файла с метриками внутри каталога прогона.
METRICS_FILENAME: str = "metrics.jsonl"

# Ключи, по которым откладывается ось X, в порядке предпочтения. `global_step`
# честнее итераций: он сравним между запусками с разным rollout_steps.
X_KEYS: tuple[str, ...] = ("global_step", "step", "iteration", "epoch", "time")
X_LABELS: dict[str, str] = {
    "global_step": "кадров среды",
    "step": "шаг обучения",
    "iteration": "итерация",
    "epoch": "эпоха",
    "time": "секунд с начала",
}

# Служебные поля, которые графиком быть не могут.
SKIP_KEYS: frozenset[str] = frozenset(
    {"wall", "split", "kind", "level_name", "note", "event", "promoted"}
)


@dataclass(frozen=True)
class PanelSpec:
    """Одна панель сетки: какие метрики на ней рисуются и как подписаны."""

    keys: tuple[str, ...]
    title: str
    ylabel: str = ""
    logy: bool = False
    ylim: tuple[float, float] | None = None


# Порядок панелей подобран так, чтобы читать сверху вниз как отчёт: сначала
# «получилось ли» (награда, прохождения), потом «как учится» (лоссы, KL),
# и только затем техника (lr, скорость).
DEFAULT_PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(("mean_reward",), "Средняя награда за эпизод", "reward"),
    PanelSpec(
        ("success_rate", "success_rate_all", "mean_progress"),
        "Прохождения и прогресс",
        "доля",
        ylim=(0.0, 1.02),
    ),
    PanelSpec(("mean_ep_len",), "Длина эпизода", "кадров"),
    PanelSpec(("difficulty",), "Сложность (учебный план)", "", ylim=(0.0, 1.02)),
    PanelSpec(("entropy",), "Энтропия политики", "нат"),
    PanelSpec(("approx_kl", "clip_fraction"), "KL и доля клиппинга", ""),
    PanelSpec(("policy_loss", "value_loss"), "Лоссы PPO", ""),
    PanelSpec(("explained_variance",), "Explained variance критика", ""),
    PanelSpec(("loss", "ce", "dice"), "Лосс зрения", ""),
    PanelSpec(("pixel_acc", "miou"), "Точность карты", "доля", ylim=(0.0, 1.02)),
    PanelSpec(
        ("iou_solid", "iou_hazard"),
        "IoU по решающим классам",
        "IoU",
        ylim=(0.0, 1.02),
    ),
    PanelSpec(("lr",), "Learning rate", "", logy=True),
    PanelSpec(("fps", "samples_per_sec"), "Скорость", "в секунду"),
)

# Цвета линий: одна палитра на весь проект, чтобы графики разных прогонов
# читались одинаково.
LINE_COLORS: tuple[str, ...] = (
    "#2b7bd9", "#d94f2b", "#2ba84a", "#9b51e0", "#e0a51e",
    "#00a3a3", "#d92b7b", "#7a7a7a",
)


def metrics_path(run: str | os.PathLike[str]) -> Path:
    """Путь к `metrics.jsonl` по каталогу прогона или по самому файлу."""
    path = Path(run)
    if path.is_dir():
        return path / METRICS_FILENAME
    return path


def load_metrics(run: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Прочитать метрики прогона (каталог `runs/...` или файл `*.jsonl`).

    Битые строки пропускаются самим `read_jsonl`: лог мог оборваться на
    середине записи, если обучение убили — это не повод не строить график.
    """
    path = metrics_path(run)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла метрик {path}. Ожидается каталог прогона с "
            f"{METRICS_FILENAME} или путь к самому файлу."
        )
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Файл метрик {path} пуст — рисовать нечего")
    _LOG.info("прочитано %d записей из %s", len(records), path)
    return records


def choose_x_key(records: Sequence[dict[str, Any]]) -> str:
    """Выбрать ось X: первый из `X_KEYS`, который реально есть в логе."""
    present = {key for record in records for key in record}
    for key in X_KEYS:
        if key in present:
            return key
    raise ValueError(
        f"В логе нет ни одного ключа для оси X ({', '.join(X_KEYS)}); "
        "добавьте 'step' или 'global_step' в записи метрик"
    )


def available_metrics(records: Sequence[dict[str, Any]]) -> list[str]:
    """Числовые ключи лога, годные для графика (без служебных и осей X)."""
    keys: dict[str, None] = {}
    for record in records:
        for key, value in record.items():
            if key in SKIP_KEYS or key in X_KEYS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            keys.setdefault(key, None)
    return list(keys)


def _splits(records: Sequence[dict[str, Any]]) -> list[str | None]:
    """Какие «сплиты» есть в логе (train/val у зрения; None у политики)."""
    found: list[str | None] = []
    for record in records:
        split = record.get("split")
        if split not in found:
            found.append(split)
    return found


def series(
    records: Sequence[dict[str, Any]],
    key: str,
    x_key: str,
    split: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Пара массивов (x, y) для одной метрики; NaN и пропуски выброшены."""
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        if split is not None and record.get("split") != split:
            continue
        if key not in record or x_key not in record:
            continue
        value = record[key]
        x = record[x_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            continue
        if not math.isfinite(float(value)) or not math.isfinite(float(x)):
            continue
        xs.append(float(x))
        ys.append(float(value))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее с «разогревом», чтобы кривая начиналась с первой точки.

    Зачем: сырые кривые RL прыгают так, что тренд не виден вовсе, а обрезать
    начало ради окна сглаживания — значит потерять самую интересную часть,
    где агент только учится.
    """
    if window <= 1 or values.size == 0:
        return values
    window = min(int(window), values.size)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def build_panels(
    records: Sequence[dict[str, Any]],
    *,
    include_extra: bool = True,
) -> list[PanelSpec]:
    """Собрать список панелей под конкретный лог.

    Сначала берутся заранее описанные группы (у которых нашлась хотя бы одна
    метрика), затем — всё остальное числовое по одной панели на ключ: так
    новая метрика в обучении не потребует правки этого модуля.
    """
    present = set(available_metrics(records))
    panels: list[PanelSpec] = []
    used: set[str] = set()
    for spec in DEFAULT_PANELS:
        keys = tuple(key for key in spec.keys if key in present)
        if not keys:
            continue
        panels.append(
            PanelSpec(keys, spec.title, spec.ylabel, spec.logy, spec.ylim)
        )
        used.update(keys)
    if include_extra:
        for key in available_metrics(records):
            if key in used:
                continue
            panels.append(PanelSpec((key,), key, ""))
            used.add(key)
    return panels


def plot_metrics(
    records: Sequence[dict[str, Any]],
    out_path: str | os.PathLike[str],
    *,
    title: str | None = None,
    x_key: str | None = None,
    smooth: int = 1,
    max_cols: int = 3,
    dpi: int = 130,
    include_extra: bool = True,
) -> str:
    """Нарисовать сетку графиков по записям лога и вернуть путь к PNG.

    `smooth > 1` включает скользящее среднее: сырая кривая остаётся бледной
    линией под сглаженной, чтобы шум был виден, но не мешал.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("Пустой список записей — рисовать нечего")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    x_name = x_key or choose_x_key(records)
    panels = build_panels(records, include_extra=include_extra)
    if not panels:
        raise ValueError(
            "В логе нет ни одной числовой метрики — возможно, это не metrics.jsonl"
        )
    splits = _splits(records)

    cols = max(1, min(int(max_cols), len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    fig, axes_grid = plt.subplots(
        rows, cols, figsize=(5.2 * cols, 3.1 * rows), dpi=int(dpi), squeeze=False
    )
    axes = [ax for row in axes_grid for ax in row]

    for ax, spec in zip(axes, panels):
        drawn = 0
        color_index = 0
        for key in spec.keys:
            for split in splits:
                xs, ys = series(records, key, x_name, split)
                if xs.size == 0:
                    continue
                order = np.argsort(xs, kind="stable")
                xs, ys = xs[order], ys[order]
                label = key if split is None else f"{key} [{split}]"
                color = LINE_COLORS[color_index % len(LINE_COLORS)]
                color_index += 1
                if smooth > 1 and ys.size > smooth:
                    ax.plot(xs, ys, color=color, alpha=0.22, linewidth=1.0)
                    ax.plot(
                        xs, moving_average(ys, smooth), color=color,
                        linewidth=1.8, label=label,
                    )
                else:
                    ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
                drawn += 1
        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(X_LABELS.get(x_name, x_name), fontsize=8)
        if spec.ylabel:
            ax.set_ylabel(spec.ylabel, fontsize=8)
        if spec.logy:
            ax.set_yscale("log")
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=8)
        if drawn > 1 or (drawn == 1 and len(spec.keys) > 1):
            ax.legend(fontsize=7, loc="best")

    # Лишние оси в последнем ряду прячем, а не оставляем пустыми рамками.
    for ax in axes[len(panels):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    _LOG.info("график сохранён: %s (%d панелей)", out, len(panels))
    return str(out)


def plot_run(
    run: str | os.PathLike[str],
    out_path: str | os.PathLike[str] = "curves.png",
    *,
    title: str | None = None,
    smooth: int = 1,
    max_cols: int = 3,
    dpi: int = 130,
    include_extra: bool = True,
) -> str:
    """Прочитать `metrics.jsonl` прогона и нарисовать все кривые в один PNG."""
    records = load_metrics(run)
    heading = title if title is not None else f"GDAI — {Path(run).name}"
    return plot_metrics(
        records,
        out_path,
        title=heading,
        smooth=smooth,
        max_cols=max_cols,
        dpi=dpi,
        include_extra=include_extra,
    )


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Последнее значение каждой метрики — короткая сводка для лога и README."""
    result: dict[str, float] = {}
    for key in available_metrics(records):
        for record in reversed(records):
            value = record.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    result[key] = float(value)
                    break
    return result


def build_parser() -> argparse.ArgumentParser:
    """Аргументы команды `python -m gdai plot`."""
    parser = argparse.ArgumentParser(
        prog="gdai-plot",
        description="Построить графики обучения из metrics.jsonl",
    )
    parser.add_argument(
        "--run", required=True, help="каталог прогона (runs/agent) или файл *.jsonl"
    )
    parser.add_argument("--out", default="curves.png", help="куда сохранить PNG")
    parser.add_argument("--title", default=None, help="заголовок графика")
    parser.add_argument(
        "--smooth", type=int, default=1, help="окно скользящего среднего (1 = без сглаживания)"
    )
    parser.add_argument("--cols", type=int, default=3, help="сколько панелей в ряду")
    parser.add_argument("--dpi", type=int, default=130, help="разрешение PNG")
    parser.add_argument(
        "--only-known",
        action="store_true",
        help="рисовать только известные метрики, без прочих числовых полей",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: `python -m gdai.viz.plots --run runs/agent --out curves.png`."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    path = plot_run(
        args.run,
        args.out,
        title=args.title,
        smooth=int(args.smooth),
        max_cols=int(args.cols),
        dpi=int(args.dpi),
        include_extra=not args.only_known,
    )
    _LOG.info("готово: %s", path)
    return 0


__all__ = [
    "PanelSpec",
    "DEFAULT_PANELS",
    "METRICS_FILENAME",
    "metrics_path",
    "load_metrics",
    "available_metrics",
    "choose_x_key",
    "series",
    "moving_average",
    "build_panels",
    "plot_metrics",
    "plot_run",
    "summarize",
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
