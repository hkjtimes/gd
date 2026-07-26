"""Карта внимания политики: на что именно смотрит ИИ (SPEC §13).

Вопрос, ради которого модуль написан: политика видит только каноническую
карту классов, но какая её часть решает исход? Ответ берётся честно —
градиентом логита действия «держать» по one-hot входу:

```
    saliency(y, x) = | d logit(hold) / d sem_onehot[:, y, x] |   (сумма по классам)
```

Большой градиент означает, что подмена класса в этом пикселе сильнее всего
изменила бы решение. Тепловая карта накладывается поверх карты классов —
получается картинка «агент смотрит на шип перед собой», которую можно
показать человеку и по которой видно, когда агент смотрит не туда
(например, на декоративный узор, просочившийся в карту из-за ошибки зрения).

Всё считается на сетке политики (36x64) и растягивается до кадра (72x128),
потому что именно на этой сетке живёт вход сети: показывать более мелкую
детализацию значило бы врать о разрешении, которым агент располагает.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from gdai.constants import ACTION_HOLD, ACTION_NONE, CLASS_NAMES, OBS_H, OBS_W
from gdai.utils.logging import get_logger

_LOG = get_logger("viz.saliency")

# Что дифференцируем. "hold" — прямо по SPEC; "margin" информативнее для
# двухдейственной политики (насколько «держать» лучше, чем «не держать»);
# "value" показывает, из чего складывается оценка положения.
TARGETS: tuple[str, ...] = ("hold", "none", "margin", "value")

DEFAULT_CMAP: str = "inferno"


def _resolve_target(logits: Any, value: Any, target: str) -> Any:
    """Скаляр, по которому берётся градиент."""
    if target not in TARGETS:
        raise ValueError(f"target={target!r} неизвестен, допустимы {TARGETS}")
    if target == "hold":
        return logits[0, ACTION_HOLD]
    if target == "none":
        return logits[0, ACTION_NONE]
    if target == "margin":
        return logits[0, ACTION_HOLD] - logits[0, ACTION_NONE]
    return value.reshape(-1)[0]


def _input_grad(
    policy: Any,
    sem_uint8: np.ndarray,
    features: np.ndarray,
    target: str,
) -> Any:
    """Градиент выбранного скаляра по one-hot входу: тензор (1, C, h, w).

    Зачем `torch.autograd.grad`, а не `backward()`: нам нужен градиент по
    входу, а не по весам, и порча `.grad` у параметров обученной сети (тем
    более во время просмотра, где рядом может идти обучение) недопустима.
    """
    import torch

    from gdai.agent.networks import semantic_to_tensor

    device = getattr(policy, "device", None)
    if device is None:
        device = next(policy.parameters()).device

    sem = semantic_to_tensor(sem_uint8, device=device).detach().clone()
    sem.requires_grad_(True)
    feat = torch.as_tensor(
        np.asarray(features, dtype=np.float32).reshape(1, -1)
    ).to(device)

    was_training = bool(getattr(policy, "training", False))
    policy.eval()
    try:
        with torch.enable_grad():
            logits, value = policy(sem, feat)
            scalar = _resolve_target(logits, value, target)
            (grad,) = torch.autograd.grad(scalar, sem, retain_graph=False)
    finally:
        if was_training:
            policy.train()
    return grad.detach()


def saliency_map(
    policy: Any,
    sem_uint8: np.ndarray,
    features: np.ndarray,
    *,
    target: str = "hold",
    normalize: bool = True,
    upsample: bool = True,
) -> np.ndarray:
    """Тепловая карта внимания: float32 (72, 128) в диапазоне 0..1.

    `target="hold"` — ровно то, что просит SPEC: модуль абсолютного градиента
    логита «держать» по one-hot карте, просуммированный по классам.
    `normalize=False` отдаёт сырые величины градиента (нужно, когда карты
    нескольких кадров сравниваются между собой).
    """
    grad = _input_grad(policy, sem_uint8, features, target)
    heat = grad.abs().sum(dim=1)[0].cpu().numpy().astype(np.float32)

    if upsample:
        fy = max(1, OBS_H // heat.shape[0])
        fx = max(1, OBS_W // heat.shape[1])
        heat = np.repeat(np.repeat(heat, fy, axis=0), fx, axis=1)
        heat = heat[:OBS_H, :OBS_W]

    if normalize:
        peak = float(heat.max())
        # Нулевой градиент бывает у неинициализированной или «уверенной» сети —
        # деление на ноль здесь дало бы NaN прямо в окне визуализации.
        heat = heat / peak if peak > 1e-12 else np.zeros_like(heat)
    return heat


def class_saliency(
    policy: Any,
    sem_uint8: np.ndarray,
    features: np.ndarray,
    *,
    target: str = "hold",
    normalize: bool = True,
) -> np.ndarray:
    """Вклад каждого семантического класса в решение: float32 (NUM_CLASSES,).

    Зачем: сводка «шипы важнее блоков в 4 раза» читается быстрее любой
    картинки и сразу показывает, не выучила ли политика ерунду (например,
    реагировать на класс GOAL, которого в кадре почти никогда нет).
    """
    grad = _input_grad(policy, sem_uint8, features, target)
    per_class = grad.abs().sum(dim=(2, 3))[0].cpu().numpy().astype(np.float32)
    if normalize:
        total = float(per_class.sum())
        per_class = per_class / total if total > 1e-12 else np.zeros_like(per_class)
    return per_class


def saliency_rgb(saliency: np.ndarray, cmap: str = DEFAULT_CMAP) -> np.ndarray:
    """Тепловая карта -> (H, W, 3) uint8 по цветовой шкале matplotlib."""
    heat = np.clip(np.asarray(saliency, dtype=np.float32), 0.0, 1.0)
    try:
        import matplotlib

        colormap = matplotlib.colormaps[cmap]
        rgba = colormap(heat)
        return (rgba[..., :3] * 255.0).astype(np.uint8)
    except Exception as exc:  # pragma: no cover - экзотическая сборка matplotlib
        _LOG.debug("matplotlib не дал палитру %s (%s) — рисуем вручную", cmap, exc)
        return _fallback_rgb(heat)


def _fallback_rgb(heat: np.ndarray) -> np.ndarray:
    """Запасная шкала чёрный -> красный -> жёлтый без matplotlib."""
    red = np.clip(heat * 2.0, 0.0, 1.0)
    green = np.clip(heat * 2.0 - 1.0, 0.0, 1.0)
    blue = np.clip(heat * 0.3, 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def overlay_saliency(
    sem_uint8: np.ndarray,
    saliency: np.ndarray,
    *,
    alpha: float = 0.7,
    cmap: str = DEFAULT_CMAP,
    gamma: float = 0.7,
) -> np.ndarray:
    """Наложить тепловую карту на карту классов -> (H, W, 3) uint8.

    Прозрачность пропорциональна самой величине внимания: там, где агенту
    всё равно, остаётся исходная карта классов, а горячие места закрашиваются
    цветом шкалы. `gamma < 1` подтягивает средние значения — иначе на карте
    видно только одно ярчайшее пятно.
    """
    from gdai.env.semantic import semantic_to_rgb

    heat = np.clip(np.asarray(saliency, dtype=np.float32), 0.0, 1.0) ** float(gamma)
    base = semantic_to_rgb(np.asarray(sem_uint8, dtype=np.uint8)).astype(np.float32)
    if heat.shape != base.shape[:2]:
        raise ValueError(
            f"Размер карты внимания {heat.shape} не совпадает с картой классов "
            f"{base.shape[:2]}"
        )
    color = saliency_rgb(heat, cmap=cmap).astype(np.float32)
    weight = (heat * float(alpha))[..., None]
    return np.clip(base * (1.0 - weight) + color * weight, 0, 255).astype(np.uint8)


def save_saliency_png(
    path: str | os.PathLike[str],
    sem_uint8: np.ndarray,
    saliency: np.ndarray,
    *,
    frame_rgb: np.ndarray | None = None,
    class_scores: np.ndarray | None = None,
    title: str | None = None,
    dpi: int = 130,
) -> str:
    """Сохранить картинку «на что смотрит ИИ» и вернуть путь к файлу.

    Панели: (опционально) исходный кадр, карта классов, тепловая карта,
    наложение и — если передан `class_scores` — столбики вклада классов.
    Всё через Agg, без интерактивного backend: функция обязана работать на
    сервере без дисплея.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from gdai.env.semantic import semantic_to_rgb

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Any, str]] = []
    if frame_rgb is not None:
        panels.append(("кадр игры", np.asarray(frame_rgb, dtype=np.uint8), "image"))
    panels.append(("карта классов", semantic_to_rgb(np.asarray(sem_uint8, np.uint8)), "image"))
    panels.append(("внимание политики", np.asarray(saliency, dtype=np.float32), "heat"))
    panels.append(("наложение", overlay_saliency(sem_uint8, saliency), "image"))
    if class_scores is not None:
        panels.append(("вклад классов", np.asarray(class_scores, dtype=np.float32), "bars"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0), dpi=dpi)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, data, kind) in zip(axes, panels):
        if kind == "image":
            ax.imshow(data, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        elif kind == "heat":
            im = ax.imshow(data, cmap=DEFAULT_CMAP, interpolation="nearest", vmin=0.0)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.bar(range(len(data)), data, color="#d94f2b")
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(CLASS_NAMES[: len(data)], rotation=60, fontsize=7, ha="right")
            ax.grid(axis="y", alpha=0.3)
        ax.set_title(name, fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("карта внимания сохранена: %s", out)
    return str(out)


def saliency_from_obs(
    policy: Any,
    obs: dict[str, np.ndarray],
    *,
    sem: np.ndarray | None = None,
    target: str = "hold",
) -> tuple[np.ndarray, np.ndarray]:
    """Удобная обёртка вокруг наблюдения среды: `(карта классов, карта внимания)`.

    Аргумент `sem` позволяет подставить предсказание зрения вместо эталона из
    `obs["semantic"]` — именно по предсказанию агент работает в бою, и именно
    его внимание интересно смотреть.
    """
    if "features" not in obs:
        raise KeyError(
            "В наблюдении нет ключа 'features' — политике нечего подать на вход"
        )
    source = sem if sem is not None else obs.get("semantic")
    if source is None:
        raise KeyError(
            "В наблюдении нет ключа 'semantic': передайте карту явно "
            "(obs_mode='semantic'/'both' или sem=...)"
        )
    grid = np.asarray(source, dtype=np.uint8)
    return grid, saliency_map(policy, grid, obs["features"], target=target)


__all__ = [
    "saliency_map",
    "class_saliency",
    "overlay_saliency",
    "saliency_rgb",
    "save_saliency_png",
    "saliency_from_obs",
    "TARGETS",
    "DEFAULT_CMAP",
]
