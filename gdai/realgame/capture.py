"""Захват экрана настоящей Geometry Dash (SPEC §14) — необязательный сценарий.

ВНИМАНИЕ. Это не основной путь проекта и не «режим по умолчанию». Модуль
требует **ручной калибровки**: человек сам указывает прямоугольник игрового
поля на экране, потому что автоматически найти его нельзя — окно игры бывает
любого размера, с любыми полями, на любом мониторе. Ошибка калибровки на
десяток пикселей меняет геометрию кадра, а зрение обучено на строго
определённой камере (16x9 тайлов, игрок на `PLAYER_X_IN_VIEW`), поэтому
неправильный прямоугольник даёт бессмысленную карту и агент играть не будет.

Что здесь есть:

* `CaptureRegion` — прямоугольник игрового поля, сохраняемый в JSON;
* `ScreenCapture` — захват через `mss` (мягкий импорт) и ресайз в 128x72;
* `calibrate_region` — интерактивный выбор прямоугольника мышью;
* `resize_frame` — ресайз через cv2/Pillow/numpy, что найдётся.

`mss` в зависимостях проекта нет. Без него модуль импортируется нормально, а
понятная ошибка появляется в момент первой попытки что-то захватить.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gdai.constants import OBS_H, OBS_W
from gdai.utils.logging import get_logger

_LOG = get_logger("realgame.capture")

# Куда по умолчанию кладётся калибровка: рядом с прогонами, а не в корень.
DEFAULT_REGION_PATH: str = "runs/realgame/region.json"

# Целевое соотношение сторон кадра наблюдения (16:9). Прямоугольник экрана
# приводится к нему обрезкой, иначе ресайз растянет мир и все расстояния,
# на которых обучалось зрение, поедут.
TARGET_ASPECT: float = float(OBS_W) / float(OBS_H)

_MSS_HINT = (
    "Захват экрана требует пакет mss, которого нет в базовых зависимостях "
    "GDAI. Установите его вручную: pip install mss. "
    "Напоминание: игра на живом экране — необязательный сценарий, основной "
    "путь проекта — собственный симулятор (python -m gdai watch)."
)


def require_mss() -> Any:
    """Вернуть модуль `mss` или объяснить, чего не хватает.

    Зачем отдельная функция: сообщение об отсутствующей опциональной
    зависимости должно быть одинаковым во всех точках модуля и объяснять не
    только «чего нет», но и «что делать вместо этого».
    """
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_MSS_HINT) from exc
    return mss


@dataclass(frozen=True)
class CaptureRegion:
    """Прямоугольник игрового поля на экране, в пикселях рабочего стола."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError(
                f"Прямоугольник захвата должен быть непустым, получено "
                f"{self.width}x{self.height}"
            )

    @property
    def aspect(self) -> float:
        """Соотношение сторон — по нему видно, нужна ли обрезка под 16:9."""
        return float(self.width) / float(self.height)

    @property
    def right(self) -> int:
        return int(self.left) + int(self.width)

    @property
    def bottom(self) -> int:
        return int(self.top) + int(self.height)

    def to_monitor(self) -> dict[str, int]:
        """Словарь в формате `mss` (`{"left","top","width","height"}`)."""
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    def to_dict(self) -> dict[str, int]:
        """JSON-представление калибровки (версионируем на будущее)."""
        return {"version": 1, **self.to_monitor()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureRegion":
        """Разобрать калибровку из словаря `to_dict`/`mss`-монитора."""
        try:
            return cls(
                left=int(data["left"]),
                top=int(data["top"]),
                width=int(data["width"]),
                height=int(data["height"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"В калибровке нет ключа {exc.args[0]!r}; ожидались "
                "left/top/width/height"
            ) from exc

    def fit_aspect(self, aspect: float = TARGET_ASPECT) -> "CaptureRegion":
        """Обрезать прямоугольник по центру до нужного соотношения сторон.

        Зачем не просто растянуть при ресайзе: зрение и политика обучены на
        камере 16x9 тайлов. Если сжать в 128x72 кадр другой пропорции, шип
        станет выше или шире, чем в симуляторе, и обученная сеть увидит
        объект, которого никогда не встречала.
        """
        target = float(aspect)
        if target <= 0.0:
            raise ValueError(f"aspect должен быть положительным, получено {aspect}")
        width, height = int(self.width), int(self.height)
        if abs(self.aspect - target) < 1e-6:
            return self
        if self.aspect > target:
            new_w = int(round(height * target))
            offset = (width - new_w) // 2
            return CaptureRegion(int(self.left) + offset, int(self.top), new_w, height)
        new_h = int(round(width / target))
        offset = (height - new_h) // 2
        return CaptureRegion(int(self.left), int(self.top) + offset, width, new_h)

    def save(self, path: str | os.PathLike[str] = DEFAULT_REGION_PATH) -> Path:
        """Сохранить калибровку в JSON — чтобы не выбирать прямоугольник каждый раз."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _LOG.info("калибровка сохранена: %s", target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_REGION_PATH) -> "CaptureRegion":
        """Прочитать калибровку из JSON с понятной ошибкой, если её ещё нет."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                f"Калибровка не найдена: {target}. Сначала укажите прямоугольник "
                "игрового поля: python -m gdai.realgame.capture --calibrate "
                f"--out {target}"
            )
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# ресайз
# ---------------------------------------------------------------------------
def _resize_numpy(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Ресайз ближайшим соседом на чистом numpy — запасной путь без зависимостей."""
    src_h, src_w = frame.shape[:2]
    rows = (np.arange(height) * (src_h / height)).astype(np.intp)
    cols = (np.arange(width) * (src_w / width)).astype(np.intp)
    rows = np.clip(rows, 0, src_h - 1)
    cols = np.clip(cols, 0, src_w - 1)
    return frame[rows][:, cols]


def resize_frame(
    frame: np.ndarray, width: int = OBS_W, height: int = OBS_H
) -> np.ndarray:
    """Привести кадр к размеру наблюдения (128x72) -> uint8 (height, width, 3).

    Порядок предпочтений: cv2 (INTER_AREA — правильное усреднение при сильном
    уменьшении), затем Pillow, затем numpy. Уменьшение в 10-15 раз без
    усреднения выбрасывает тонкие шипы целиком, поэтому «ближайший сосед» —
    именно запасной вариант, а не основной.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Ожидался цветной кадр (H,W,3), получено {arr.shape}")
    arr = np.ascontiguousarray(arr[:, :, :3].astype(np.uint8, copy=False))
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr

    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.resize(arr, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore[import-not-found]

        image = Image.fromarray(arr).resize((int(width), int(height)), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
    except ImportError:
        _LOG.debug("ни cv2, ни Pillow не найдены — ресайз ближайшим соседом")
    return _resize_numpy(arr, int(width), int(height))


def bgra_to_rgb(raw: np.ndarray) -> np.ndarray:
    """Кадр `mss` (BGRA) -> RGB. Отдельная функция, потому что порядок легко перепутать."""
    arr = np.asarray(raw)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Ожидался кадр (H,W,4) от mss, получено {arr.shape}")
    return np.ascontiguousarray(arr[:, :, 2::-1])


# ---------------------------------------------------------------------------
# захват
# ---------------------------------------------------------------------------
class ScreenCapture:
    """Быстрый захват прямоугольника экрана в кадр наблюдения 128x72.

    Экземпляр держит открытым соединение с сервером экрана (`mss` создаёт его
    один раз), поэтому захват в цикле стоит единицы миллисекунд, а не десятки.
    Использовать как контекстный менеджер: `with ScreenCapture(region) as cap:`.
    """

    def __init__(
        self,
        region: CaptureRegion | None = None,
        *,
        monitor: int = 1,
        fit_aspect: bool = True,
    ) -> None:
        mss = require_mss()
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if not 0 <= int(monitor) < len(monitors):
            raise ValueError(
                f"Монитор {monitor} не существует: доступны 0..{len(monitors) - 1} "
                f"(0 — все экраны разом)"
            )
        self._monitor_index = int(monitor)
        if region is None:
            region = CaptureRegion.from_dict(monitors[self._monitor_index])
            _LOG.warning(
                "Прямоугольник не задан — захватывается весь монитор %d. "
                "Для игры нужна калибровка игрового поля.",
                self._monitor_index,
            )
        self._region = region.fit_aspect() if fit_aspect else region
        self._closed = False
        _LOG.info(
            "захват экрана: %dx%d в (%d,%d)",
            self._region.width, self._region.height,
            self._region.left, self._region.top,
        )

    @property
    def region(self) -> CaptureRegion:
        """Текущий прямоугольник захвата (уже приведённый к 16:9)."""
        return self._region

    def grab_raw(self) -> np.ndarray:
        """Кадр в исходном разрешении -> uint8 (H, W, 3) RGB."""
        if self._closed:
            raise RuntimeError("Захват закрыт (close()) — снимать нечего")
        shot = self._sct.grab(self._region.to_monitor())
        return bgra_to_rgb(np.asarray(shot))

    def grab(self) -> np.ndarray:
        """Кадр наблюдения -> uint8 (OBS_H, OBS_W, 3) RGB: ровно вход зрения."""
        return resize_frame(self.grab_raw(), OBS_W, OBS_H)

    def close(self) -> None:
        """Закрыть соединение с сервером экрана; повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._sct, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # pragma: no cover - зависит от платформы
                _LOG.debug("mss не закрылся штатно: %s", exc)

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def monitor_region(monitor: int = 1) -> CaptureRegion:
    """Прямоугольник целого монитора — отправная точка для калибровки."""
    mss = require_mss()
    with mss.mss() as sct:
        monitors = sct.monitors
        if not 0 <= int(monitor) < len(monitors):
            raise ValueError(
                f"Монитор {monitor} не существует: доступны 0..{len(monitors) - 1}"
            )
        return CaptureRegion.from_dict(monitors[int(monitor)])


def region_from_values(
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    save_path: str | os.PathLike[str] | None = None,
    fit_aspect: bool = True,
) -> CaptureRegion:
    """Калибровка числами — путь для сервера, скрипта и повторяемости.

    Интерактивный выбор мышью удобен человеку, но невоспроизводим; когда
    координаты окна известны (например, игра всегда запускается в одном и том
    же положении), надёжнее задать их явно.
    """
    region = CaptureRegion(int(left), int(top), int(width), int(height))
    if fit_aspect:
        region = region.fit_aspect()
    if save_path is not None:
        region.save(save_path)
    return region


def calibrate_region(
    *,
    monitor: int = 1,
    save_path: str | os.PathLike[str] | None = DEFAULT_REGION_PATH,
) -> CaptureRegion:
    """Выбрать прямоугольник игрового поля мышью поверх снимка экрана.

    Как это работает: делается один снимок монитора, он показывается в окне
    pygame, человек протягивает мышью рамку по игровому полю (без интерфейса
    и полей), Enter — подтвердить, Esc — отменить. Результат сразу
    приводится к 16:9 и сохраняется в JSON.

    Требует и `mss`, и работающий дисплей. На машине без экрана осмысленной
    интерактивной калибровки не бывает — используйте `region_from_values`.
    """
    import sys

    if not (
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform in ("win32", "darwin")
    ):
        raise RuntimeError(
            "Интерактивная калибровка требует дисплей, а его нет. "
            "Задайте прямоугольник числами: "
            "gdai.realgame.capture.region_from_values(left, top, width, height)"
        )

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    full = monitor_region(monitor)
    with ScreenCapture(full, monitor=monitor, fit_aspect=False) as capture:
        shot = capture.grab_raw()

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((full.width, full.height))
    pygame.display.set_caption("GDAI — выделите игровое поле, Enter = ОК, Esc = отмена")
    font = pygame.font.Font(None, 22)
    background = pygame.surfarray.make_surface(
        np.ascontiguousarray(shot.transpose(1, 0, 2))
    )

    start: tuple[int, int] | None = None
    rect: tuple[int, int, int, int] | None = None
    result: CaptureRegion | None = None
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    start = event.pos
                    rect = None
                elif event.type == pygame.MOUSEMOTION and start is not None:
                    rect = _rect_from_points(start, event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and start is not None:
                    rect = _rect_from_points(start, event.pos)
                    start = None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and rect:
                        x, y, w, h = rect
                        if w > 8 and h > 8:
                            result = CaptureRegion(
                                full.left + x, full.top + y, w, h
                            ).fit_aspect()
                            running = False

            screen.blit(background, (0, 0))
            if rect:
                pygame.draw.rect(screen, (255, 80, 80), rect, 2)
                x, y, w, h = rect
                hint = f"{w}x{h} (16:9 после обрезки: {int(round(h * TARGET_ASPECT))}x{h})"
                screen.blit(font.render(hint, True, (255, 255, 255)), (x, max(0, y - 24)))
            screen.blit(
                font.render(
                    "Протяните рамку по игровому полю. Enter — подтвердить, Esc — отмена",
                    True, (255, 255, 0),
                ),
                (12, 12),
            )
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.display.quit()

    if result is None:
        raise RuntimeError("Калибровка отменена — прямоугольник не выбран")
    _LOG.info(
        "калибровка: %dx%d в (%d,%d)",
        result.width, result.height, result.left, result.top,
    )
    if save_path is not None:
        result.save(save_path)
    return result


def _rect_from_points(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    """Прямоугольник по двум углам, в каком бы порядке их ни протянули."""
    x0, y0 = a
    x1, y1 = b
    return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))


def load_region(
    path: str | os.PathLike[str] | None = DEFAULT_REGION_PATH,
) -> CaptureRegion | None:
    """Прочитать калибровку, если файл есть; иначе None (без исключения)."""
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return CaptureRegion.load(target)


def build_parser() -> Any:
    """Аргументы калибровки: `python -m gdai.realgame.capture --calibrate`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="gdai-capture",
        description="Калибровка прямоугольника игрового поля настоящей Geometry Dash",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="выбрать прямоугольник мышью"
    )
    parser.add_argument("--monitor", type=int, default=1, help="номер монитора (1 — первый)")
    parser.add_argument("--left", type=int, default=None, help="левый край поля, px")
    parser.add_argument("--top", type=int, default=None, help="верхний край поля, px")
    parser.add_argument("--width", type=int, default=None, help="ширина поля, px")
    parser.add_argument("--height", type=int, default=None, help="высота поля, px")
    parser.add_argument("--out", default=DEFAULT_REGION_PATH, help="куда сохранить JSON")
    parser.add_argument(
        "--show", action="store_true", help="показать текущую сохранённую калибровку"
    )
    return parser


def main(argv: Any = None) -> int:
    """Точка входа модуля: калибровка мышью или числами."""
    args = build_parser().parse_args(argv)
    if args.show:
        region = load_region(args.out)
        _LOG.info("калибровка %s: %s", args.out, region.to_dict() if region else "нет")
        return 0 if region else 1
    if None not in (args.left, args.top, args.width, args.height):
        region = region_from_values(
            args.left, args.top, args.width, args.height, save_path=args.out
        )
    elif args.calibrate:
        region = calibrate_region(monitor=args.monitor, save_path=args.out)
    else:
        build_parser().print_help()
        return 2
    _LOG.info("готово: %s", region.to_dict())
    return 0


__all__ = [
    "CaptureRegion",
    "ScreenCapture",
    "require_mss",
    "resize_frame",
    "bgra_to_rgb",
    "monitor_region",
    "region_from_values",
    "calibrate_region",
    "load_region",
    "build_parser",
    "main",
    "DEFAULT_REGION_PATH",
    "TARGET_ASPECT",
]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
