"""Игра в настоящую Geometry Dash: захват -> зрение -> политика -> пробел (SPEC §14).

ВНИМАНИЕ, это необязательный и заведомо хрупкий сценарий.

* Он требует **ручной калибровки** прямоугольника игрового поля
  (`gdai.realgame.capture`) — без неё карта будет бессмысленной.
* Он требует пакетов `mss` и `pynput`, которых нет в зависимостях проекта.
* Он *эмулирует нажатия клавиш*: запускайте только тогда, когда активно окно
  игры, и держите наготове аварийную клавишу (по умолчанию Esc), которая
  немедленно отпускает пробел и останавливает цикл.
* Отличий от симулятора всегда останется много: у настоящей игры другая
  камера, свои эффекты и задержка ввода порядка десятков миллисекунд.

Основной сценарий проекта — собственный симулятор (`python -m gdai watch`),
на нём и проверяется, что архитектура работает. Этот модуль — демонстрация
того, что зрение переносится на чужие пиксели, а не боевой инструмент.

Для отладки без игры и без зависимостей есть `dry_run=True` (никаких нажатий)
и возможность подставить свои `capture`/`agent`/`presser` — тогда цикл
проверяется целиком на симуляторе.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from gdai.constants import (
    ACTION_HOLD,
    DEFAULT_SPEED_INDEX,
    MAX_FALL_V,
    OBS_H,
    OBS_W,
    SPEEDS,
)
from gdai.realgame.capture import (
    DEFAULT_REGION_PATH,
    CaptureRegion,
    ScreenCapture,
    load_region,
)
from gdai.utils.logging import get_logger

_LOG = get_logger("realgame.play")

# Размерность вектора признаков среды (SPEC §8). Дублируется числом, а не
# импортом из gd_env, чтобы игра на живом экране не тянула весь симулятор.
FEATURE_DIM: int = 8

_PYNPUT_HINT = (
    "Управление клавиатурой требует пакет pynput, которого нет в базовых "
    "зависимостях GDAI. Установите его вручную: pip install pynput. "
    "Без него доступен только режим наблюдения (dry_run=True): агент считает "
    "действия и пишет их в лог, но ничего не нажимает."
)


def require_pynput() -> Any:
    """Вернуть модуль `pynput.keyboard` или объяснить, чего не хватает."""
    try:
        from pynput import keyboard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_PYNPUT_HINT) from exc
    return keyboard


@dataclass
class RealGameConfig:
    """Настройки игры на живом экране. Значения по умолчанию — самые безопасные."""

    policy_path: str | None = None          # веса политики
    perception_path: str | None = None      # веса зрения (без них смысла мало)
    region_path: str | None = DEFAULT_REGION_PATH   # калибровка игрового поля
    monitor: int = 1                        # какой монитор захватывать
    fps: float = 60.0                       # частота цикла (игра идёт на 60)
    max_seconds: float = 120.0              # жёсткий предел прогона
    device: str = "auto"
    deterministic: bool = True              # argmax: случайный прыжок = смерть
    dry_run: bool = True                    # НЕ нажимать клавиши, только считать
    quit_key: str = "esc"                   # аварийный выход
    hold_key: str = "space"                 # чем прыгаем
    speed_index: int = DEFAULT_SPEED_INDEX  # скорость уровня для вектора признаков
    log_every: float = 2.0                  # как часто писать строку статистики


# ---------------------------------------------------------------------------
# клавиатура
# ---------------------------------------------------------------------------
class KeyHolder:
    """Удержание пробела: нажать один раз, отпустить один раз.

    Зачем состояние, а не «нажать-отпустить каждый кадр»: в Geometry Dash
    удержание — самостоятельное действие (корабль летит вверх, пока держишь),
    и дробить его на 60 нажатий в секунду значит играть совсем в другую игру.
    """

    def __init__(self, hold_key: str = "space", dry_run: bool = True) -> None:
        self.dry_run = bool(dry_run)
        self._held = False
        self._controller: Any = None
        self._key: Any = None
        if not self.dry_run:
            keyboard = require_pynput()
            self._controller = keyboard.Controller()
            self._key = getattr(keyboard.Key, str(hold_key), None)
            if self._key is None:
                # Обычный символ (например "w") — тоже допустимая кнопка прыжка.
                self._key = str(hold_key)[:1]

    @property
    def held(self) -> bool:
        """Держится ли кнопка прямо сейчас."""
        return self._held

    def set_hold(self, hold: bool) -> None:
        """Привести состояние кнопки к нужному; лишних событий не шлём."""
        hold = bool(hold)
        if hold == self._held:
            return
        self._held = hold
        if self.dry_run or self._controller is None:
            return
        if hold:
            self._controller.press(self._key)
        else:
            self._controller.release(self._key)

    def release(self) -> None:
        """Гарантированно отпустить кнопку — обязательный шаг при любом выходе."""
        self.set_hold(False)

    def close(self) -> None:
        """Отпустить кнопку и забыть контроллер."""
        try:
            self.release()
        finally:
            self._controller = None


class EmergencyStop:
    """Слушатель аварийной клавиши: единственный способ прервать цикл руками.

    Пока агент играет, клавиатура занята эмулированными нажатиями, а окно
    игры — на переднем плане. Без глобального слушателя остановить процесс
    было бы нечем, поэтому он поднимается ДО первого нажатия и снимается
    только в `finally`.
    """

    def __init__(self, quit_key: str = "esc", enabled: bool = True) -> None:
        self.quit_key = str(quit_key)
        self.enabled = bool(enabled)
        self._triggered = False
        self._listener: Any = None

    @property
    def triggered(self) -> bool:
        """Была ли нажата аварийная клавиша."""
        return self._triggered

    def trigger(self) -> None:
        """Взвести флаг вручную (используется в тестах и по таймауту)."""
        self._triggered = True

    def start(self) -> None:
        """Запустить фоновый слушатель; без pynput тихо работает как заглушка."""
        if not self.enabled or self._listener is not None:
            return
        try:
            keyboard = require_pynput()
        except ImportError as exc:
            _LOG.warning("аварийная клавиша недоступна: %s", exc)
            return
        target = getattr(keyboard.Key, self.quit_key, None)

        def on_press(key: Any) -> None:
            if target is not None and key == target:
                self._triggered = True
            elif getattr(key, "char", None) == self.quit_key:
                self._triggered = True

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        _LOG.info("аварийный выход: клавиша %s", self.quit_key)

    def stop(self) -> None:
        """Снять слушатель; повторный вызов безопасен."""
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:  # pragma: no cover - зависит от платформы
                _LOG.debug("слушатель клавиш не остановился штатно: %s", exc)

    def __enter__(self) -> "EmergencyStop":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# признаки
# ---------------------------------------------------------------------------
def estimate_features(
    *,
    progress: float = 0.0,
    vy: float = 0.0,
    on_ground: bool = True,
    gravity: int = 1,
    mode: str = "cube",
    speed_index: int = DEFAULT_SPEED_INDEX,
) -> np.ndarray:
    """Собрать вектор признаков (FEATURE_DIM) для политики по догадкам.

    Честное предупреждение: в симуляторе этот вектор берётся из состояния
    игрока, а на живом экране его взять неоткуда — вертикальной скорости и
    флага «на земле» в кадре нет. Здесь он собирается из грубых оценок
    (режим считается кубом, гравитация обычной, прогресс — по времени), и это
    главная причина, по которой агент играет в настоящей игре хуже, чем в
    симуляторе. Порядок полей строго как в `gdai.env.gd_env` (SPEC §8).
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[0] = float(np.clip(vy / MAX_FALL_V, -1.0, 1.0))
    feat[1] = 1.0 if on_ground else 0.0
    feat[2] = float(gravity)
    feat[3] = 1.0 if mode == "cube" else 0.0
    feat[4] = 1.0 if mode == "ship" else 0.0
    feat[5] = 1.0 if mode == "wave" else 0.0
    feat[6] = float(int(speed_index)) / float(len(SPEEDS) - 1)
    feat[7] = float(np.clip(progress, 0.0, 1.0))
    return feat


# ---------------------------------------------------------------------------
# агент
# ---------------------------------------------------------------------------
def load_agent(config: RealGameConfig) -> Any:
    """Поднять `gdai.pipeline.GDAgent` по контракту SPEC §12.

    Импорт ленивый и с понятным сообщением: игра на живом экране — крайний
    сценарий, и падать с `ModuleNotFoundError` в середине запуска нельзя.
    """
    try:
        from gdai.pipeline import GDAgent
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать gdai.pipeline.GDAgent — полный агент "
            f"(зрение + политика) недоступен: {exc}. Передайте готовый агент "
            "аргументом agent=... или соберите связку сами "
            "(gdai.perception.model.load_perception_net + gdai.agent.ppo.load_policy)."
        ) from exc
    return GDAgent(
        policy_path=config.policy_path,
        perception_path=config.perception_path,
        device=config.device,
        use_perception=True,
    )


# ---------------------------------------------------------------------------
# основной цикл
# ---------------------------------------------------------------------------
def play_real(
    config: RealGameConfig | None = None,
    *,
    agent: Any = None,
    capture: Any = None,
    presser: KeyHolder | None = None,
    stopper: EmergencyStop | None = None,
    on_frame: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Цикл «захват -> зрение -> политика -> пробел» с ограничением частоты.

    Возвращает статистику прогона: сколько кадров, сколько реального времени,
    какая доля кадров с удержанием, средний FPS и по какой причине вышли.

    Все внешние части (захват, агент, нажатия, аварийный выход) можно
    подставить своими объектами — так цикл тестируется целиком без mss,
    pynput и запущенной игры.
    """
    cfg = config if config is not None else RealGameConfig()
    period = 1.0 / max(1e-3, float(cfg.fps))

    own_capture = capture is None
    if own_capture:
        region: CaptureRegion | None = load_region(cfg.region_path)
        if region is None:
            raise FileNotFoundError(
                f"Нет калибровки игрового поля ({cfg.region_path}). Сначала "
                "выделите прямоугольник: python -m gdai.realgame.capture "
                f"--calibrate --out {cfg.region_path}"
            )
        capture = ScreenCapture(region, monitor=cfg.monitor)

    own_agent = agent is None
    if own_agent:
        agent = load_agent(cfg)

    own_presser = presser is None
    if own_presser:
        presser = KeyHolder(hold_key=cfg.hold_key, dry_run=bool(cfg.dry_run))
    own_stopper = stopper is None
    if own_stopper:
        stopper = EmergencyStop(cfg.quit_key, enabled=not cfg.dry_run)

    if cfg.dry_run:
        _LOG.warning(
            "dry_run: клавиши НЕ нажимаются, агент только считает действия"
        )

    frames = 0
    holds = 0
    reason = "max_seconds"
    started = time.perf_counter()
    next_frame = started
    last_log = started

    stopper.start()
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            if elapsed >= float(cfg.max_seconds):
                reason = "max_seconds"
                break
            if stopper.triggered:
                reason = "quit_key"
                break

            frame = np.asarray(capture.grab(), dtype=np.uint8)
            if frame.shape[:2] != (OBS_H, OBS_W):
                raise ValueError(
                    f"Захват вернул кадр {frame.shape[:2]}, а зрение ждёт "
                    f"({OBS_H}, {OBS_W}) — проверьте калибровку"
                )

            sem = np.asarray(agent.see(frame), dtype=np.uint8)
            features = estimate_features(
                progress=min(1.0, elapsed / max(1e-6, float(cfg.max_seconds))),
                speed_index=int(cfg.speed_index),
            )
            action = int(
                agent.act(
                    {"semantic": sem, "pixels": frame, "features": features},
                    deterministic=bool(cfg.deterministic),
                )
            )
            hold = action == ACTION_HOLD
            presser.set_hold(hold)

            frames += 1
            holds += int(hold)
            if on_frame is not None:
                on_frame(
                    {
                        "frame": frame,
                        "semantic": sem,
                        "features": features,
                        "action": action,
                        "elapsed": elapsed,
                        "index": frames,
                    }
                )

            if cfg.log_every > 0 and now - last_log >= float(cfg.log_every):
                last_log = now
                _LOG.info(
                    "кадров %d, %.1f FPS, удержание %.0f%%",
                    frames, frames / max(1e-6, elapsed), 100.0 * holds / max(1, frames),
                )

            # Ограничение частоты по абсолютным меткам: sleep(period) копил бы
            # отставание, а игра не ждёт — она идёт ровно 60 кадров в секунду.
            next_frame += period
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.perf_counter()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        # Порядок важен: сначала отпустить кнопку (иначе игра останется с
        # зажатым пробелом), и только потом закрывать всё остальное.
        presser.release()
        if own_presser:
            presser.close()
        if own_stopper:
            stopper.stop()
        if own_capture:
            closer = getattr(capture, "close", None)
            if callable(closer):
                closer()

    total = max(1e-6, time.perf_counter() - started)
    stats = {
        "frames": frames,
        "seconds": round(total, 3),
        "mean_fps": round(frames / total, 2),
        "hold_fraction": round(holds / max(1, frames), 4),
        "stopped_by": reason,
        "dry_run": bool(cfg.dry_run),
    }
    _LOG.info("итог: %s", stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Аргументы команды `python -m gdai play-real`."""
    parser = argparse.ArgumentParser(
        prog="gdai-play-real",
        description=(
            "Играть в настоящую Geometry Dash (нужны mss и pynput, "
            "требуется ручная калибровка игрового поля)"
        ),
    )
    parser.add_argument("--policy", default=None, help="веса политики")
    parser.add_argument("--perception", default=None, help="веса зрения")
    parser.add_argument(
        "--region", default=DEFAULT_REGION_PATH, help="файл калибровки игрового поля"
    )
    parser.add_argument("--monitor", type=int, default=1, help="номер монитора")
    parser.add_argument("--fps", type=float, default=60.0, help="частота цикла")
    parser.add_argument(
        "--seconds", type=float, default=120.0, help="сколько секунд играть"
    )
    parser.add_argument(
        "--press",
        action="store_true",
        help="РАЗРЕШИТЬ нажатия клавиш (по умолчанию только наблюдение)",
    )
    parser.add_argument("--quit-key", default="esc", help="клавиша аварийного выхода")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument(
        "--stochastic", action="store_true", help="выбирать действие выборкой"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: `python -m gdai.realgame.play --policy ... --press`."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = RealGameConfig(
        policy_path=args.policy,
        perception_path=args.perception,
        region_path=args.region,
        monitor=int(args.monitor),
        fps=float(args.fps),
        max_seconds=float(args.seconds),
        device=args.device,
        deterministic=not args.stochastic,
        dry_run=not args.press,
        quit_key=args.quit_key,
    )
    stats = play_real(cfg)
    _LOG.info("готово: %s", stats)
    return 0


__all__ = [
    "RealGameConfig",
    "KeyHolder",
    "EmergencyStop",
    "play_real",
    "load_agent",
    "estimate_features",
    "require_pynput",
    "build_parser",
    "main",
    "FEATURE_DIM",
]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
