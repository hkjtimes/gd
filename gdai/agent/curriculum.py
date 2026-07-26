"""Учебный план: сложность растёт только тогда, когда агент реально справляется.

Зачем это нужно
---------------
Сложность 0.8 для необученного агента — это разреженная награда: он умирает на
третьей секунде, никогда не видит финиша, и единственное, чему может научиться,
— это «умирать чуть позже». Начав с почти пустой дорожки, агент получает
плотный сигнал (шипы одиночные, окна широкие), выучивает базовый прыжок, и
только затем ему поднимают ставку. Каждая следующая ступень опирается на уже
работающий навык, а не начинается с нуля.

Как принимается решение
-----------------------
Ступень повышается, когда доля пройденных ЦЕЛИКОМ уровней в скользящем окне
последних `window` эпизодов превышает порог. Окно, а не «всё среднее за
прогон»: среднее по всей истории тормозит на порядок дольше, чем меняется
политика, и агент застревал бы на ступени, которую перерос ещё сто тысяч шагов
назад. После повышения окно очищается — старые эпизоды относятся к другой
задаче и голосовать за следующее повышение не имеют права.

Про practice-чекпойнты
----------------------
Второй механизм борьбы с разреженной наградой живёт в среде
(`EnvConfig.practice_checkpoints`): после смерти эпизод с вероятностью
`checkpoint_prob` начинается с последнего пройденного чекпойнта, а не с начала
уровня. Учебному плану важно НЕ считать такие эпизоды за успех: пройти уровень
с середины — не то же самое, что пройти его целиком, и `record_episode` для них
вызывать не следует (за это отвечает `ppo.py`). Здесь же есть
`practice_probability`, чтобы доля практики падала по мере роста мастерства.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

from gdai.config import CurriculumConfig
from gdai.utils.logging import get_logger

_log = get_logger("agent.curriculum")


class Curriculum:
    """Скользящее окно исходов эпизодов и текущая ступень сложности."""

    def __init__(self, cfg: CurriculumConfig | None = None) -> None:
        self.cfg: CurriculumConfig = cfg if cfg is not None else CurriculumConfig()
        if self.cfg.window <= 0:
            raise ValueError(f"window должно быть >= 1, получено {self.cfg.window}")
        self._difficulty: float = self._clamp(self.cfg.start_difficulty)
        self._window: Deque[bool] = deque(maxlen=int(self.cfg.window))
        self._episodes: int = 0        # всего засчитанных эпизодов
        self._finished: int = 0        # из них пройденных целиком
        self._promotions: int = 0      # сколько раз ступень поднималась

    # -- служебное ----------------------------------------------------------
    def _clamp(self, value: float) -> float:
        """Загнать сложность в допустимый диапазон плана."""
        lo = 0.0
        hi = float(self.cfg.max_difficulty)
        v = float(value)
        return lo if v < lo else hi if v > hi else v

    # -- состояние ----------------------------------------------------------
    def current_difficulty(self) -> float:
        """Сложность, которую надо выставить средам прямо сейчас."""
        return self._difficulty

    def success_rate(self) -> float:
        """Доля пройденных уровней в окне (0.0, пока эпизодов не было)."""
        if not self._window:
            return 0.0
        return sum(1 for ok in self._window if ok) / len(self._window)

    @property
    def window_size(self) -> int:
        """Сколько эпизодов уже накоплено в окне (для отчётности)."""
        return len(self._window)

    @property
    def promotions(self) -> int:
        """Сколько ступеней пройдено с начала обучения."""
        return self._promotions

    @property
    def at_max(self) -> bool:
        """Достигнут ли потолок сложности — дальше расти некуда."""
        return self._difficulty >= float(self.cfg.max_difficulty) - 1e-9

    # -- обновление ---------------------------------------------------------
    def record_episode(self, finished: bool) -> None:
        """Учесть исход эпизода, сыгранного С НАЧАЛА уровня.

        `finished=True` — агент дошёл до финиша. Эпизоды практики (старт с
        чекпойнта) сюда подавать нельзя: они завышают оценку и толкают план
        вперёд раньше времени.
        """
        ok = bool(finished)
        self._window.append(ok)
        self._episodes += 1
        self._finished += int(ok)

    def maybe_promote(self) -> bool:
        """Поднять сложность, если окно набрано и порог успеха превышен.

        Возвращает True, если ступень изменилась, — вызывающий обязан передать
        новую сложность средам (`SyncVectorEnv.set_difficulty`).

        Полное окно — обязательное условие: по трём эпизодам «100% успеха»
        означает только удачу, и план ускакал бы в максимум за пару итераций.
        """
        if self.at_max:
            return False
        if len(self._window) < self._window.maxlen:  # type: ignore[operator]
            return False
        if self.success_rate() <= float(self.cfg.promote_success_rate):
            return False
        previous = self._difficulty
        self._difficulty = self._clamp(previous + float(self.cfg.step))
        if self._difficulty <= previous:
            return False
        self._promotions += 1
        # Окно очищается: прошлые эпизоды играли на предыдущей сложности и к
        # новой задаче отношения не имеют.
        self._window.clear()
        _log.info(
            "сложность повышена %.2f -> %.2f (ступень %d)",
            previous,
            self._difficulty,
            self._promotions,
        )
        return True

    def practice_probability(self, base: float) -> float:
        """Насколько часто стоит стартовать с чекпойнта на текущей ступени.

        Практика — костыль против разреженной награды, и на лёгких ступенях он
        не нужен, а на тяжёлых незаменим. Линейно поднимаем долю практики от
        нуля (пустая дорожка) до `base` (максимальная сложность), чтобы агент
        не привыкал начинать с середины там, где может пройти уровень целиком.
        """
        top = float(self.cfg.max_difficulty)
        share = 1.0 if top <= 0.0 else min(max(self._difficulty / top, 0.0), 1.0)
        return float(base) * share

    # -- сериализация -------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Снимок плана для чекпойнта: без него дообучение начнёт ступень заново."""
        return {
            "difficulty": self._difficulty,
            "window": list(self._window),
            "episodes": self._episodes,
            "finished": self._finished,
            "promotions": self._promotions,
            "config": {
                "start_difficulty": self.cfg.start_difficulty,
                "max_difficulty": self.cfg.max_difficulty,
                "step": self.cfg.step,
                "promote_success_rate": self.cfg.promote_success_rate,
                "window": self.cfg.window,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Восстановить план из снимка (конфиг остаётся текущим — он мог поменяться)."""
        self._difficulty = self._clamp(float(state.get("difficulty", self._difficulty)))
        window = [bool(x) for x in state.get("window", [])]
        self._window = deque(window[-int(self.cfg.window):], maxlen=int(self.cfg.window))
        self._episodes = int(state.get("episodes", 0))
        self._finished = int(state.get("finished", 0))
        self._promotions = int(state.get("promotions", 0))

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return (
            f"Curriculum(difficulty={self._difficulty:.2f}, "
            f"success_rate={self.success_rate():.2f}, "
            f"window={len(self._window)}/{self.cfg.window})"
        )


__all__ = ["Curriculum"]
