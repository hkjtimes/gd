"""Окно «что видит ИИ»: три панели, HUD и запись демонстрации (SPEC §13).

Зачем этот модуль вообще существует: главный аргумент архитектуры GDAI —
разделение «зрение / политика» — невозможно проверить на глаз по числам в
логах. Здесь он показывается буквально: слева живой кадр с любыми
декорациями, в центре — каноническая карта, в которую его превратило зрение,
справа — эталонная карта и подсвеченные ошибки. Если правая панель почти
чёрная, зрение работает; если политика проходит уровень, глядя только на
центральную панель, значит она действительно не смотрит на дизайн.

Модуль обязан работать в двух режимах:

* интерактивно (`run_viewer`) — окно pygame, горячие клавиши из SPEC §13;
* headless (`record_demo`) — ни одного окна, кадры пишутся в PNG и, если в
  системе есть `imageio`/`cv2`/`PIL`, собираются в GIF или MP4.

Второй режим — не «запасной»: именно он используется на сервере без дисплея,
в CI и для картинок в README, поэтому вся отрисовка идёт в обычную
`pygame.Surface` и ничего не знает про наличие экрана.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gdai.config import EnvConfig
from gdai.constants import (
    ACTION_HOLD,
    ACTION_NONE,
    NUM_ACTIONS,
    OBS_H,
    OBS_W,
)
from gdai.utils.logging import get_logger
from gdai.utils.seeding import make_rng, seed_from

_LOG = get_logger("viz.viewer")


# ---------------------------------------------------------------------------
# headless-инициализация SDL до импорта pygame
# ---------------------------------------------------------------------------
def _prepare_sdl() -> None:
    """Включить dummy-драйвер SDL, если дисплея нет.

    Зачем строго до `import pygame`: SDL читает переменные окружения при
    инициализации, и выставленный позже `SDL_VIDEODRIVER` уже ничего не меняет.
    Без этого `record_demo` на сервере без X падал бы на первой же попытке
    создать поверхность.
    """
    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform in ("win32", "darwin")
    )
    if not has_display:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


_prepare_sdl()

try:
    import pygame
except ImportError as exc:  # pragma: no cover - зависит от окружения
    raise ImportError(
        "Для окна визуализации нужен pygame. Установите его: pip install pygame>=2.1"
    ) from exc

from gdai.env.gd_env import GeometryDashEnv  # noqa: E402  (после _prepare_sdl)
from gdai.env.render import Renderer  # noqa: E402
from gdai.env.semantic import semantic_to_rgb  # noqa: E402
from gdai.env.themes import Theme, random_theme, theme_by_name, theme_names  # noqa: E402


# ---------------------------------------------------------------------------
# внешний вид
# ---------------------------------------------------------------------------
PANEL_GAP: int = 8          # отступ между панелями и от краёв окна, px
LABEL_H: int = 18           # полоса подписи над панелью, px
HUD_H: int = 70             # нижняя панель телеметрии, px

BG_COLOR: tuple[int, int, int] = (16, 18, 24)
PANEL_EDGE: tuple[int, int, int] = (58, 64, 78)
TEXT_COLOR: tuple[int, int, int] = (228, 234, 242)
DIM_COLOR: tuple[int, int, int] = (144, 154, 170)
ACCENT_COLOR: tuple[int, int, int] = (90, 200, 255)
GOOD_COLOR: tuple[int, int, int] = (120, 230, 140)
BAD_COLOR: tuple[int, int, int] = (255, 92, 92)
BAR_BG: tuple[int, int, int] = (38, 42, 52)
# Цвет, которым на третьей панели подсвечиваются пиксели, где зрение ошиблось.
DIFF_COLOR: tuple[int, int, int] = (255, 48, 48)
# Насколько гасится эталон под разницей: ошибки должны бросаться в глаза.
DIFF_DIM: float = 0.55

# Ступени уровня декора по клавише D.
DECOR_STEPS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
# Сколько кадров показывать итог эпизода (СМЕРТЬ/ФИНИШ) до авто-рестарта.
FREEZE_FRAMES: int = 12

HOTKEYS: tuple[tuple[str, str], ...] = (
    ("Space", "играть самому (удержание = прыжок)"),
    ("A", "отдать управление ИИ"),
    ("T", "сменить тему оформления"),
    ("D", "уровень декора"),
    ("P", "зрение: эталон <-> предсказание сети"),
    ("S", "карта внимания политики"),
    ("R", "рестарт уровня"),
    ("N", "новый уровень"),
    ("H", "показать/скрыть подсказку"),
    ("Esc", "выход"),
)


@dataclass
class ViewerConfig:
    """Настройки просмотра. Отдельный dataclass — чтобы CLI и тесты собирали одно и то же."""

    policy_path: str | None = None        # веса политики (runs/agent/best.pt)
    perception_path: str | None = None    # веса зрения (runs/perception/best.pt)
    level_path: str | None = None         # фиксированный уровень из файла
    difficulty: float = 0.3               # сложность процедурной генерации
    seed: int | None = 0                  # seed уровня/темы: демо должно повторяться
    scale: int = 3                        # во сколько раз увеличить панель 128x72
    fps: int = 60                         # частота интерактивного окна
    use_perception: bool = True           # показывать предсказание, а не эталон
    ai_control: bool = True               # кто играет: ИИ или человек
    deterministic: bool = True            # argmax вместо выборки действия
    decoration_level: float = 1.0         # стартовая плотность декора
    randomize_theme: bool = True          # новая тема на каждый новый уровень
    theme: str | None = None              # фиксированная тема по имени
    practice_checkpoints: bool = False    # в просмотре уровень честно начинается с нуля
    max_steps: int = 6000
    device: str = "auto"
    show_saliency: bool = False           # наложить карту внимания на панель зрения
    show_help: bool = False               # подсказка по горячим клавишам поверх кадра


# ---------------------------------------------------------------------------
# «мозг»: зрение + политика
# ---------------------------------------------------------------------------
class _Brain:
    """Зрение и политика для просмотра — обе части необязательны.

    Зачем не требовать обученных весов: смотреть «что видит ИИ» полезно и до
    обучения (проверить связку env -> карта -> сеть), поэтому при отсутствии
    файлов создаются сети со случайной инициализацией, а HUD честно пишет
    «случайная». Так демонстрация никогда не падает из-за отсутствия чекпойнта.

    Внешний агент (`gdai.pipeline.GDAgent`, SPEC §12) можно передать готовым —
    тогда используются его `see`/`act`, а вероятности берутся из его политики,
    если он её отдаёт атрибутом `policy`.
    """

    def __init__(
        self,
        *,
        policy_path: str | None = None,
        perception_path: str | None = None,
        device: str = "auto",
        use_perception: bool = True,
        agent: Any = None,
    ) -> None:
        import torch  # ленивый импорт: без ИИ модуль должен подниматься и без torch

        self._torch = torch
        self._agent = agent
        self._device = self._resolve_device(device)
        self.policy_source: str = "случайная"
        self.perception_source: str = "нет"

        self._policy = self._load_policy(policy_path)
        self._perception = self._load_perception(perception_path, use_perception)

        if agent is not None:
            # У готового агента приоритет: он уже собрал связку по своим правилам.
            agent_policy = getattr(agent, "policy", None)
            if agent_policy is not None:
                self._policy = agent_policy
                self.policy_source = "GDAgent"
            if getattr(agent, "see", None) is not None:
                self.perception_source = "GDAgent"

    # -- загрузка ----------------------------------------------------------
    def _resolve_device(self, name: str) -> Any:
        torch = self._torch
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _load_policy(self, path: str | None) -> Any:
        """Политика из чекпойнта или случайная сеть той же архитектуры."""
        from gdai.agent.networks import ActorCritic

        if path:
            from gdai.agent.ppo import load_policy

            model = load_policy(path, device=str(self._device))
            self.policy_source = Path(path).name
            _LOG.info("политика загружена: %s", path)
        else:
            model = ActorCritic().to(self._device)
            _LOG.info("политика не задана — используется случайная инициализация")
        model.eval()
        return model

    def _load_perception(self, path: str | None, use_perception: bool) -> Any:
        """Зрение из чекпойнта; без файла — случайная сеть (её ошибки тоже наглядны)."""
        if not use_perception:
            return None
        from gdai.perception.model import PerceptionNet, load_perception_net

        if path:
            model = load_perception_net(path, device=str(self._device))
            self.perception_source = Path(path).name
            _LOG.info("зрение загружено: %s", path)
        else:
            model = PerceptionNet().to(self._device)
            self.perception_source = "случайная"
            _LOG.info("зрение не задано — используется случайная инициализация")
        model.eval()
        return model

    # -- работа ------------------------------------------------------------
    @property
    def has_perception(self) -> bool:
        """Есть ли чем предсказывать карту (иначе панель показывает эталон)."""
        return self._perception is not None or getattr(self._agent, "see", None) is not None

    @property
    def policy(self) -> Any:
        """Сама сеть — нужна модулю saliency для градиентов."""
        return self._policy

    @property
    def device(self) -> Any:
        """Устройство, на котором считается ИИ."""
        return self._device

    def see(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Кадр (H,W,3) -> карта классов (H,W). Это и есть «что видит ИИ»."""
        agent_see = getattr(self._agent, "see", None)
        if callable(agent_see):
            return np.asarray(agent_see(frame_rgb), dtype=np.uint8)
        if self._perception is None:
            raise RuntimeError("Зрение выключено — предсказывать карту нечем")
        return np.asarray(self._perception.predict(frame_rgb), dtype=np.uint8)

    def decide(
        self, sem: np.ndarray, features: np.ndarray, deterministic: bool = True
    ) -> tuple[int, float, float]:
        """Решение политики по карте: `(действие, P(hold), value)`.

        Вероятность и ценность нужны не сети, а человеку: по ним сразу видно,
        уверен ли агент перед шипом и считает ли он положение выигрышным.
        """
        torch = self._torch
        from gdai.agent.networks import semantic_to_tensor

        with torch.no_grad():
            sem_t = semantic_to_tensor(sem, device=self._device)
            feat_t = torch.as_tensor(
                np.asarray(features, dtype=np.float32).reshape(1, -1)
            ).to(self._device)
            logits, value = self._policy(sem_t, feat_t)
            probs = torch.softmax(logits, dim=-1)[0]
            p_hold = float(probs[ACTION_HOLD]) if NUM_ACTIONS > 1 else 1.0
            v = float(value.reshape(-1)[0])
            if deterministic:
                action = int(torch.argmax(logits, dim=-1)[0])
            else:
                action = int(torch.multinomial(probs, num_samples=1)[0])
        return action, p_hold, v


# ---------------------------------------------------------------------------
# сборка кадра визуализации
# ---------------------------------------------------------------------------
class Viewer:
    """Три панели + HUD поверх среды `GeometryDashEnv`.

    Один и тот же объект обслуживает интерактивный режим (`run`) и запись
    демонстрации (`record`): различаются они только источником действий
    человека и тем, куда уходит готовая поверхность — на экран или в PNG.
    """

    def __init__(
        self,
        config: ViewerConfig | None = None,
        *,
        agent: Any = None,
        env: GeometryDashEnv | None = None,
    ) -> None:
        self.config = config if config is not None else ViewerConfig()
        cfg = self.config
        if int(cfg.scale) < 1:
            raise ValueError(f"scale должен быть >= 1, получено {cfg.scale}")

        self.scale = int(cfg.scale)
        self._pw = OBS_W * self.scale
        self._ph = OBS_H * self.scale
        self.width = PANEL_GAP + 3 * (self._pw + PANEL_GAP)
        self.height = PANEL_GAP + LABEL_H + self._ph + HUD_H

        self._rng = make_rng(cfg.seed)
        self._theme_index = 0
        self._decor_index = _nearest_decor_index(cfg.decoration_level)

        # Рендерер создаём сами и отдаём среде: только так вьюер может менять
        # тему по клавише T (иначе среда сбрасывала бы её на каждом reset).
        self._renderer = Renderer(
            width=OBS_W,
            height=OBS_H,
            decoration_level=DECOR_STEPS[self._decor_index],
            seed=seed_from("viz.viewer", cfg.seed),
        )
        if cfg.theme:
            self._renderer.set_theme(theme_by_name(cfg.theme))

        if env is None:
            env_cfg = EnvConfig(
                obs_mode="both",
                max_steps=int(cfg.max_steps),
                difficulty=float(cfg.difficulty),
                level_path=cfg.level_path,
                seed=cfg.seed,
                randomize_theme=False,       # темой управляет вьюер, а не среда
                decoration_level=DECOR_STEPS[self._decor_index],
                practice_checkpoints=bool(cfg.practice_checkpoints),
            )
            env = GeometryDashEnv(env_cfg, renderer=self._renderer)
            self._owns_env = True
        else:
            self._owns_env = False
        self.env = env

        self._brain = _Brain(
            policy_path=cfg.policy_path,
            perception_path=cfg.perception_path,
            device=cfg.device,
            use_perception=cfg.use_perception,
            agent=agent,
        )

        # Инициализируем только шрифты: `pygame.init()` поднял бы ещё и звук,
        # который в headless-записи не нужен и сыплет предупреждениями ALSA.
        pygame.font.init()
        self._font = pygame.font.Font(None, 18)
        self._font_small = pygame.font.Font(None, 15)
        self._font_big = pygame.font.Font(None, int(28 + 6 * self.scale))
        self._surface = pygame.Surface((self.width, self.height))

        # Состояние просмотра.
        self.ai_control: bool = bool(cfg.ai_control)
        self.vision_on: bool = bool(cfg.use_perception) and self._brain.has_perception
        self.show_saliency: bool = bool(cfg.show_saliency)
        self.show_help: bool = bool(cfg.show_help)
        self.running: bool = True

        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_tick: float = time.perf_counter()
        self._freeze: int = 0
        self._episode: int = 0
        self._outcome: str = ""
        self._last_action: int = ACTION_NONE
        self._view: dict[str, Any] = {}
        self._closed = False

        self.reset(new_level=True)

    # -- жизненный цикл ----------------------------------------------------
    def reset(self, new_level: bool = False) -> None:
        """Начать эпизод. `new_level=True` — заодно сгенерировать другой уровень.

        Тема меняется только вместе с уровнем: иначе картинка прыгала бы при
        каждой смерти, и сравнивать поведение агента было бы невозможно.
        """
        if new_level:
            obs, info = self.env.reset()
            if self.config.randomize_theme and not self.config.theme:
                self._renderer.randomize(self._rng)
        else:
            obs, info = self.env.reset(level=self.env.level)
        self._episode += 1
        self._outcome = ""
        self._freeze = 0
        self._last_action = ACTION_NONE
        self._steps = 0
        self._reward = 0.0
        self._obs = obs
        self._info = info
        self._observe()

    def close(self) -> None:
        """Освободить среду и pygame; повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        if self._owns_env:
            try:
                self.env.close()
            except Exception as exc:  # pragma: no cover - зависит от среды
                _LOG.debug("среда не закрылась штатно: %s", exc)
        try:
            pygame.display.quit()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "Viewer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- переключатели (горячие клавиши) -----------------------------------
    def set_ai_control(self, ai: bool) -> None:
        """Кто нажимает кнопку: ИИ или человек (SPEC §13, клавиши A и Space)."""
        self.ai_control = bool(ai)

    def cycle_theme(self) -> None:
        """Следующая тема: сначала все встроенные, затем полностью случайная."""
        names = theme_names()
        self._theme_index = (self._theme_index + 1) % (len(names) + 1)
        theme: Theme
        if self._theme_index == len(names):
            theme = random_theme(self._rng)
        else:
            theme = theme_by_name(names[self._theme_index])
        self._renderer.set_theme(theme)
        self._observe()

    def cycle_decoration(self) -> None:
        """Следующая ступень плотности декора — видно, что карта от неё не зависит."""
        self._decor_index = (self._decor_index + 1) % len(DECOR_STEPS)
        self._renderer.set_decoration_level(DECOR_STEPS[self._decor_index])
        self._observe()

    def toggle_vision(self) -> None:
        """Переключить источник карты: эталон растеризатора <-> предсказание сети."""
        if not self._brain.has_perception:
            _LOG.info("зрение не подключено — переключать нечего")
            return
        self.vision_on = not self.vision_on
        self._observe()

    def toggle_saliency(self) -> None:
        """Наложить/снять карту внимания политики на панель зрения."""
        self.show_saliency = not self.show_saliency
        self._observe()

    def toggle_help(self) -> None:
        """Показать/скрыть список горячих клавиш поверх кадра."""
        self.show_help = not self.show_help

    # -- один кадр ---------------------------------------------------------
    def _observe(self) -> None:
        """Пересчитать всё, что показывается: карты, разницу, решение политики.

        Вынесено отдельно от шага физики, потому что смена темы или режима
        зрения обязана обновлять картинку сразу, не дожидаясь следующего кадра.
        """
        obs = self._obs
        frame = obs["pixels"]
        sem_gt = obs["semantic"]
        if self.vision_on:
            sem_pred = self._brain.see(frame)
        else:
            sem_pred = sem_gt

        # Политика смотрит ровно на ту карту, которую мы показываем в центре:
        # иначе демонстрация врала бы про то, чем агент руководствуется.
        action, p_hold, value = self._brain.decide(
            sem_pred, obs["features"], deterministic=self.config.deterministic
        )

        mismatch = np.count_nonzero(sem_pred != sem_gt) / float(sem_gt.size)
        saliency = None
        if self.show_saliency:
            saliency = self._saliency(sem_pred, obs["features"])

        self._view = {
            "frame": frame,
            "sem_gt": sem_gt,
            "sem_pred": sem_pred,
            "mismatch": float(mismatch),
            "action": int(action),
            "p_hold": float(p_hold),
            "value": float(value),
            "saliency": saliency,
        }

    def _saliency(self, sem: np.ndarray, features: np.ndarray) -> np.ndarray | None:
        """Карта внимания политики; при любой ошибке — просто нет наложения."""
        try:
            from gdai.viz.saliency import saliency_map

            return saliency_map(self._brain.policy, sem, features)
        except Exception as exc:  # pragma: no cover - зависит от версии torch
            _LOG.debug("карта внимания недоступна: %s", exc)
            return None

    def tick(self, human_hold: bool = False) -> dict[str, Any]:
        """Продвинуть игру на один кадр и вернуть телеметрию показанного кадра.

        Порядок важен: сначала показывается кадр и решение по нему, и только
        потом делается шаг — иначе HUD относился бы к уже прошедшему состоянию.
        """
        now = time.perf_counter()
        self._frame_times.append(now - self._last_tick)
        self._last_tick = now

        if self._freeze > 0:
            # Пауза «после смерти»: кадр стоит, чтобы человек успел увидеть исход.
            self._freeze -= 1
            if self._freeze == 0:
                self.reset(new_level=False)
            return self._telemetry()

        # `_view["action"]` — всегда предложение политики (его показывает P(hold)),
        # а в игру уходит либо оно, либо кнопка человека.
        action = self._view["action"] if self.ai_control else (
            ACTION_HOLD if human_hold else ACTION_NONE
        )
        self._last_action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self._obs = obs
        self._info = info
        self._steps += 1
        self._reward += float(reward)

        if terminated or truncated:
            if info.get("finished"):
                self._outcome = "ФИНИШ"
            elif info.get("died"):
                self._outcome = "СМЕРТЬ"
            else:
                self._outcome = "ЛИМИТ"
            self._freeze = FREEZE_FRAMES
            return self._telemetry()

        self._observe()
        return self._telemetry()

    def _telemetry(self) -> dict[str, Any]:
        """Снимок показанного кадра для внешнего кода (запись, тесты, CLI)."""
        data = dict(self._view)
        data["applied_action"] = int(self._last_action)
        data["progress"] = float(self._info.get("progress", 0.0))
        data["outcome"] = self._outcome
        data["episode"] = int(self._episode)
        data["step"] = int(self._steps)
        return data

    # -- отрисовка ---------------------------------------------------------
    def compose(self) -> "pygame.Surface":
        """Собрать полную поверхность окна: три панели + HUD."""
        surf = self._surface
        surf.fill(BG_COLOR)

        view = self._view
        gt = view["sem_gt"]
        pred = view["sem_pred"]

        panel_frame = view["frame"]
        if view.get("saliency") is not None:
            # Общая функция наложения из gdai.viz.saliency: у окна и у
            # сохранённых картинок внимание обязано выглядеть одинаково.
            from gdai.viz.saliency import overlay_saliency

            panel_pred = overlay_saliency(pred, view["saliency"])
        else:
            panel_pred = semantic_to_rgb(pred)
        panel_diff = _diff_panel(gt, pred)

        theme_name = self._renderer.theme.name
        source = (
            f"предсказание ({self._brain.perception_source})"
            if self.vision_on
            else "эталон (зрение выкл.)"
        )
        labels = (
            f"1. Кадр игры — тема «{theme_name}», декор {DECOR_STEPS[self._decor_index]:.2f}",
            f"2. Что видит ИИ — {source}",
            f"3. Эталон + разница: {view['mismatch'] * 100:.2f}%",
        )
        panels = (panel_frame, panel_pred, panel_diff)

        for i, (arr, label) in enumerate(zip(panels, labels)):
            x = PANEL_GAP + i * (self._pw + PANEL_GAP)
            self._draw_label(surf, label, x, PANEL_GAP - 2)
            self._blit_panel(surf, arr, x, PANEL_GAP + LABEL_H)

        if self._outcome:
            self._draw_outcome(surf, PANEL_GAP, PANEL_GAP + LABEL_H)
        self._draw_hud(surf)
        if self.show_help:
            self._draw_help(surf)
        return surf

    def frame_array(self) -> np.ndarray:
        """Готовый кадр визуализации как (H, W, 3) uint8 — для записи видео/тестов."""
        surf = self.compose()
        return np.ascontiguousarray(
            pygame.surfarray.array3d(surf).transpose(1, 0, 2)
        )

    def _blit_panel(self, target: "pygame.Surface", arr: np.ndarray, x: int, y: int) -> None:
        """Положить картинку 128x72 на своё место с целочисленным увеличением."""
        src = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8).transpose(1, 0, 2))
        surf = pygame.surfarray.make_surface(src)
        if self.scale != 1:
            # Именно scale, а не smoothscale: пиксельная карта не должна мылиться.
            surf = pygame.transform.scale(surf, (self._pw, self._ph))
        target.blit(surf, (x, y))
        pygame.draw.rect(target, PANEL_EDGE, (x - 1, y - 1, self._pw + 2, self._ph + 2), 1)

    def _draw_label(self, target: "pygame.Surface", text: str, x: int, y: int) -> None:
        """Подпись панели, обрезанная по её ширине.

        Зачем обрезка: имена случайных тем длинные, и без неё подпись первой
        панели наезжала бы на подпись второй при небольшом масштабе.
        """
        fitted = _fit_text(self._font_small, text, self._pw)
        target.blit(self._font_small.render(fitted, True, DIM_COLOR), (x, y))

    def _draw_outcome(self, target: "pygame.Surface", x: int, y: int) -> None:
        """Крупная надпись СМЕРТЬ/ФИНИШ поверх первой панели."""
        color = GOOD_COLOR if self._outcome == "ФИНИШ" else BAD_COLOR
        text = self._font_big.render(self._outcome, True, color)
        rect = text.get_rect(center=(x + self._pw // 2, y + self._ph // 2))
        shade = pygame.Surface((rect.width + 16, rect.height + 10), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        target.blit(shade, (rect.x - 8, rect.y - 5))
        target.blit(text, rect)

    def _draw_hud(self, target: "pygame.Surface") -> None:
        """Полоса прогресса, вероятность удержания, value, FPS, тема и режим."""
        view = self._view
        info = self._info
        top = PANEL_GAP + LABEL_H + self._ph + 6
        left = PANEL_GAP
        inner = self.width - 2 * PANEL_GAP

        progress = float(info.get("progress", 0.0))
        self._draw_bar(target, left, top, inner, 10, progress, ACCENT_COLOR)
        for cp in self.env.checkpoints:
            length = max(1e-6, float(self.env.level.length))
            cx = left + int(inner * min(max(cp / length, 0.0), 1.0))
            pygame.draw.line(target, (110, 120, 140), (cx, top), (cx, top + 10), 1)

        fps = self._fps()
        mode = "ИИ" if self.ai_control else "человек"
        hold = "держит" if self._last_action == ACTION_HOLD else "отпущено"
        # Строки собираются кусками и обрезаются по ширине окна: при маленьком
        # масштабе панелей текст обязан не наезжать сам на себя, а исчезать
        # с конца — самое важное стоит в начале.
        line1 = [
            f"прогресс {progress * 100:5.1f}%",
            f"x {info.get('x', 0.0):.2f}/{self.env.level.length:.0f}",
            f"эпизод {self._episode}",
            f"кадр {self._steps}",
            f"награда {self._reward:+.2f}",
            f"уровень «{info.get('level_name', '?')}»",
            f"сложность {info.get('difficulty', 0.0):.2f}",
        ]
        line2 = [
            f"режим: {mode} ({hold})",
            f"P(hold) {view['p_hold']:.3f}",
            f"value {view['value']:+.3f}",
            f"FPS {fps:.1f}",
            f"политика: {self._brain.policy_source}",
            f"зрение: {self._brain.perception_source if self.vision_on else 'выкл'}",
            "[H] подсказка",
        ]
        self._draw_segments(target, line1, left, top + 16, inner)
        self._draw_segments(target, line2, left, top + 36, inner)

        # Маленькая шкала P(hold) справа — быстрее читается, чем число.
        bar_w = 120
        bx = self.width - PANEL_GAP - bar_w
        self._draw_bar(target, bx, top + 54, bar_w, 8, view["p_hold"], GOOD_COLOR)
        target.blit(
            self._font_small.render("P(hold)", True, DIM_COLOR),
            (bx - 52, top + 52),
        )

    def _draw_segments(
        self,
        target: "pygame.Surface",
        segments: Sequence[str],
        x: int,
        y: int,
        max_w: int,
    ) -> None:
        """Нарисовать строку HUD по кускам, отбрасывая то, что не влезло."""
        font = self._font
        gap = font.size("   ")[0]
        cursor = x
        for segment in segments:
            width = font.size(segment)[0]
            if cursor + width > x + max_w:
                break
            target.blit(font.render(segment, True, TEXT_COLOR), (cursor, y))
            cursor += width + gap

    def _draw_bar(
        self,
        target: "pygame.Surface",
        x: int,
        y: int,
        w: int,
        h: int,
        value: float,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(target, BAR_BG, (x, y, w, h))
        filled = int(w * min(max(float(value), 0.0), 1.0))
        if filled > 0:
            pygame.draw.rect(target, color, (x, y, filled, h))
        pygame.draw.rect(target, PANEL_EDGE, (x, y, w, h), 1)

    def _draw_help(self, target: "pygame.Surface") -> None:
        """Список горячих клавиш поверх второй панели."""
        pad = 10
        lines = [f"{key:<6} — {text}" for key, text in HOTKEYS]
        width = max(self._font.size(line)[0] for line in lines) + 2 * pad
        height = len(lines) * 18 + 2 * pad
        box = pygame.Surface((width, height), pygame.SRCALPHA)
        box.fill((10, 12, 18, 225))
        pygame.draw.rect(box, PANEL_EDGE, box.get_rect(), 1)
        for i, line in enumerate(lines):
            box.blit(self._font.render(line, True, TEXT_COLOR), (pad, pad + i * 18))
        x = PANEL_GAP + (self._pw + PANEL_GAP) + (self._pw - width) // 2
        y = PANEL_GAP + LABEL_H + max(0, (self._ph - height) // 2)
        target.blit(box, (max(PANEL_GAP, x), max(PANEL_GAP, y)))

    def _fps(self) -> float:
        """Скользящее среднее по времени кадра — мгновенный FPS слишком дёргается."""
        if not self._frame_times:
            return 0.0
        mean = float(np.mean(self._frame_times))
        return 1.0 / mean if mean > 1e-9 else 0.0

    # -- режимы работы -----------------------------------------------------
    def handle_event(self, event: Any) -> None:
        """Обработать одно событие pygame по раскладке из SPEC §13."""
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_SPACE:
            self.set_ai_control(False)
        elif key == pygame.K_a:
            self.set_ai_control(True)
        elif key == pygame.K_t:
            self.cycle_theme()
        elif key == pygame.K_d:
            self.cycle_decoration()
        elif key == pygame.K_p:
            self.toggle_vision()
        elif key == pygame.K_s:
            self.toggle_saliency()
        elif key == pygame.K_r:
            self.reset(new_level=False)
        elif key == pygame.K_n:
            self.reset(new_level=True)
        elif key == pygame.K_h:
            self.toggle_help()

    def run(self) -> None:
        """Интерактивный цикл: окно, клавиши, ограничение частоты кадров."""
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("GDAI — что видит нейросеть")
        clock = pygame.time.Clock()
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                hold = bool(pygame.key.get_pressed()[pygame.K_SPACE])
                self.tick(human_hold=hold)
                screen.blit(self.compose(), (0, 0))
                pygame.display.flip()
                clock.tick(int(self.config.fps))
        finally:
            self.close()

    def record(
        self,
        out_path: str | os.PathLike[str],
        *,
        frames: int = 360,
        fps: int = 30,
        keep_png: bool = False,
        png_dir: str | os.PathLike[str] | None = None,
    ) -> str:
        """Записать `frames` кадров без окна и собрать их в GIF/MP4.

        Возвращает путь к собранному файлу; если в системе нет ни одного
        сборщика (imageio / PIL / cv2), возвращается путь к каталогу с PNG —
        демонстрация не должна пропадать только из-за отсутствия кодека.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp_dir: str | None = None
        if png_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="gdai-demo-")
            frames_dir = Path(temp_dir)
        else:
            frames_dir = Path(png_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        t0 = time.time()
        for i in range(int(frames)):
            self.tick(human_hold=False)
            surface = self.compose()
            png = frames_dir / f"frame_{i:05d}.png"
            pygame.image.save(surface, str(png))
            paths.append(png)
        _LOG.info(
            "записано %d кадров %dx%d за %.1f с -> %s",
            len(paths), self.width, self.height, time.time() - t0, frames_dir,
        )

        try:
            result = assemble_animation(paths, out, fps=fps)
        except ImportError as exc:
            _LOG.warning(
                "собрать %s нечем (%s); кадры остались в %s", out.name, exc, frames_dir
            )
            return str(frames_dir)

        if temp_dir is not None and not keep_png:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return result


# ---------------------------------------------------------------------------
# вспомогательные функции отрисовки
# ---------------------------------------------------------------------------
def _fit_text(font: "pygame.font.Font", text: str, max_w: int) -> str:
    """Обрезать строку многоточием так, чтобы она поместилась в `max_w` пикселей."""
    if font.size(text)[0] <= max_w:
        return text
    ellipsis = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.size(text[:mid] + ellipsis)[0] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis


def _nearest_decor_index(level: float) -> int:
    """Ближайшая ступень декора к запрошенному значению."""
    value = float(min(max(float(level), 0.0), 1.0))
    return int(np.argmin([abs(value - step) for step in DECOR_STEPS]))


def _diff_panel(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Эталонная карта с подсветкой пикселей, где зрение ошиблось.

    Зачем гасить совпадения: без затемнения красные точки теряются на пёстрой
    карте классов, а именно они и есть предмет разговора.
    """
    rgb = semantic_to_rgb(gt).astype(np.float32) * DIFF_DIM
    panel = rgb.astype(np.uint8)
    mismatch = np.asarray(pred) != np.asarray(gt)
    if mismatch.any():
        panel[mismatch] = DIFF_COLOR
    return panel


# ---------------------------------------------------------------------------
# сборка анимации
# ---------------------------------------------------------------------------
GIF_SUFFIXES: tuple[str, ...] = (".gif",)
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".webm", ".mov")


def assemble_animation(
    frame_paths: Sequence[str | os.PathLike[str]],
    out_path: str | os.PathLike[str],
    fps: int = 30,
) -> str:
    """Собрать PNG-кадры в GIF или видео; вернуть путь к файлу.

    Зачем сборка идёт по путям, а не по массивам в памяти: демонстрация на
    несколько тысяч кадров при полном разрешении не поместилась бы в память,
    а так каждый кадр читается ровно один раз.

    Бросает `ImportError` со списком того, что нужно поставить, если ни одна
    из опциональных библиотек (imageio / PIL / cv2) не доступна.
    """
    out = Path(out_path)
    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise ValueError("Нечего собирать: список кадров пуст")
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()

    if suffix in GIF_SUFFIXES:
        writer = _gif_imageio(paths, out, fps) or _gif_pillow(paths, out, fps)
        if writer is None:
            raise ImportError(
                "GIF собрать нечем: нужен imageio (pip install imageio) или "
                "Pillow (pip install pillow). PNG-кадры уже записаны."
            )
        return writer
    if suffix in VIDEO_SUFFIXES:
        writer = _video_imageio(paths, out, fps) or _video_cv2(paths, out, fps)
        if writer is None:
            raise ImportError(
                "Видео собрать нечем: нужен imageio[ffmpeg] или opencv-python-headless. "
                "PNG-кадры уже записаны."
            )
        return writer
    raise ValueError(
        f"Неизвестный формат {out.suffix!r}: поддерживаются "
        f"{', '.join(GIF_SUFFIXES + VIDEO_SUFFIXES)}"
    )


def _gif_imageio(paths: list[Path], out: Path, fps: int) -> str | None:
    """GIF через imageio (лучшее качество палитры); None — библиотеки нет."""
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        writer = imageio.get_writer(
            str(out), mode="I", duration=1.0 / max(1, fps), loop=0
        )
    except Exception as exc:  # pragma: no cover - разные версии плагинов
        _LOG.debug("imageio не открыл GIF-писатель (%s), пробуем Pillow", exc)
        return None
    try:
        for path in paths:
            writer.append_data(imageio.imread(str(path)))
    finally:
        writer.close()
    return str(out)


def _gif_pillow(paths: list[Path], out: Path, fps: int) -> str | None:
    """GIF через Pillow — он всегда рядом с matplotlib, поэтому это рабочий путь."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    duration = max(20, int(round(1000.0 / max(1, fps))))
    images = [Image.open(str(p)).convert("P", palette=Image.ADAPTIVE, colors=256) for p in paths]
    try:
        images[0].save(
            str(out),
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()
    return str(out)


def _video_imageio(paths: list[Path], out: Path, fps: int) -> str | None:
    """Видео через imageio+ffmpeg; None — нет библиотеки или плагина."""
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with imageio.get_writer(str(out), fps=max(1, fps)) as writer:
            for path in paths:
                writer.append_data(imageio.imread(str(path)))
    except Exception as exc:  # pragma: no cover - отсутствие ffmpeg
        _LOG.debug("imageio не собрал видео (%s), пробуем cv2", exc)
        return None
    return str(out)


def _video_cv2(paths: list[Path], out: Path, fps: int) -> str | None:
    """Видео через OpenCV (mp4v) — запасной путь без ffmpeg-плагина."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    first = cv2.imread(str(paths[0]))
    if first is None:  # pragma: no cover - битый PNG
        return None
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(max(1, fps)), (width, height))
    if not writer.isOpened():  # pragma: no cover - нет кодека
        return None
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    return str(out)


# ---------------------------------------------------------------------------
# публичные точки входа
# ---------------------------------------------------------------------------
def run_viewer(config: ViewerConfig | None = None, *, agent: Any = None) -> None:
    """Открыть интерактивное окно «что видит ИИ» (`python -m gdai watch`).

    `agent` — готовый `gdai.pipeline.GDAgent`; если не передан, вьюер сам
    поднимет политику и зрение из путей в конфиге (или случайные сети).
    """
    viewer = Viewer(config, agent=agent)
    viewer.run()


def record_demo(
    out_path: str | os.PathLike[str] = "demo.gif",
    *,
    config: ViewerConfig | None = None,
    frames: int = 360,
    fps: int = 30,
    keep_png: bool = False,
    png_dir: str | os.PathLike[str] | None = None,
    agent: Any = None,
) -> str:
    """Записать демонстрацию без окна и вернуть путь к готовому файлу.

    Это headless-режим из SPEC §13: ни одного вызова `display.set_mode`, всё
    рисуется в поверхность и сохраняется в PNG, после чего собирается в GIF
    (imageio/Pillow) или MP4 (imageio/cv2). Именно этой функцией делаются
    картинки для README и проверка связки на сервере без дисплея.
    """
    viewer = Viewer(config, agent=agent)
    try:
        return viewer.record(
            out_path, frames=frames, fps=fps, keep_png=keep_png, png_dir=png_dir
        )
    finally:
        viewer.close()


def build_parser() -> argparse.ArgumentParser:
    """Аргументы просмотра — используются и `python -m gdai watch`, и этим модулем."""
    parser = argparse.ArgumentParser(
        prog="gdai-watch",
        description="Показать, что видит нейросеть: кадр | карта зрения | эталон и разница",
    )
    parser.add_argument("--policy", default=None, help="веса политики (runs/agent/best.pt)")
    parser.add_argument(
        "--perception", default=None, help="веса зрения (runs/perception/best.pt)"
    )
    parser.add_argument("--level", default=None, help="уровень из файла *.json")
    parser.add_argument("--difficulty", type=float, default=0.3, help="сложность 0..1")
    parser.add_argument("--seed", type=int, default=0, help="seed уровня и темы")
    parser.add_argument("--scale", type=int, default=3, help="увеличение панелей")
    parser.add_argument("--fps", type=int, default=60, help="частота кадров окна")
    parser.add_argument("--theme", default=None, help="фиксированная тема по имени")
    parser.add_argument(
        "--decoration", type=float, default=1.0, help="плотность декора 0..1"
    )
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="не подключать зрение: показывать эталонную карту",
    )
    parser.add_argument(
        "--manual", action="store_true", help="стартовать под управлением человека"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="выбирать действие выборкой, а не argmax"
    )
    parser.add_argument(
        "--saliency", action="store_true", help="сразу включить карту внимания"
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="OUT",
        help="headless-запись демонстрации в GIF/MP4 вместо окна",
    )
    parser.add_argument(
        "--frames", type=int, default=360, help="сколько кадров записать (--record)"
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="частота кадров в записанном файле (окно всегда идёт на --fps)",
    )
    parser.add_argument(
        "--keep-png", action="store_true", help="не удалять PNG-кадры после сборки"
    )
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    return parser


def config_from_args(args: argparse.Namespace) -> ViewerConfig:
    """Разобранные аргументы -> `ViewerConfig` (одна точка правды для CLI)."""
    return ViewerConfig(
        policy_path=args.policy,
        perception_path=args.perception,
        level_path=args.level,
        difficulty=float(args.difficulty),
        seed=args.seed,
        scale=int(args.scale),
        fps=int(args.fps),
        use_perception=not args.no_perception,
        ai_control=not args.manual,
        deterministic=not args.stochastic,
        decoration_level=float(args.decoration),
        theme=args.theme,
        device=args.device,
        show_saliency=bool(args.saliency),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа модуля: окно или запись демо (`python -m gdai.viz.viewer`)."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    if args.record:
        path = record_demo(
            args.record,
            config=config,
            frames=int(args.frames),
            fps=max(1, int(args.record_fps)),
            keep_png=bool(args.keep_png),
        )
        _LOG.info("демонстрация записана: %s", path)
        return 0
    run_viewer(config)
    return 0


__all__ = [
    "Viewer",
    "ViewerConfig",
    "run_viewer",
    "record_demo",
    "assemble_animation",
    "build_parser",
    "config_from_args",
    "main",
    "HOTKEYS",
    "DECOR_STEPS",
    "PANEL_GAP",
    "LABEL_H",
    "HUD_H",
]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
