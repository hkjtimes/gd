"""Единая точка входа проекта: `python -m gdai <команда>` (SPEC §15).

Зачем один CLI на всё
---------------------
У проекта пять независимых подсистем (генератор уровней, среда, зрение,
политика, визуализация), и у каждой свой набор ручек. Без общей команды
пользователю пришлось бы помнить пять способов запуска и пять наборов
умолчаний. Здесь же одна точка входа, одинаковые имена аргументов
(`--out`, `--seed`, `--device`) и русская справка у каждой подкоманды.

Про вывод в stdout
------------------
Правило проекта «никаких print в библиотеке» относится к библиотеке: там
единственный канал — `gdai.utils.logging.get_logger`. CLI же и есть интерфейс
пользователя, поэтому отчёты команд печатаются в stdout через `_emit`
(диагностика по-прежнему уходит в логгер). Так вывод команды можно спокойно
перенаправить в файл, а логи — оставить на экране.

Все тяжёлые импорты (torch, pygame, matplotlib) сделаны ВНУТРИ обработчиков
команд: `python -m gdai --help` обязан отвечать мгновенно, а справка по
`gen-level` не должна поднимать torch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from gdai.utils.logging import get_logger

_log = get_logger("cli")

# Значения по умолчанию, общие для нескольких команд: путь к обученным весам
# один и тот же во всех сценариях, и разъезжаться им нельзя.
DEFAULT_PERCEPTION_RUN: str = "runs/perception"
DEFAULT_AGENT_RUN: str = "runs/agent"
DEFAULT_LEVELS_DIR: str = "levels"
DEFAULT_SELFCHECK_DIR: str = "runs/selfcheck"

# Параметры «крошечных» прогонов selfcheck. Вынесены в константы, потому что
# от них напрямую зависит бюджет времени всей проверки (SPEC §15: < 60 c).
SELFCHECK_PERCEPTION_STEPS: int = 30
SELFCHECK_AGENT_STEPS: int = 2000
SELFCHECK_ENV_STEPS: int = 200
SELFCHECK_DEMO_FRAMES: int = 40
SELFCHECK_EVAL_EPISODES: int = 2
# Порог «кадры действительно разные»: средняя попиксельная разница двух тем.
# 6 из 255 — заведомо больше шума компрессии и заведомо меньше разницы
# между любыми двумя настоящими оформлениями.
SELFCHECK_FRAME_DIFF: float = 6.0


# ---------------------------------------------------------------------------
# вывод
# ---------------------------------------------------------------------------
_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color_enabled(force: bool | None = None) -> bool:
    """Красить ли вывод: только живой терминал и без запрета через NO_COLOR."""
    if force is not None:
        return bool(force)
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, code: str, enabled: bool) -> str:
    """Обернуть текст ANSI-кодом, если цвет разрешён."""
    return f"{code}{text}{_RESET}" if enabled else text


def _emit(text: str = "") -> None:
    """Напечатать строку отчёта команды (см. «Про вывод в stdout» в docstring)."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _fmt_metrics(metrics: dict[str, Any], keys: Sequence[str]) -> list[str]:
    """Отобрать и красиво отформатировать метрики для отчёта."""
    lines: list[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"  {key:<18} {value:.4f}")
        else:
            lines.append(f"  {key:<18} {value}")
    return lines


# ---------------------------------------------------------------------------
# gen-level
# ---------------------------------------------------------------------------
def cmd_gen_level(args: argparse.Namespace) -> int:
    """Сгенерировать процедурный уровень и сохранить его в JSON."""
    from gdai.env.generator import generate_level, is_solvable, make_checkpoints
    from gdai.utils.seeding import make_rng

    rng = make_rng(args.seed)
    name = args.name or Path(args.out).stem
    t0 = time.time()
    level = generate_level(
        float(args.difficulty), rng, name=name, length=args.length
    )
    if not level.checkpoints:
        level.checkpoints = make_checkpoints(level)

    solvable: bool | None = None
    if not args.no_check:
        solvable = is_solvable(level)
        if not solvable:
            _emit("Уровень получился непроходимым — файл не сохранён.")
            _log.error("непроходимый уровень: сложность %.2f, seed %s",
                       args.difficulty, args.seed)
            return 1

    path = level.save(args.out)
    _emit(f"Уровень сохранён: {path}")
    _emit(f"  имя               {level.name}")
    _emit(f"  сложность         {float(args.difficulty):.2f}")
    _emit(f"  длина             {level.length:.1f} тайлов")
    _emit(f"  объектов          {len(level.objects)}")
    _emit(f"  чекпойнтов        {len(level.checkpoints)}")
    _emit(f"  проходимость      {'проверена' if solvable else 'не проверялась'}")
    _emit(f"  время             {time.time() - t0:.2f} с")
    return 0


# ---------------------------------------------------------------------------
# train-perception / train-agent
# ---------------------------------------------------------------------------
def cmd_train_perception(args: argparse.Namespace) -> int:
    """Обучить зрение (U-Net) на синтетике с доменной рандомизацией."""
    from gdai.perception.train import config_from_args, train_perception

    cfg = config_from_args(args)
    _emit(f"Обучение зрения: {cfg.steps} шагов, батч {cfg.batch_size}, "
          f"каталог {cfg.out_dir}")
    result = train_perception(cfg)
    _emit("Готово.")
    _emit("\n".join(_fmt_metrics(result, (
        "steps", "params", "device", "duration_sec", "steps_per_sec",
        "pixel_acc", "miou", "iou_solid", "iou_hazard",
    ))))
    _emit(f"  веса               {result['best_path']}")
    _emit(f"  метрики            {result['metrics_path']}")
    return 0


def cmd_train_agent(args: argparse.Namespace) -> int:
    """Обучить политику PPO на канонических картах."""
    from gdai.agent.ppo import train_agent
    from gdai.config import CurriculumConfig, EnvConfig, PPOConfig

    cfg = PPOConfig(
        num_envs=int(args.num_envs),
        rollout_steps=int(args.rollout_steps),
        total_steps=int(args.total_steps),
        lr=float(args.lr),
        device=str(args.device),
        out_dir=str(args.out),
    )
    env_cfg = EnvConfig(
        obs_mode="semantic",          # политика не видит декораций по определению
        max_steps=int(args.max_steps),
        difficulty=float(args.difficulty),
        level_path=args.level,
        seed=int(args.seed),
        practice_checkpoints=not args.no_practice,
        semantic_noise=float(args.semantic_noise),
    )
    curriculum = CurriculumConfig() if args.curriculum else None

    _emit(f"Обучение политики: {cfg.total_steps} шагов, {cfg.num_envs} сред, "
          f"каталог {cfg.out_dir}"
          + (", учебный план включён" if curriculum else ""))
    result = train_agent(cfg, env_cfg, curriculum)
    _emit("Готово.")
    _emit("\n".join(_fmt_metrics(result, (
        "global_step", "iterations", "elapsed", "fps", "episodes",
        "finished_episodes", "success_rate", "best_success_rate",
        "mean_reward", "mean_progress", "mean_ep_len", "difficulty",
        "entropy", "approx_kl", "explained_variance", "stop_reason",
    ))))
    _emit(f"  веса               {result['best_path'] or result['last_path']}")
    _emit(f"  метрики            {result['metrics_path']}")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    """Оценить агента: доля прохождений и средний прогресс по эпизодам."""
    from gdai.config import EnvConfig
    from gdai.env.gd_env import GeometryDashEnv
    from gdai.pipeline import GDAgent, evaluate

    # Зрение включаем, если заданы веса или явно попросили: без этого нет
    # смысла платить за рендер кадров на каждом шаге.
    use_vision = bool(args.perception) or bool(args.use_perception)
    env_cfg = EnvConfig(
        obs_mode="both" if use_vision else "semantic",
        max_steps=int(args.max_steps),
        difficulty=float(args.difficulty),
        level_path=args.level,
        seed=int(args.seed),
        practice_checkpoints=False,   # честная оценка: только полные прогоны
        randomize_theme=True,
    )
    agent = GDAgent(
        policy_path=args.policy,
        perception_path=args.perception,
        device=str(args.device),
        use_perception=use_vision,
    )
    env = GeometryDashEnv(env_cfg)
    t0 = time.time()
    try:
        result = evaluate(
            agent,
            env,
            episodes=int(args.episodes),
            use_perception=use_vision,
            deterministic=not args.stochastic,
            seed=int(args.seed),
        )
    finally:
        env.close()

    info = agent.describe()
    _emit(f"Оценка: {result['episodes']} эпизодов, сложность {args.difficulty}, "
          f"{time.time() - t0:.1f} с")
    _emit(f"  политика           {info['policy']} ({info['policy_params']} параметров)")
    _emit(f"  зрение             {info['perception']}"
          + (f" ({info['perception_params']} параметров)" if use_vision else ""))
    _emit("\n".join(_fmt_metrics(result, (
        "success_rate", "mean_progress", "mean_reward", "mean_len",
        "max_progress", "deaths", "timeouts",
    ))))
    if info["policy"] == "случайная":
        _emit("  ВНИМАНИЕ: веса политики случайные — цифры показывают только, "
              "что связка работает.")
    return 0


# ---------------------------------------------------------------------------
# play / watch / demo
# ---------------------------------------------------------------------------
def cmd_play(args: argparse.Namespace) -> int:
    """Играть самому в окне визуализатора (пробел — прыжок)."""
    from gdai.viz.viewer import ViewerConfig, run_viewer

    config = ViewerConfig(
        level_path=args.level,
        difficulty=float(args.difficulty),
        seed=args.seed,
        scale=int(args.scale),
        fps=int(args.fps),
        use_perception=False,   # человеку интереснее эталон, а не шум сети
        ai_control=False,
        decoration_level=float(args.decoration),
        theme=args.theme,
        show_help=True,
    )
    run_viewer(config)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Показать, что видит ИИ: кадр | предсказанная карта | эталон и разница."""
    from gdai.viz.viewer import config_from_args, record_demo, run_viewer

    config = config_from_args(args)
    if args.record:
        path = record_demo(
            args.record,
            config=config,
            frames=int(args.frames),
            fps=max(1, int(args.record_fps)),
            keep_png=bool(args.keep_png),
        )
        _emit(f"Запись готова: {path}")
        return 0
    run_viewer(config)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Записать демонстрацию (GIF/MP4) без окна — для README и отчётов."""
    from gdai.viz.viewer import ViewerConfig, record_demo

    config = ViewerConfig(
        policy_path=args.policy,
        perception_path=args.perception,
        level_path=args.level,
        difficulty=float(args.difficulty),
        seed=args.seed,
        scale=int(args.scale),
        use_perception=not args.no_perception,
        ai_control=True,
        decoration_level=float(args.decoration),
        theme=args.theme,
        device=str(args.device),
        show_saliency=bool(args.saliency),
    )
    t0 = time.time()
    try:
        path = record_demo(
            args.out,
            config=config,
            frames=int(args.frames),
            fps=max(1, int(args.fps)),
            keep_png=bool(args.keep_png),
        )
    except ValueError as exc:
        _emit(f"Не удалось записать демо: {exc}")
        return 1
    _emit(f"Демонстрация готова: {path} ({args.frames} кадров, "
          f"{time.time() - t0:.1f} с)")
    return 0


# ---------------------------------------------------------------------------
# plot / play-real
# ---------------------------------------------------------------------------
def cmd_plot(args: argparse.Namespace) -> int:
    """Построить графики обучения из `metrics.jsonl` прогона."""
    from gdai.viz.plots import load_metrics, plot_run, summarize

    try:
        path = plot_run(
            args.run,
            args.out,
            title=args.title,
            smooth=int(args.smooth),
            max_cols=int(args.cols),
            dpi=int(args.dpi),
            include_extra=not args.only_known,
        )
    except (FileNotFoundError, ValueError) as exc:
        _emit(f"Графики не построены: {exc}")
        return 1
    _emit(f"График сохранён: {path}")
    summary = summarize(load_metrics(args.run))
    if summary:
        _emit("Последние значения:")
        _emit("\n".join(_fmt_metrics(summary, tuple(summary)[:12])))
    return 0


def cmd_play_real(args: argparse.Namespace) -> int:
    """Играть в настоящую Geometry Dash (нужны mss/pynput и калибровка)."""
    from gdai.realgame.play import RealGameConfig, play_real

    cfg = RealGameConfig(
        policy_path=args.policy,
        perception_path=args.perception,
        # None означает «путь калибровки по умолчанию»: его знает realgame,
        # и дублировать константу в CLI незачем.
        region_path=args.region or RealGameConfig().region_path,
        monitor=int(args.monitor),
        fps=float(args.fps),
        max_seconds=float(args.seconds),
        device=str(args.device),
        deterministic=not args.stochastic,
        dry_run=not args.press,
        quit_key=str(args.quit_key),
    )
    if not args.press:
        _emit("Режим наблюдения: клавиши НЕ нажимаются. Добавьте --press, "
              "когда убедитесь, что калибровка верна.")
    try:
        stats = play_real(cfg)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        _emit(f"Настоящая игра недоступна: {exc}")
        return 2
    _emit("\n".join(_fmt_metrics(stats, tuple(stats))))
    return 0


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------
def _selfcheck_steps(
    args: argparse.Namespace, ctx: dict[str, Any]
) -> list[tuple[str, Callable[[], str]]]:
    """Список проверок «весь путь от уровня до демо-кадра».

    Каждая проверка возвращает короткую строку с фактами (чтобы отчёт был не
    «ok», а «ok, вот цифры») и бросает исключение при провале. Порядок
    повторяет порядок конвейера: то, что ломается раньше, и должно падать
    раньше — так по номеру шага сразу видно, какой слой виноват.
    """
    out_dir = Path(args.out)

    def step_level() -> str:
        from gdai.env.generator import generate_level, make_checkpoints
        from gdai.utils.seeding import make_rng

        level = generate_level(
            0.45, make_rng(args.seed), name="selfcheck", length=70.0
        )
        if not level.checkpoints:
            level.checkpoints = make_checkpoints(level)
        ctx["level"] = level
        return (f"«{level.name}»: {level.length:.0f} тайлов, "
                f"{len(level.objects)} объектов, {len(level.checkpoints)} чекпойнтов")

    def step_solvable() -> str:
        from gdai.env.generator import is_solvable
        from gdai.env.level import Level

        if not is_solvable(ctx["level"]):
            raise AssertionError("сгенерированный уровень непроходим")
        checked: list[str] = []
        levels_dir = Path(args.levels_dir)
        for path in sorted(levels_dir.glob("*.json")):
            level = Level.load(path)
            if not is_solvable(level):
                raise AssertionError(f"уровень {path.name} непроходим")
            checked.append(path.stem)
        extra = f"; из {levels_dir}/ проверено {len(checked)}" if checked else ""
        return f"процедурный уровень проходится поиском по кадрам{extra}"

    def step_render() -> str:
        from gdai.env.physics import make_initial_state, step_physics
        from gdai.env.render import Renderer
        from gdai.env.themes import BUILTIN_THEMES, random_theme
        from gdai.utils.seeding import make_rng

        level = ctx["level"]
        # Отходим от старта: на первых кадрах в камере пусто, и «разные темы»
        # различались бы только фоном.
        state = make_initial_state(level)
        for i in range(45):
            state, _events = step_physics(state, level, hold=(i % 17 == 0))
        ctx["state"] = state

        from gdai.env.semantic import render_semantic

        rng = make_rng(args.seed)
        themes = list(BUILTIN_THEMES[:3]) + [random_theme(rng) for _ in range(2)]
        renderer = Renderer(seed=args.seed)
        frames = []
        maps = []
        for i, theme in enumerate(themes):
            renderer.set_theme(theme)
            frame = renderer.render(level, state, i * 7)
            # Карта снимается СРАЗУ ПОСЛЕ рендера этой темы: так проверка ловит
            # и запрещённый SPEC §9 случай, когда декорации сдвинули игровой
            # объект или состояние игрока (тогда разметка поедет вслед за ними).
            maps.append(render_semantic(level, state))
            if frame.shape != (72, 128, 3) or frame.dtype.name != "uint8":
                raise AssertionError(f"кадр темы {theme.name}: {frame.shape} {frame.dtype}")
            if float(frame.std()) < 1.0:
                raise AssertionError(f"кадр темы {theme.name} почти однотонный")
            frames.append(frame)
        ctx["frames"] = frames
        ctx["maps"] = maps
        ctx["themes"] = themes
        ctx["renderer"] = renderer
        names = ", ".join(t.name for t in themes)
        return f"{len(frames)} тем отрисованы 128x72: {names}"

    def step_invariance() -> str:
        import numpy as np

        from gdai.constants import NUM_CLASSES, PLAYER

        maps, themes = ctx["maps"], ctx["themes"]
        reference = maps[0]
        for theme, sem in zip(themes, maps):
            if not np.array_equal(sem, reference):
                raise AssertionError(f"карта изменилась при теме {theme.name}")
        if int(reference.max()) >= NUM_CLASSES or int(reference.min()) < 0:
            raise AssertionError("классы карты вышли за 0..9")
        if not np.any(reference == PLAYER):
            raise AssertionError("на карте нет игрока — камера смотрит не туда")

        frames = ctx["frames"]
        diffs = [
            float(np.mean(np.abs(f.astype(np.int16) - frames[0].astype(np.int16))))
            for f in frames[1:]
        ]
        weak = [d for d in diffs if d < SELFCHECK_FRAME_DIFF]
        if weak:
            raise AssertionError(
                f"кадры разных тем почти одинаковы (мин. разница {min(diffs):.1f})"
            )
        return (f"карта байт-в-байт одна и та же, кадры различаются на "
                f"{min(diffs):.0f}..{max(diffs):.0f} из 255")

    def step_env() -> str:
        from gdai.config import EnvConfig
        from gdai.env.gd_env import FEATURE_DIM, GeometryDashEnv

        env = GeometryDashEnv(EnvConfig(
            obs_mode="both",             # проверяем сразу оба канала наблюдения
            difficulty=0.3,
            max_steps=1200,
            seed=args.seed,
        ))
        try:
            obs, _info = env.reset(seed=args.seed)
            episodes = 0
            total = 0.0
            for i in range(SELFCHECK_ENV_STEPS):
                obs, reward, terminated, truncated, info = env.step(int(i % 11 == 0))
                total += float(reward)
                if obs["semantic"].shape != (72, 128):
                    raise AssertionError(f"карта {obs['semantic'].shape}")
                if obs["pixels"].shape != (72, 128, 3):
                    raise AssertionError(f"кадр {obs['pixels'].shape}")
                if obs["features"].shape != (FEATURE_DIM,):
                    raise AssertionError(f"признаки {obs['features'].shape}")
                if terminated or truncated:
                    episodes += 1
                    env.reset()
            last_x = float(info.get("x", 0.0))
        finally:
            env.close()
        return (f"{SELFCHECK_ENV_STEPS} шагов, эпизодов {episodes + 1}, "
                f"награда {total:+.2f}, x={last_x:.1f}")

    def step_perception() -> str:
        from gdai.config import PerceptionConfig
        from gdai.perception.train import train_perception

        steps = int(args.perception_steps)
        cfg = PerceptionConfig(
            steps=steps,
            batch_size=4,
            val_every=steps,           # ровно одна валидация — в конце
            device=str(args.device),
            out_dir=str(out_dir / "perception"),
        )
        # Узкое место шага — не обучение, а генерация данных (уровень + рендер
        # на каждый сэмпл). Штатная эвристика загрузчика на одноядерной машине
        # даёт ноль worker'ов, и проверка перестала бы укладываться в бюджет
        # SPEC §15, поэтому здесь минимум два процесса. Явно заданный
        # GDAI_NUM_WORKERS уважаем — это осознанный выбор пользователя.
        # Ядро оставляем самому обучению: worker'ов ровно столько, сколько
        # свободных ядер (но не меньше двух — иначе генерация встанет в один
        # поток), иначе процессы данных и потоки torch дерутся за CPU и шаг
        # замедляется в разы.
        restore = os.environ.get("GDAI_NUM_WORKERS")
        if restore is None:
            os.environ["GDAI_NUM_WORKERS"] = str(
                max(2, min(4, (os.cpu_count() or 2) - 1))
            )
        try:
            result = train_perception(cfg)
        finally:
            if restore is None:
                os.environ.pop("GDAI_NUM_WORKERS", None)
        best = Path(result["best_path"])
        if not best.exists():
            raise AssertionError("чекпойнт зрения не сохранён")
        ctx["perception_path"] = str(best)
        return (f"{steps} шагов, {result['params']} параметров, "
                f"acc {result['pixel_acc']:.3f}, mIoU {result['miou']:.3f} "
                f"(на отложенных темах)")

    def step_agent() -> str:
        from gdai.agent.ppo import train_agent
        from gdai.config import EnvConfig, PPOConfig

        total = int(args.agent_steps)
        cfg = PPOConfig(
            num_envs=4,
            rollout_steps=64,
            total_steps=total,
            device=str(args.device),
            out_dir=str(out_dir / "agent"),
        )
        env_cfg = EnvConfig(
            obs_mode="semantic",
            difficulty=0.15,
            max_steps=1200,
            seed=args.seed,
        )
        result = train_agent(cfg, env_cfg)
        path = result["best_path"] or result["last_path"]
        if not Path(path).exists():
            raise AssertionError("чекпойнт политики не сохранён")
        ctx["policy_path"] = str(path)
        return (f"{result['global_step']} шагов за {result['elapsed']:.1f} с "
                f"({result['fps']:.0f} шагов/с), награда {result['mean_reward']:+.2f}, "
                f"эпизодов {result['episodes']}")

    def step_pipeline() -> str:
        from gdai.config import EnvConfig
        from gdai.env.gd_env import GeometryDashEnv
        from gdai.pipeline import GDAgent, evaluate

        agent = GDAgent(
            policy_path=ctx.get("policy_path"),
            perception_path=ctx.get("perception_path"),
            device=str(args.device),
            use_perception=True,
        )
        ctx["agent"] = agent

        sem = agent.see(ctx["frames"][0])
        if sem.shape != (72, 128):
            raise AssertionError(f"зрение вернуло {sem.shape}")

        env = GeometryDashEnv(EnvConfig(
            obs_mode="both",
            difficulty=0.15,
            max_steps=300,
            seed=args.seed,
            practice_checkpoints=False,
        ))
        try:
            result = evaluate(
                agent, env,
                episodes=SELFCHECK_EVAL_EPISODES,
                use_perception=True,
                seed=args.seed,
            )
        finally:
            env.close()
        return (f"кадр -> карта -> действие работает; {result['episodes']} эпизода "
                f"со зрением: прогресс {result['mean_progress']:.2f}, "
                f"награда {result['mean_reward']:+.2f}")

    def step_demo() -> str:
        import pygame

        from gdai.viz.viewer import Viewer, ViewerConfig

        config = ViewerConfig(
            policy_path=ctx.get("policy_path"),
            perception_path=ctx.get("perception_path"),
            difficulty=0.3,
            seed=args.seed,
            scale=2,
            use_perception=True,
            device=str(args.device),
            practice_checkpoints=False,
            max_steps=600,
        )
        viewer = Viewer(config, agent=ctx.get("agent"))
        try:
            for _ in range(SELFCHECK_DEMO_FRAMES):
                viewer.tick()
            surface = viewer.compose()
            out_dir.mkdir(parents=True, exist_ok=True)
            png = out_dir / "demo_frame.png"
            pygame.image.save(surface, str(png))
        finally:
            viewer.close()
        if not png.exists() or png.stat().st_size < 1024:
            raise AssertionError("демо-кадр не записался")
        return (f"{png} ({viewer.width}x{viewer.height}, "
                f"{png.stat().st_size // 1024} КБ)")

    return [
        ("генерация уровня", step_level),
        ("проверка проходимости", step_solvable),
        ("рендер нескольких тем", step_render),
        ("инвариантность карты к дизайну", step_invariance),
        (f"{SELFCHECK_ENV_STEPS} шагов среды", step_env),
        (f"обучение зрения ({args.perception_steps} шагов)", step_perception),
        (f"обучение политики ({args.agent_steps} шагов)", step_agent),
        ("агент целиком: зрение -> политика", step_pipeline),
        ("запись демо-кадра", step_demo),
    ]


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Прогнать всю связку за десятки секунд и напечатать отчёт по пунктам."""
    color = _color_enabled(True if args.color else (False if args.no_color else None))
    ok_mark = _paint("[ ok ]", _GREEN, color)
    fail_mark = _paint("[ СБОЙ ]", _RED, color)
    skip_mark = _paint("[ — ]", _DIM, color)

    ctx: dict[str, Any] = {}
    steps = _selfcheck_steps(args, ctx)
    total = len(steps)

    _emit(_paint("GDAI selfcheck — быстрая проверка всей связки", _BOLD, color))
    _emit(f"каталог артефактов: {args.out}, seed {args.seed}, "
          f"устройство {args.device}")
    _emit("")

    started = time.time()
    failed_at: int | None = None
    for index, (title, func) in enumerate(steps, start=1):
        if failed_at is not None:
            _emit(f"  {skip_mark} {index}/{total} {title} — пропущено")
            continue
        t0 = time.time()
        try:
            detail = func()
        except Exception as exc:  # отчёт важнее стека: печатаем суть
            failed_at = index
            _emit(f"  {fail_mark} {index}/{total} {title}: "
                  f"{type(exc).__name__}: {exc}")
            _log.exception("selfcheck: шаг %d (%s) упал", index, title)
            continue
        _emit(f"  {ok_mark} {index}/{total} {title} — {detail} "
              f"{_paint(f'({time.time() - t0:.1f} с)', _DIM, color)}")

    elapsed = time.time() - started
    _emit("")
    if failed_at is None:
        _emit(_paint(f"ВСЁ РАБОТАЕТ: {total}/{total} проверок пройдено "
                     f"за {elapsed:.1f} с", _GREEN + _BOLD, color))
        _emit(f"артефакты: {args.out} (веса, метрики, demo_frame.png)")
        return 0
    _emit(_paint(f"ПРОВАЛ на шаге {failed_at}/{total} (за {elapsed:.1f} с)",
                 _RED + _BOLD, color))
    _emit("подробности — в логе выше (GDAI_LOG_LEVEL=DEBUG даёт больше)")
    return 1


# ---------------------------------------------------------------------------
# сборка парсера
# ---------------------------------------------------------------------------
def _add_viewer_args(parser: argparse.ArgumentParser) -> None:
    """Аргументы просмотра.

    Имена совпадают с `gdai.viz.viewer.config_from_args`: конфиг собирает та же
    функция, что и у самостоятельного запуска модуля, поэтому расходиться
    настройкам окна и CLI просто негде.
    """
    parser.add_argument("--policy", default=None, help="веса политики (runs/agent/best.pt)")
    parser.add_argument("--perception", default=None,
                        help="веса зрения (runs/perception/best.pt)")
    parser.add_argument("--level", default=None, help="уровень из файла *.json")
    parser.add_argument("--difficulty", type=float, default=0.3, help="сложность 0..1")
    parser.add_argument("--seed", type=int, default=0, help="seed уровня и темы")
    parser.add_argument("--scale", type=int, default=3, help="увеличение панелей")
    parser.add_argument("--fps", type=int, default=60, help="частота кадров окна")
    parser.add_argument("--theme", default=None, help="фиксированная тема по имени")
    parser.add_argument("--decoration", type=float, default=1.0,
                        help="плотность декора 0..1")
    parser.add_argument("--no-perception", action="store_true",
                        help="не подключать зрение: показывать эталонную карту")
    parser.add_argument("--manual", action="store_true",
                        help="стартовать под управлением человека")
    parser.add_argument("--stochastic", action="store_true",
                        help="выбирать действие выборкой, а не argmax")
    parser.add_argument("--saliency", action="store_true",
                        help="сразу включить карту внимания")
    parser.add_argument("--record", default=None, metavar="OUT",
                        help="headless-запись демонстрации в GIF/MP4 вместо окна")
    parser.add_argument("--frames", type=int, default=360,
                        help="сколько кадров записать (--record)")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="частота кадров в записанном файле")
    parser.add_argument("--keep-png", action="store_true",
                        help="не удалять PNG-кадры после сборки")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер со всеми подкомандами SPEC §15 (справка — на русском)."""
    parser = argparse.ArgumentParser(
        prog="python -m gdai",
        description=(
            "GDAI — нейросеть, которая учится проходить Geometry Dash. "
            "Зрение переводит любой дизайн в каноническую карту, политика "
            "играет только по этой карте."
        ),
        epilog="Начните с «python -m gdai selfcheck» — она проверит всю связку.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="store_true", help="показать версию пакета и выйти"
    )
    sub = parser.add_subparsers(dest="command", metavar="команда")

    # --- selfcheck ---------------------------------------------------------
    p = sub.add_parser(
        "selfcheck",
        help="быстрая проверка всей связки (< 60 с)",
        description=(
            "Прогоняет весь путь проекта: генерация уровня -> проверка "
            "проходимости -> рендер нескольких тем -> инвариантность карты -> "
            "шаги среды -> крошечное обучение зрения -> крошечное обучение "
            "политики -> агент целиком -> демо-кадр."
        ),
    )
    p.add_argument("--out", default=DEFAULT_SELFCHECK_DIR,
                   help="каталог для весов, метрик и демо-кадра")
    p.add_argument("--seed", type=int, default=0, help="seed всей проверки")
    p.add_argument("--device", default="cpu", help="cpu|cuda|auto")
    p.add_argument("--levels-dir", default=DEFAULT_LEVELS_DIR,
                   help="каталог с уровнями *.json для проверки проходимости")
    p.add_argument("--perception-steps", type=int, default=SELFCHECK_PERCEPTION_STEPS,
                   help="шагов крошечного обучения зрения")
    p.add_argument("--agent-steps", type=int, default=SELFCHECK_AGENT_STEPS,
                   help="шагов крошечного обучения политики")
    p.add_argument("--color", action="store_true", help="всегда красить отчёт")
    p.add_argument("--no-color", action="store_true", help="никогда не красить отчёт")
    p.set_defaults(func=cmd_selfcheck)

    # --- play --------------------------------------------------------------
    p = sub.add_parser(
        "play",
        help="играть самому (окно pygame)",
        description="Окно визуализатора под управлением человека: пробел — прыжок.",
    )
    p.add_argument("--level", default=None, help="уровень из файла *.json")
    p.add_argument("--difficulty", type=float, default=0.3, help="сложность 0..1")
    p.add_argument("--seed", type=int, default=0, help="seed уровня и темы")
    p.add_argument("--scale", type=int, default=3, help="увеличение панелей")
    p.add_argument("--fps", type=int, default=60, help="частота кадров окна")
    p.add_argument("--theme", default=None, help="фиксированная тема по имени")
    p.add_argument("--decoration", type=float, default=1.0, help="плотность декора 0..1")
    p.set_defaults(func=cmd_play)

    # --- watch -------------------------------------------------------------
    p = sub.add_parser(
        "watch",
        help="смотреть, что видит ИИ (три панели)",
        description=(
            "Кадр игры | предсказанная зрением карта | эталон с подсветкой "
            "ошибок. Без --policy/--perception используются случайные веса — "
            "связку видно и до обучения."
        ),
    )
    _add_viewer_args(p)
    p.set_defaults(func=cmd_watch)

    # --- gen-level ---------------------------------------------------------
    p = sub.add_parser(
        "gen-level",
        help="сгенерировать уровень и сохранить в JSON",
        description=("Процедурный уровень заданной сложности. Перед сохранением "
                     "проверяется проходимость поиском по кадрам."),
    )
    p.add_argument("--difficulty", type=float, default=0.4, help="сложность 0..1")
    p.add_argument("--out", default="levels/generated.json", help="куда сохранить *.json")
    p.add_argument("--seed", type=int, default=0, help="seed генерации")
    p.add_argument("--length", type=float, default=None,
                   help="целевая длина в тайлах (по умолчанию — от сложности)")
    p.add_argument("--name", default=None, help="имя уровня (по умолчанию — имя файла)")
    p.add_argument("--no-check", action="store_true",
                   help="не проверять проходимость (быстрее, но опаснее)")
    p.set_defaults(func=cmd_gen_level)

    # --- train-perception --------------------------------------------------
    p = sub.add_parser(
        "train-perception",
        help="обучить зрение (U-Net) на синтетике",
        description=("Supervised-обучение сегментации: симулятор рисует кадр со "
                     "случайным дизайном и идеальную разметку к нему. Валидация "
                     "идёт на отложенных темах — это честная проверка обобщения."),
    )
    p.add_argument("--steps", type=int, default=None, help="число шагов обучения")
    p.add_argument("--batch-size", type=int, default=None, help="размер батча")
    p.add_argument("--lr", type=float, default=None, help="скорость обучения")
    p.add_argument("--base-channels", type=int, default=None, help="ширина сети")
    p.add_argument("--depth", type=int, default=None, help="число уровней U-Net")
    p.add_argument("--val-every", type=int, default=None, help="шагов между валидациями")
    p.add_argument("--device", default=None, help="cpu|cuda|auto")
    p.add_argument("--out", default=None, help=f"каталог прогона ({DEFAULT_PERCEPTION_RUN})")
    p.add_argument("--no-augment", action="store_true", help="отключить аугментации кадра")
    p.set_defaults(func=cmd_train_perception)

    # --- train-agent -------------------------------------------------------
    p = sub.add_parser(
        "train-agent",
        help="обучить политику (PPO) на канонических картах",
        description=("PPO по семантическим картам. Политика не видит декораций "
                     "в принципе, поэтому переобучиться на дизайн не может."),
    )
    p.add_argument("--total-steps", type=int, default=2_000_000,
                   help="сколько кадров опыта собрать всего")
    p.add_argument("--num-envs", type=int, default=8, help="сколько сред идут параллельно")
    p.add_argument("--rollout-steps", type=int, default=256,
                   help="длина роллаута одной среды")
    p.add_argument("--lr", type=float, default=3e-4, help="скорость обучения")
    p.add_argument("--difficulty", type=float, default=0.3,
                   help="сложность (стартовая, если включён --curriculum)")
    p.add_argument("--curriculum", action="store_true",
                   help="учебный план: сложность растёт по мере успехов")
    p.add_argument("--level", default=None, help="учиться на фиксированном уровне *.json")
    p.add_argument("--max-steps", type=int, default=6000, help="лимит кадров на эпизод")
    p.add_argument("--semantic-noise", type=float, default=0.0,
                   help="доля испорченных пикселей карты (робастность к ошибкам зрения)")
    p.add_argument("--no-practice", action="store_true",
                   help="отключить practice-чекпойнты")
    p.add_argument("--seed", type=int, default=0, help="seed обучения")
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.add_argument("--out", default=DEFAULT_AGENT_RUN, help="каталог прогона")
    p.set_defaults(func=cmd_train_agent)

    # --- eval --------------------------------------------------------------
    p = sub.add_parser(
        "eval",
        help="оценить агента (доля прохождений и прогресс)",
        description=("Каждый эпизод начинается С НАЧАЛА уровня, поэтому "
                     "success_rate означает именно «прошёл уровень целиком». "
                     "Без --policy берутся случайные веса — так проверяют связку."),
    )
    p.add_argument("--policy", default=None, help="веса политики (runs/agent/best.pt)")
    p.add_argument("--perception", default=None,
                   help="веса зрения: с ними оценка идёт по предсказанной карте")
    p.add_argument("--use-perception", action="store_true",
                   help="включить зрение даже без весов (случайная сеть)")
    p.add_argument("--episodes", type=int, default=20, help="сколько эпизодов сыграть")
    p.add_argument("--difficulty", type=float, default=0.3, help="сложность 0..1")
    p.add_argument("--level", default=None, help="оценивать на уровне из файла *.json")
    p.add_argument("--max-steps", type=int, default=6000, help="лимит кадров на эпизод")
    p.add_argument("--stochastic", action="store_true",
                   help="выбирать действие выборкой, а не argmax")
    p.add_argument("--seed", type=int, default=0, help="seed уровней и тем")
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.set_defaults(func=cmd_eval)

    # --- plot --------------------------------------------------------------
    p = sub.add_parser(
        "plot",
        help="построить графики обучения из metrics.jsonl",
        description="Сетка кривых прогона (награда, прохождения, лоссы, KL) в один PNG.",
    )
    p.add_argument("--run", required=True,
                   help="каталог прогона (runs/agent) или файл *.jsonl")
    p.add_argument("--out", default="curves.png", help="куда сохранить PNG")
    p.add_argument("--title", default=None, help="заголовок графика")
    p.add_argument("--smooth", type=int, default=1,
                   help="окно скользящего среднего (1 = без сглаживания)")
    p.add_argument("--cols", type=int, default=3, help="сколько панелей в ряду")
    p.add_argument("--dpi", type=int, default=130, help="разрешение PNG")
    p.add_argument("--only-known", action="store_true",
                   help="рисовать только известные метрики")
    p.set_defaults(func=cmd_plot)

    # --- play-real ---------------------------------------------------------
    p = sub.add_parser(
        "play-real",
        help="играть в настоящую Geometry Dash (опционально)",
        description=("НЕ основной сценарий: нужны mss и pynput, запущенная игра "
                     "и ручная калибровка игрового поля. По умолчанию клавиши "
                     "не нажимаются — только наблюдение."),
    )
    p.add_argument("--policy", default=None, help="веса политики")
    p.add_argument("--perception", default=None, help="веса зрения")
    p.add_argument("--region", default=None, help="файл калибровки игрового поля")
    p.add_argument("--monitor", type=int, default=1, help="номер монитора")
    p.add_argument("--fps", type=float, default=60.0, help="частота цикла")
    p.add_argument("--seconds", type=float, default=120.0, help="сколько секунд играть")
    p.add_argument("--press", action="store_true",
                   help="РАЗРЕШИТЬ нажатия клавиш (по умолчанию только наблюдение)")
    p.add_argument("--quit-key", default="esc", help="клавиша аварийного выхода")
    p.add_argument("--stochastic", action="store_true",
                   help="выбирать действие выборкой")
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.set_defaults(func=cmd_play_real)

    # --- demo --------------------------------------------------------------
    p = sub.add_parser(
        "demo",
        help="записать демонстрацию (GIF/MP4) без окна",
        description=("Headless-запись трёх панелей: кадр, что видит ИИ, эталон. "
                     "Формат определяется расширением --out."),
    )
    p.add_argument("--out", default="demo.gif", help="куда сохранить (*.gif|*.mp4)")
    p.add_argument("--frames", type=int, default=300, help="сколько кадров записать")
    p.add_argument("--fps", type=int, default=30, help="частота кадров в файле")
    p.add_argument("--policy", default=None, help="веса политики")
    p.add_argument("--perception", default=None, help="веса зрения")
    p.add_argument("--level", default=None, help="уровень из файла *.json")
    p.add_argument("--difficulty", type=float, default=0.3, help="сложность 0..1")
    p.add_argument("--seed", type=int, default=0, help="seed уровня и темы")
    p.add_argument("--scale", type=int, default=3, help="увеличение панелей")
    p.add_argument("--theme", default=None, help="фиксированная тема по имени")
    p.add_argument("--decoration", type=float, default=1.0, help="плотность декора 0..1")
    p.add_argument("--no-perception", action="store_true",
                   help="показывать эталонную карту вместо предсказания")
    p.add_argument("--saliency", action="store_true", help="наложить карту внимания")
    p.add_argument("--keep-png", action="store_true", help="сохранить и PNG-кадры")
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Разобрать аргументы и выполнить команду; вернуть код возврата процесса."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "version", False):
        from gdai import __version__

        _emit(f"gdai {__version__}")
        return 0

    func: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    try:
        return int(func(args))
    except KeyboardInterrupt:  # pragma: no cover - интерактивный сценарий
        _emit("Прервано пользователем.")
        return 130


__all__ = ["build_parser", "main", "cmd_selfcheck"]


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
