"""Тесты зрения (SPEC §10, §16).

Зрение — это функция «кадр -> каноническая карта», и от неё требуется три
вещи, которые здесь и проверяются:

* **контракт форм**: (B,3,72,128) float32 в [0,1] -> (B,10,72,128) логиты;
* **бюджет**: меньше 500k параметров, иначе сеть не будет успевать рядом с
  политикой и начнёт запоминать частности тем вместо форм;
* **обучаемость**: переобучение на одном сэмпле за 200 шагов даёт accuracy
  выше 0.95. Это не тест качества, а тест того, что архитектура и градиенты
  вообще работают: если сеть не может выучить один кадр наизусть, обучать её
  на миллионах бессмысленно.

Плюс два свойства, специфичных для этого проекта: нормировка НЕ батчевая
(в бою кадры приходят по одному) и предсказание для батча 1 совпадает с
предсказанием в батче.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gdai.constants import NUM_CLASSES, OBS_H, OBS_W  # noqa: E402
from gdai.env.level import Level  # noqa: E402
from gdai.env.physics import PlayerState  # noqa: E402
from gdai.env.render import Renderer  # noqa: E402
from gdai.env.semantic import render_semantic  # noqa: E402
from gdai.perception.model import (  # noqa: E402
    PerceptionNet,
    build_perception_net,
    input_shape,
    resolve_device,
)

MAX_PARAMETERS: int = 500_000
OVERFIT_STEPS: int = 200
OVERFIT_TARGET: float = 0.95


@pytest.fixture(scope="module")
def sample(demo_level_module: Level) -> tuple[np.ndarray, np.ndarray]:
    """Пара (красивый кадр, каноническая карта) — ровно то, на чём учится зрение."""
    state = PlayerState(x=11.0, y=1.6, vy=-3.0, mode="cube", on_ground=False)
    renderer = Renderer(seed=0)
    frame = renderer.render(demo_level_module, state, 0)
    label = render_semantic(demo_level_module, state)
    renderer.close()
    return frame, label


@pytest.fixture(scope="module")
def demo_level_module() -> Level:
    """Копия `demo_level` с областью видимости модуля (рендер стоит денег)."""
    from gdai.env.level import LevelObject

    objects = [
        LevelObject("block", 8.0, 0.5),
        LevelObject("platform", 9.0, 2.75),
        LevelObject("spike", 12.0, 0.5),
        LevelObject("saw", 15.0, 1.0),
        LevelObject("orb_yellow", 13.5, 2.0),
        LevelObject("pad_pink", 16.0, 0.25),
        LevelObject("goal", 30.0, 6.0),
    ]
    return Level(name="demo", length=30.0, objects=objects, ceiling_y=12.0)


# ---------------------------------------------------------------------------
# контракт сети
# ---------------------------------------------------------------------------
def test_forward_shapes() -> None:
    """Вход (B,3,72,128) -> выход (B,10,72,128) логитов."""
    net = PerceptionNet()
    assert input_shape() == (3, OBS_H, OBS_W)
    x = torch.zeros(2, 3, OBS_H, OBS_W)
    out = net(x)
    assert out.shape == (2, NUM_CLASSES, OBS_H, OBS_W)
    assert out.dtype == torch.float32


def test_forward_rejects_wrong_input() -> None:
    """Неверная форма входа — понятная ошибка, а не молчаливый мусор."""
    net = PerceptionNet()
    with pytest.raises(ValueError):
        net(torch.zeros(3, OBS_H, OBS_W))
    with pytest.raises(ValueError):
        net(torch.zeros(1, 4, OBS_H, OBS_W))


def test_parameter_budget() -> None:
    """Меньше 500k параметров (SPEC §10) — сеть обязана быть маленькой."""
    net = PerceptionNet()
    count = net.count_parameters()
    assert 0 < count < MAX_PARAMETERS, f"параметров {count}"
    assert build_perception_net().count_parameters() == count


def test_no_batchnorm_only_group_norm() -> None:
    """BatchNorm запрещён: в бою кадры идут по одному, и его статистика вырождается."""
    net = PerceptionNet()
    assert not any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules()
    )
    assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())


def test_batch_size_does_not_change_output() -> None:
    """Кадр в батче 1 и в батче N обрабатывается одинаково (следствие GroupNorm)."""
    torch.manual_seed(0)
    net = PerceptionNet().eval()
    x = torch.rand(1, 3, OBS_H, OBS_W)
    with torch.no_grad():
        alone = net(x)
        in_batch = net(torch.cat([x, torch.rand(3, 3, OBS_H, OBS_W)], dim=0))[:1]
    assert torch.allclose(alone, in_batch, atol=1e-4)


def test_arbitrary_input_size_supported() -> None:
    """Сеть работает и на кадре, сторона которого не делится на 2^depth."""
    net = PerceptionNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 45, 77))
    assert out.shape == (1, NUM_CLASSES, 45, 77)


def test_resolve_device_returns_cpu_without_cuda() -> None:
    """`device="auto"` разрешается в реальное устройство и принимает явное имя."""
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")
    assert resolve_device("cpu").type == "cpu"


# ---------------------------------------------------------------------------
# инференс
# ---------------------------------------------------------------------------
def test_predict_returns_class_map(sample: tuple[np.ndarray, np.ndarray]) -> None:
    """`predict` даёт (H,W) uint8 с классами в допустимом диапазоне."""
    frame, _label = sample
    net = PerceptionNet()
    pred = net.predict(frame)
    assert pred.shape == (OBS_H, OBS_W)
    assert pred.dtype == np.uint8
    assert int(pred.max()) < NUM_CLASSES


def test_predict_batch_matches_predict(sample: tuple[np.ndarray, np.ndarray]) -> None:
    """Батчевое предсказание совпадает с поштучным."""
    frame, _label = sample
    net = PerceptionNet()
    batch = np.stack([frame, frame[::-1].copy()])
    predicted = net.predict_batch(batch)
    assert predicted.shape == (2, OBS_H, OBS_W)
    assert np.array_equal(predicted[0], net.predict(frame))


def test_predict_keeps_training_flag(sample: tuple[np.ndarray, np.ndarray]) -> None:
    """Предсказание не оставляет сеть в режиме eval посреди обучения."""
    frame, _label = sample
    net = PerceptionNet()
    net.train()
    net.predict(frame)
    assert net.training is True


def test_predict_rejects_bad_shapes() -> None:
    """Кадр неверной формы отвергается явно."""
    net = PerceptionNet()
    with pytest.raises(ValueError):
        net.predict(np.zeros((OBS_H, OBS_W), dtype=np.uint8))
    with pytest.raises(ValueError):
        net.predict_batch(np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# обучаемость
# ---------------------------------------------------------------------------
def test_overfits_single_sample(sample: tuple[np.ndarray, np.ndarray]) -> None:
    """200 шагов на одном сэмпле дают accuracy > 0.95 (SPEC §16).

    Зачем именно переобучение: это самая быстрая проверка того, что сеть,
    инициализация, лосс и градиенты образуют рабочую связку. Настоящее
    качество меряет обучение на held-out темах, а не тест.
    """
    frame, label = sample
    assert len(np.unique(label)) >= 3, "в сэмпле должно быть несколько классов"

    torch.manual_seed(0)
    net = PerceptionNet()
    x = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    y = torch.from_numpy(label.astype(np.int64))[None]
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    net.train()
    for _ in range(OVERFIT_STEPS):
        loss = torch.nn.functional.cross_entropy(net(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)

    net.eval()
    with torch.no_grad():
        accuracy = float((net(x).argmax(dim=1) == y).float().mean())
    assert accuracy > OVERFIT_TARGET, f"accuracy {accuracy:.4f} после {OVERFIT_STEPS} шагов"


# ---------------------------------------------------------------------------
# честность валидации
# ---------------------------------------------------------------------------
def test_theme_split_is_honest() -> None:
    """Обучающие и отложенные темы не пересекаются (SPEC §10).

    Зачем тестом: пересечение не уронит обучение, а просто превратит
    «обобщение на новый дизайн» в «запоминание знакомого» — и метрика станет
    ложью, которую никто не заметит.
    """
    dataset = pytest.importorskip("gdai.perception.dataset")
    dataset.check_theme_split()
    train = {t.name for t in dataset.train_themes()}
    held = {t.name for t in dataset.held_out_themes()}
    assert train and held
    assert not (train & held)


@pytest.mark.slow
def test_inference_throughput(sample: tuple[np.ndarray, np.ndarray]) -> None:
    """Зрение обязано успевать в реальном времени на CPU (SPEC §10).

    Порог намеренно ниже заявленных 100 кадров/с: тест не должен мигать на
    загруженном CI-раннере, а его задача — ловить регрессию в разы, а не
    измерять производительность.
    """
    frame, _label = sample
    net = PerceptionNet().eval()
    batch = np.stack([frame] * 32)
    net.predict_batch(batch)          # прогрев: первый прогон включает аллокации
    start = time.perf_counter()
    net.predict_batch(batch)
    fps = 32 / max(time.perf_counter() - start, 1e-9)
    assert fps > 40.0, f"всего {fps:.0f} кадров/с"
