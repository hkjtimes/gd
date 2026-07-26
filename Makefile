# Makefile GDAI — короткие имена для длинных команд.
#
# Зачем он нужен: у проекта пять сценариев (проверка, два обучения, просмотр,
# демо), и у каждого свои пути к весам и каталогам прогонов. Держать их в
# голове (и одинаково писать в README, CI и терминале) невозможно — поэтому
# все умолчания живут здесь, в одном месте, и переопределяются переменными:
#
#   make train-agent TOTAL_STEPS=5000000 DEVICE=cuda
#   make demo DEMO_OUT=docs/demo.mp4 FRAMES=600
#
# Ни одна цель не требует установки пакета: python -m gdai работает из корня
# репозитория как есть.

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip

# --- каталоги прогонов и артефактов ---
PERCEPTION_OUT ?= runs/perception
AGENT_OUT      ?= runs/agent
SELFCHECK_OUT  ?= runs/selfcheck
DEMO_OUT       ?= demo.gif
CURVES_OUT     ?= curves.png

# --- гиперпараметры «по умолчанию для человека» ---
# Значения подобраны так, чтобы команду можно было запустить, не читая SPEC:
# зрение за десятки минут доходит до вменяемого mIoU, политика — до первых
# прохождений лёгких уровней.
PERCEPTION_STEPS ?= 4000
TOTAL_STEPS      ?= 2000000
DIFFICULTY       ?= 0.3
FRAMES           ?= 300
DEVICE           ?= auto
SEED             ?= 0

# Веса, которые подхватывают watch/demo/eval. Их отсутствие не ошибка:
# визуализатор честно скажет «случайная» и всё равно покажет связку.
POLICY     ?= $(AGENT_OUT)/best.pt
PERCEPTION ?= $(PERCEPTION_OUT)/best.pt

PYTEST_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help install selfcheck train-perception train-agent watch test demo plot eval clean

help:  ## показать список целей
	@echo "GDAI — доступные команды:"
	@echo "  make install           поставить зависимости (numpy, torch, pygame, matplotlib)"
	@echo "  make selfcheck         быстрая проверка всей связки (< 60 с)"
	@echo "  make train-perception  обучить зрение (U-Net) на синтетике"
	@echo "  make train-agent       обучить политику (PPO) на канонических картах"
	@echo "  make watch             окно «что видит ИИ»"
	@echo "  make demo              записать демонстрацию в $(DEMO_OUT)"
	@echo "  make eval              оценить агента на процедурных уровнях"
	@echo "  make plot              графики обучения из metrics.jsonl"
	@echo "  make test              прогнать pytest"
	@echo "  make clean             удалить прогоны, кэши и медиа-выхлоп"
	@echo ""
	@echo "Переменные: PYTHON DEVICE SEED DIFFICULTY TOTAL_STEPS PERCEPTION_STEPS"
	@echo "            POLICY PERCEPTION *_OUT FRAMES PYTEST_ARGS"

install:  ## зависимости проекта + dev-инструменты
	$(PIP) install -e ".[dev]"

selfcheck:  ## весь путь: уровень -> рендер -> среда -> зрение -> политика -> демо-кадр
	$(PYTHON) -m gdai selfcheck --out $(SELFCHECK_OUT) --seed $(SEED)

train-perception:  ## зрение: кадр с любым дизайном -> каноническая карта
	$(PYTHON) -m gdai train-perception \
		--steps $(PERCEPTION_STEPS) \
		--device $(DEVICE) \
		--out $(PERCEPTION_OUT)

train-agent:  ## политика: PPO с учебным планом по сложности
	$(PYTHON) -m gdai train-agent \
		--total-steps $(TOTAL_STEPS) \
		--curriculum \
		--difficulty $(DIFFICULTY) \
		--device $(DEVICE) \
		--seed $(SEED) \
		--out $(AGENT_OUT)

watch:  ## интерактивно: кадр | что видит ИИ | эталон и разница
	$(PYTHON) -m gdai watch \
		--policy $(POLICY) \
		--perception $(PERCEPTION) \
		--difficulty $(DIFFICULTY) \
		--seed $(SEED) \
		--device $(DEVICE)

demo:  ## headless-запись демонстрации (GIF/MP4 по расширению DEMO_OUT)
	$(PYTHON) -m gdai demo \
		--out $(DEMO_OUT) \
		--frames $(FRAMES) \
		--policy $(POLICY) \
		--perception $(PERCEPTION) \
		--difficulty $(DIFFICULTY) \
		--seed $(SEED) \
		--device $(DEVICE)

eval:  ## честная оценка: каждый эпизод с начала уровня
	$(PYTHON) -m gdai eval \
		--policy $(POLICY) \
		--difficulty $(DIFFICULTY) \
		--seed $(SEED) \
		--device $(DEVICE)

plot:  ## кривые обучения политики в $(CURVES_OUT)
	$(PYTHON) -m gdai plot --run $(AGENT_OUT) --out $(CURVES_OUT)

test:  ## pytest (тесты держатся в пределах ~3 минут на CPU)
	$(PYTHON) -m pytest $(PYTEST_ARGS)

clean:  ## удалить прогоны, кэши и медиа-выхлоп (уровни и код не трогаются)
	rm -rf runs $(CURVES_OUT) $(DEMO_OUT) demo frames
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[co]" -delete
