# Makefile for activation-ckpt-offload

PY     := uv run python
UV     := uv

# Default args for quick runs
TOKENS ?= 12000
LAYERS ?= 12
HIDDEN ?= 1024
HEAD   ?= 64
FF     ?= 4096
BATCH  ?= 48
STEPS  ?= 3
DEVICE ?= cuda

.PHONY: setup sync lint fmt test bench timeline run cpu cuda clean

setup:
	@echo "Installing uv (if missing)"; \
	curl -LsSf https://astral.sh/uv/install.sh | sh || true

sync:
	$(UV) sync

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .

bench:
	$(PY) -m activation_ckpt_offload --device $(DEVICE) \
	  --total_tokens $(TOKENS) --num_layers $(LAYERS) \
	  --hidden_dim $(HIDDEN) --head_dim $(HEAD) --ff_dim $(FF) \
	  --batch_size $(BATCH) --steps $(STEPS)

timeline:
	$(PY) -m activation_ckpt_offload --device $(DEVICE) \
	  --total_tokens $(TOKENS) --num_layers $(LAYERS) \
	  --hidden_dim $(HIDDEN) --head_dim $(HEAD) --ff_dim $(FF) \
	  --batch_size $(BATCH) --steps 1 --timeline-only

run: bench timeline

cpu:
	$(MAKE) bench DEVICE=cpu

cuda:
	$(MAKE) bench DEVICE=cuda

clean:
	rm -rf .venv .uv *.png *.html **/__pycache__ .pytest_cache dist build
