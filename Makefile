PYTHON ?= python
export PYTHONPATH := $(CURDIR)

# Fast tests: the hybrid connector logic, byte store, codecs and the CPU demo.
HYBRID_TESTS := tests/test_hybrid_layout.py tests/test_hybrid_logic.py \
	tests/test_hybrid_store.py tests/test_hybrid_gpu_codec.py \
	tests/test_memory_connector.py tests/test_cpu_quickstart.py \
	tests/test_qwen_next_acceptance.py

.PHONY: help install test test-hybrid lint fmt demo

help:
	@echo "install      install CPU dev dependencies into the active environment"
	@echo "test         full CPU suite (CUDA/nvCOMP cases skip)"
	@echo "test-hybrid  hybrid connector + byte store tests only (seconds)"
	@echo "lint         ruff check and format check"
	@echo "fmt          apply ruff fixes and formatting"
	@echo "demo         run the CPU storage-contract demo into work/cpu-demo"

install:
	$(PYTHON) -m pip install -r requirements/test.txt -r requirements/lint.txt

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider tests

test-hybrid:
	$(PYTHON) -m pytest -q -p no:cacheprovider $(HYBRID_TESTS)

lint:
	$(PYTHON) -m ruff check aibrix_kvcache tools
	$(PYTHON) -m ruff format --check aibrix_kvcache tools

fmt:
	$(PYTHON) -m ruff check --fix aibrix_kvcache tools
	$(PYTHON) -m ruff format aibrix_kvcache tools

demo:
	rm -rf work/cpu-demo
	$(PYTHON) -m tools.cpu_quickstart --output work/cpu-demo
