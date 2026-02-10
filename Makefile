.PHONY: install test lint clean help load preprocess sims

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make load       - Download Kaggle dataset to data/raw/"
	@echo "  make preprocess - Build processed data, sequences, features"
	@echo "  make sims       - Run customer simulation validation"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

load:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	python scripts/data/download_dataset.py $(if $(DATASET_ID),--dataset-id $(DATASET_ID),)

preprocess:
	python scripts/data/preprocess_data.py $(if $(PROCESSED_FORMAT),--format $(PROCESSED_FORMAT),)

sims:
	python scripts/rl_run_simulation.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
