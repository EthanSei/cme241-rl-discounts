.PHONY: install test lint clean help load preprocess

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make load       - Download Kaggle dataset to data/raw/"
	@echo "  make preprocess - Build processed data, sequences, features"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

load:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	python scripts/download_dataset.py $(if $(DATASET_ID),--dataset-id $(DATASET_ID),)

preprocess:
	python scripts/preprocess_data.py $(if $(PROCESSED_FORMAT),--format $(PROCESSED_FORMAT),) \
		$(if $(SEQUENCES_PATH),--sequences-path $(SEQUENCES_PATH),) \
		$(if $(FEATURES_PATH),--features-path $(FEATURES_PATH),)

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

