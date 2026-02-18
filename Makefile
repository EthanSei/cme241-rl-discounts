.PHONY: install test lint clean help load preprocess calibrate sims

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make load       - Download Kaggle dataset to data/raw/"
	@echo "  make preprocess - Build processed data, sequences, features"
	@echo "  make calibrate  - Calibrate MDP params from configs/data/calibration.yaml"
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

calibrate:
	python scripts/data/calibrate_mdp_params.py \
		--config $(if $(CALIBRATION_CONFIG),$(CALIBRATION_CONFIG),configs/data/calibration.yaml) \
		$(if $(PROCESSED_DIR),--processed-dir $(PROCESSED_DIR),) \
		$(if $(PARAMS_OUTPUT),--output-path $(PARAMS_OUTPUT),) \
		$(if $(N_CATEGORIES),--n-categories $(N_CATEGORIES),) \
		$(if $(CATEGORY_COLUMN),--category-column $(CATEGORY_COLUMN),) \
		$(if $(DELTA),--delta $(DELTA),) \
		$(if $(GAMMA),--gamma $(GAMMA),) \
		$(if $(INACTIVITY_HORIZON),--inactivity-horizon $(INACTIVITY_HORIZON),) \
		$(if $(VALIDATION_FRACTION),--validation-fraction $(VALIDATION_FRACTION),)

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
