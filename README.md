# Discount Engine

![CI](https://github.com/EthanSei/cme241-rl-discounts/actions/workflows/python-app.yml/badge.svg)

**Reinforcement Learning for Discount Pricing Optimization**

A Stanford CME 241 (Winter 2026) project applying stochastic control and reinforcement learning to optimize discount pricing strategies for personal finance decisions.

## Project Overview

This project formulates discount pricing as a Markov Decision Process (MDP) and implements both Dynamic Programming (DP) and Reinforcement Learning (RL) solutions to find optimal discount policies that maximize long-term utility.

### Phases

| Phase | Technique | Target | Due Date |
|-------|-----------|--------|----------|
| 1 | Problem Definition | MDP Formulation | Feb 9 |
| 2 | Dynamic Programming | Simplified Version | Feb 23 |
| 3 | Reinforcement Learning | Realistic Version | Mar 16 |

## Installation

**Requirements:** Python 3.12+

```bash
# Clone the repository
git clone https://github.com/your-username/cme241-rl-discounts.git
cd cme241-rl-discounts

# Install in development mode
pip install -e ".[dev]"
```

## Project Structure

```
cme241-rl-discounts/
├── configs/              # Namespaced configs (data/dp/rl)
├── data/                 # Data storage (gitignored)
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Executable entry points
├── specs/                # Architecture documentation
├── src/discount_engine/  # Main package
│   ├── core/             # Shared MDP logic
│   ├── dp/               # Version C (DP) components
│   ├── rl/               # Version B (RL) components
│   └── utils/            # Shared utilities
└── tests/                # Core/DP/RL test suites
```

## Usage

### Training an RL Agent

```bash
python scripts/rl_train.py
```

### Running the DP Solver

```bash
python scripts/dp_solve.py --n-categories 3 --tag phase2
```

### Validating a DP Run

```bash
python scripts/dp_validate.py --run-dir runs/dp/<timestamp_tag>
```

### Evaluating a DP Run

```bash
python scripts/dp_evaluate.py --run-dir runs/dp/<timestamp_tag>
```

### Evaluating a Trained Model

```bash
python scripts/rl_evaluate.py
```

### Make Targets (Inputs & Outputs)
- `make load`
  - **Inputs:** `.env` (optional), `DATASET_ID` (optional)
  - **Outputs:** Raw CSVs in `data/raw/`
  - **Notes:** Downloads the dataset via `scripts/data/download_dataset.py`.
- `make preprocess`
  - **Overview:** Standardizes raw tables and writes cleaned artifacts to
    `data/processed/`.
  - **Inputs:** Raw CSVs in `data/raw/`
  - **Outputs:** Processed tables in `data/processed/`
  - **Notes:** Uses `scripts/data/preprocess_data.py`.
- `make test`
  - **Inputs:** `tests/`, `src/`
  - **Outputs:** Pytest results in terminal.
- `make lint`
  - **Inputs:** `src/`, `tests/`
  - **Outputs:** Ruff and mypy results in terminal.

### Parameter Artifact
Calibrated parameter artifact target path:
- `data/processed/mdp_params.yaml`

### DP Run Artifacts
DP solve/evaluate/validate outputs are written to:
- `runs/dp/<timestamp>_<tag>/`

Each run contains:
- `config_resolved.yaml`
- `values.json`
- `policy.json`
- `q_values.json`
- `solver_metrics.json`
- `policy_table.csv`
- `quality_report.json`
- `quality_warnings.json`
- `evaluation_summary.json`

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/
```

## Configuration

Environment and agent hyperparameters are managed via YAML files in `configs/`:

- `configs/data/calibration.yaml` — Dataset calibration inputs/outputs
- `configs/dp/solver.yaml` — DP solver parameters
- `configs/rl/env.yaml` — RL environment parameters
- `configs/rl/agent.yaml` — RL agent parameters

## License

MIT
