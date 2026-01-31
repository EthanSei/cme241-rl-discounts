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
├── configs/              # Hyperparameter configs (YAML)
├── data/                 # Data storage (gitignored)
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Executable entry points
├── specs/                # Architecture documentation
├── src/discount_engine/  # Main package
│   ├── agents/           # DP and RL agent implementations
│   ├── envs/             # Gymnasium environment
│   ├── simulators/       # Customer/demand models
│   └── utils/            # Shared utilities
└── tests/                # Unit and meta tests
```

## Usage

### Training an RL Agent

```bash
python scripts/train_rl.py
```

### Running the DP Solver

```bash
python scripts/solve_dp.py
```

### Evaluating a Trained Model

```bash
python scripts/evaluate.py
```

### Make Targets (Inputs & Outputs)
- `make load`
  - **Inputs:** `.env` (optional), `DATASET_ID` (optional)
  - **Outputs:** Raw CSVs in `data/raw/`
  - **Notes:** Downloads the dataset via `scripts/download_dataset.py`.
- `make preprocess`
  - **Overview:** Standardizes raw tables, builds per-household sequences with
    coupon flags, generates transaction-level modeling features, and estimates
    elasticity parameters for the simulator.
  - **Inputs:** Raw CSVs in `data/raw/`
  - **Outputs:**
    - Processed tables in `data/processed/`
    - `data/processed/training_sequences.(parquet|csv)`
    - `data/processed/training_features.(parquet|csv)`
    - `data/processed/elasticity_params.json`
  - **Notes:** Uses `scripts/preprocess_data.py`. See `--feature-sets`,
    `--feature-columns`, `--drop-feature-columns`, `--sequence-check`,
    `--category`, `--category-column`, `--elasticity-path`.
- `make test`
  - **Inputs:** `tests/`, `src/`
  - **Outputs:** Pytest results in terminal.
- `make lint`
  - **Inputs:** `src/`, `tests/`
  - **Outputs:** Ruff and mypy results in terminal.

### Table Schemas (Preprocessed Outputs)
#### Training Sequences (`data/processed/training_sequences.*`)
- **Base columns:** All standardized columns from `transaction_data`.
- **Derived identifiers:** `transaction_id`, `sequence_index`
- **Coupon availability:** `coupon_available`, `coupon_upc`, `campaign`
- **Coupon usage:** `coupon_used`, `coupon_upc_used`

#### Training Features (`data/processed/training_features.*`)
- **Base columns:** All standardized columns from `transaction_data`.
- **Derived numeric features:** `total_discount`, `price_per_unit`,
  `discount_rate`, `promo_flag`
- **Product attributes (join):** `department`, `commodity_desc`, `brand`
- **Household attributes (join):** `age_desc`, `income_desc`, `homeowner_desc`

#### Elasticity Parameters (`data/processed/elasticity_params.json`)
- **Values:** `alpha`, `beta`, `intercept`, `memory_coef`, `mse`, `n_samples`
- **Metadata:** `category`, `category_column`

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

- `env_config.yaml` — Environment parameters (max steps, base price, elasticity)
- `agent_config.yaml` — Agent parameters (learning rate, gamma, epsilon)

## License

MIT
