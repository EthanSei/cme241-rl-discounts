# Repository Architecture & File Structure

## Purpose
This document serves as the **source of truth** for the project structure. All coding agents must adhere to this layout when creating new files or refactoring existing ones.

## 1. Directory Tree
```text
cme241-rl-discounts/
├── README.md                 # Project documentation (Install, Run, Results)
├── specs/                    # Context specs for coding agents (Architecture, Constraints)
├── pyproject.toml            # Dependencies (Poetry/Setuptools) & Project Config
├── Makefile                  # Makefile for scripting commands, set up, and running tests
├── .gitignore                # Excludes data/, .env, __pycache__, .DS_Store
├── configs/                  # Hyperparameter management (Hydra/YAML)
│   ├── env_config.yaml       # Environment params (max_steps, penalties, elasticity)
│   └── agent_config.yaml     # RL params (learning_rate, gamma, epsilon, buffer_size)
├── data/                     # DATA STORE (Local Only - .gitignore this)
│   ├── raw/                  # Original input CSVs (Immutable)
│   └── processed/            # Cleaned parquet/csv files ready for loading
├── notebooks/                # Experimental Analysis & Reports
│   ├── 01_data_analysis.ipynb
│   └── 02_dp_solution.ipynb
├── scripts/                  # Executable Entry Points
│   ├── train_rl.py           # Main training loop (loads Env + Agent)
│   ├── solve_dp.py           # Phase 2 DP solver execution
│   └── evaluate.py           # Inference & Plotting (loads saved models)
├── src/                      # Source Code (Installable Package)
│   └── discount_engine/      # Main Package Name
│       ├── __init__.py       # Exposes key classes
│       ├── agents/           # Decision Making Logic
│       │   ├── __init__.py
│       │   ├── base.py       # Abstract Base Class (ABC) for all agents
│       │   ├── dp_agent.py   # Dynamic Programming implementation (Phase 2)
│       │   └── q_learning.py # Tabular/Approx Q-Learning (Phase 3)
│       ├── envs/             # Gymnasium Environments
│       │   ├── __init__.py   # Registers env with Gym via `register()`
│       │   ├── market_env.py # The main Gym Class (inherits gym.Env)
│       │   └── wrappers.py   # Observation normalization, Reward scaling
│       ├── simulators/       # User/Market Logic (Decoupled from Gym)
│       │   ├── __init__.py
│       │   ├── customer.py   # Logic for User State
│       │   └── demand.py     # Conversion probability models (Sigmoid/Linear)
│       └── utils/            # Shared Helpers
│           ├── __init__.py
│           ├── data_loader.py
│           └── plotting.py
└── tests/
    ├── unit/                 # Your actual logic tests
    │   ├── test_env.py
    │   └── test_agent.py
    └── meta/                 # Your repo hygiene tests
        └── test_structure.py