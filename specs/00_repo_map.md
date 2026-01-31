# Repo Map
## File Tree
```
cme241-rl-discounts
├── .github
│   └── workflows
│       └── python-app.yml
├── configs
│   ├── agent_config.yaml
│   └── env_config.yaml
├── data
│   ├── processed
│   └── raw
├── notebooks
│   ├── 01_data_analysis.ipynb
│   └── 02_dp_solution.ipynb
├── scripts
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── preprocess_data.py
│   ├── solve_dp.py
│   └── train_rl.py
├── specs
│   ├── 00_repo_map.md
│   ├── 01_project_core.md
│   ├── 02_roadmap.md
│   ├── 03_memory.md
│   ├── project_guidelines.md
│   └── active_brief.md
├── src
│   ├── discount_engine
│   │   ├── __init__.py
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── dp_agent.py
│   │   │   └── q_learning.py
│   │   ├── envs
│   │   │   ├── __init__.py
│   │   │   ├── market_env.py
│   │   │   └── wrappers.py
│   │   ├── simulators
│   │   │   ├── __init__.py
│   │   │   ├── customer.py
│   │   │   └── demand.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── data_loader.py
│   │       ├── elasticity.py
│   │       ├── feature_engineering.py
│   │       ├── plotting.py
│   │       └── sequence_builder.py
├── tests
│   ├── meta
│   │   ├── __init__.py
│   │   └── test_structure.py
│   ├── unit
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   ├── test_data_loader.py
│   │   ├── test_elasticity.py
│   │   ├── test_env.py
│   │   ├── test_feature_engineering.py
│   │   └── test_sequence_builder.py
│   ├── __init__.py
│   └── conftest.py
├── .cursorrules
├── .env
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Python Source Files
- `src/discount_engine/__init__.py`
- `src/discount_engine/agents/__init__.py`
- `src/discount_engine/agents/base.py`
- `src/discount_engine/agents/dp_agent.py`
- `src/discount_engine/agents/q_learning.py`
- `src/discount_engine/envs/__init__.py`
- `src/discount_engine/envs/market_env.py`
- `src/discount_engine/envs/wrappers.py`
- `src/discount_engine/simulators/__init__.py`
- `src/discount_engine/simulators/customer.py`
- `src/discount_engine/simulators/demand.py`
- `src/discount_engine/utils/__init__.py`
- `src/discount_engine/utils/data_loader.py`
- `src/discount_engine/utils/elasticity.py`
- `src/discount_engine/utils/feature_engineering.py`
- `src/discount_engine/utils/plotting.py`
- `src/discount_engine/utils/sequence_builder.py`

## Component Interaction
- `simulators` must NEVER import `envs` or `agents`.

# Repository Map & Standards

## 3. Coding Standards (The Hygiene Rules)
* **Type Hinting:** Strict typing required for all function arguments and return values. Use `typing.Optional` for non-guaranteed returns.
* **Docstrings:** All public classes and methods must use **Google Style** docstrings.
* **Path Handling:** ALWAYS use `pathlib.Path`, NEVER use `os.path.join`.
* **Imports:**
    * Absolute imports only (e.g., `from pricing_engine.utils import ...`).
    * Group standard library first, then third-party (numpy/gym), then local imports.
* **Testing:**
    * No hardcoded numbers in tests; use constants.
    * Use `pytest.fixture` for setup logic.
