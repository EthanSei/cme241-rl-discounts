# Project Roadmap: Discount Addiction RL

This document outlines the strategic phases for building the Reinforcement Learning Pricing Engine. The project is divided into three distinct phases aligning with the CME 241 course structure, evolving from a theoretical DP model to a production-grade RL agent.

---

## 🚩 Phase 1: Infrastructure & Data Foundations
**Goal:** Establish the ground truth data, build the simulation engine, and define the MDP.

### 1.1 Data Pipeline (The "Truth")
- [x] **Ingest:** Load Dunnhumby `transaction_data` and `product` CSVs.
- [x] **Filter:** Isolate a single high-frequency product category (e.g., "Soft Drinks") to ensure sufficient signal.
- [x] **User Sequencing:** Aggregate raw logs into `UserSequence` objects (Time-series of Price Paid vs. Discount).
- [x] **Elasticity Modeling:** Fit a simple demand curve to the real data to extract `beta` (price sensitivity) and `alpha` (memory decay) parameters.

### 1.2 The Simulator (The "World")
- [ ] **Simulated User:** Implement `src/discount_engine/simulators/customer.py` using the parameters derived from 1.1.
- [ ] **Validation:** Verify that the Simulated User "addicts" correctly (i.e., purchase probability drops after repeated discounts are removed).

### 1.3 Environment Setup (The "Interface")
- [ ] **Gym Wrapper:** Implement `MarketEnv(gym.Env)`.
- [ ] **Spaces:** Define `MultiDiscrete` action space (Buy/No-Buy, Discount Depth) and `Box` observation space.
- [ ] **Unit Tests:** Verify `env.reset()` and `env.step()` adhere to the Gymnasium API standard.

---

## 🚩 Phase 2: Dynamic Programming (The Theoretical Solver)
**Goal:** Solve the problem mathematically using "God Mode" (perfect knowledge of dynamics) on a simplified state space.

### 2.1 Discretization
- [ ] **State Binning:** Reduce continuous `ReferencePrice` to 5 discrete buckets (Very Low -> Very High).
- [ ] **Transition Matrix:** Explicitly calculate the probability matrix $P(s' | s, a)$ using the Simulator logic.

### 2.2 The Solver
- [ ] **Value Iteration:** Implement standard VI to find the optimal policy $\pi^*$.
- [ ] **Visual Analysis:** Generate a heatmap of the Value Function $V(s)$. *Expectation: The value should be lower in "addicted" states.*

---

## 🚩 Phase 3: Deep Reinforcement Learning (The Production Agent)
**Goal:** Train a model-free agent that learns the optimal policy without knowing the user's hidden physics.

### 3.1 The Baselines (The "Control Group")
- [ ] **Random Agent:** Benchmark random actions.
- [ ] **Constant Policy:** Benchmark "Always 10% Off" and "Never Discount."
- [ ] **Rule-Based:** Benchmark a simple heuristic (e.g., "Discount if inactive for 7 days").

### 3.2 The RL Agent
- [ ] **Implementation:** specific PPO or DQN agent using `Stable Baselines3`.
- [ ] **Reward Shaping:** Introduce the **Engagement Bonus ($\lambda$)** to solve reward sparsity.
- [ ] **Training Loop:** Train on the Simulator for 1M+ timesteps.

### 3.3 Evaluation & Tuning
- [ ] **Hyperparameter Sweep:** Tune `learning_rate`, `gamma`, and `ent_coef` (exploration).
- [ ] **A/B Simulation:** Run the trained Policy vs. the Baseline on 1,000 unseen synthetic users.
- [ ] **Business Metrics:** Report `Total Revenue`, `Conversion Rate`, and `Margin` (not just "Mean Reward").

---

## 🚩 Phase 4: Final Reporting & Polish
**Goal:** Synthesize findings into the final course deliverable.

- [ ] **Code Cleanup:** Add docstrings and type hints to `src/`.
- [ ] **Visualization:** Create standard plots:
    - *Learning Curves (Reward vs. Time)*
    - *Policy Inspection (What does the agent do when Ref Price is high?)*
    - *Pareto Frontier (Revenue vs. Conversion)*
- [ ] **Final Paper:** Draft the report detailing the "Discount Addiction" constraint and the impact of Reward Shaping.