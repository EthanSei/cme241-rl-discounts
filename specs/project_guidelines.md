# Project Context: Stanford CME 241 (Winter 2026)

## Metadata
* **Course:** CME 241 - Stochastic Control and Reinforcement Learning
* **Instructor:** Prof. Ashwin Rao
* **Project Topic:** Pricing Discount Optimization
* **Total Grade Weight:** 50%
* **Goal:** Maximize aggregated Utility of Consumption using RL.

---

## 1. Project Overview & Objective

**Broad Topic:** Pricing Discount Optimization.
The goal is to make a sequence of optimal decisions about when to offer users discounts and how large those discounts should be.

**Core Problem:**
* **Inputs:** User context and market signals (seasonality, price sensitivity, campaign constraints).
* **Decisions:** Timing and magnitude of discount offers to users.
* **Optimization Target:** Maximize long-run revenue and user engagement utility.
* **Methodology:** Stochastic Control and Reinforcement Learning (RL).

**Success Criteria:**
1.  Identify a problem specialization (e.g., Retirement Planning, Career Path Optimization).
2.  Ensure the problem is practically relevant but mathematically tractable.
3.  Formulate the problem using the `MarkovDecisionProcess` framework.

---

## 2. Project Roadmap & Deliverables

The project is divided into three distinct phases, moving from definition to simplified solution (DP) to realistic solution (RL).

### Phase 1: Problem Definition
* **Due Date:** February 9
* **Weight:** 10 Points
* **Key Deliverable:** Mathematical formulation and `MarkovDecisionProcess` class code specification.

**Requirements:**
1.  **Describe the Specialization:** Define the specific finance problem and its practical relevance.
2.  **Code Specification:** Create a `MarkovDecisionProcess` class.
3.  **Define 3 Problem Versions:**
    * **Version A (Ideal/Commercial):** The full-blown problem you would solve with unlimited time/resources.
    * **Version B (Phase 3 Target):** A realistic but slightly diluted version solvable by the end of the course using RL.
    * **Version C (Phase 2 Target):** A highly simplified version solvable via Dynamic Programming (DP) within the first 4 weeks.

### Phase 2: Simplified Solution (DP/ADP)
* **Due Date:** February 23
* **Weight:** 15 Points
* **Technique:** Dynamic Programming (DP) or Approximate Dynamic Programming (ADP).

**Requirements:**
* Solve **Version C** (defined in Phase 1).
* **Constraints:** Keep state/action space small to ensure convergence in reasonable time.
* **Learning Goal:** Experience the trade-off between practical detail and computational tractability (Curse of Dimensionality / Curse of Modeling).

### Phase 3: Realistic Solution (RL)
* **Presentation Date:** March 13
* **Submission Due Date:** March 16
* **Weight:** 25 Points
* **Technique:** Reinforcement Learning (RL).

**Requirements:**
1.  **Simulation:** Generate sampling traces using the `MarkovDecisionProcess` setup from Phase 1.
2.  **Target Problem:** Solve **Version B** (Realistic version).
3.  **Algorithm:** Identify and implement an appropriate RL algorithm (standard or modified).
4.  **Implementation:**
    * Validate with course-provided RL code if necessary.
    * **Recommended:** Use open-source RL libraries built for performance and scale.
5.  **Deliverables:** Final Code and Paper.

---

## 3. Technical Implementation Guidelines

### Class Structure
The project relies on a flexible `MarkovDecisionProcess` class capable of handling variable parameters to support the three versions of complexity.

* **Input:** Problem parameters (State space size, Action space complexity, transition probabilities).
* **Methods:** Should support generation of sampling traces for the RL environment.

### Optimization Goals
* **Utility Function:** Must reflect the trade-off between discount cost and downstream conversion/retention value.
* **State Space:** Must be manageable for Phase 2 (DP) but scalable for Phase 3 (RL).

### Tools & Libraries
* **Course Code:** Useful for educational validation and simple parameterizations.
* **External Libraries:** Encouraged for Phase 3 to handle performance/scale.
