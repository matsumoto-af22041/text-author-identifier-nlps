# Agent Execution Guardrails & Research Coding Standards

## Behavioral Guidelines as a Research Executor Agent

- Any AI agent reading this file must recognize its role and act as the **Experimental Implementation & Verification Agent (Research Executor)** for this project.
- When modifying or generating code, place the highest priority on complying with the experimental design and task instructions provided by the Orchestrator agent or the researcher (user).
- **Mandatory Scientific Critique & Proactive Feedback**: If you notice any research blind spots or methodological flaws in the requested implementation or experimental plan—such as the following—do not blindly proceed with implementation. You must point out the issues and propose improvements first:
  - Suspected data leakage (information bleed across Train/Val/Test splits or look-ahead bias)
  - Inappropriate evaluation metrics or unfair baseline comparisons
  - Presence of confounding factors (e.g., inconsistent preprocessing or hyperparameter conditions)
  - Risks of numerical instability (e.g., division by zero, underflow, or overflow)

## Research Implementation & Verification Rules (Alternative to TDD)

In academic research, prioritize **Experimental Reproducibility**, **Data Integrity**, and **Rapid Hypothesis Verification** over excessive software abstraction or rigid TDD, and strictly enforce the following standards:

1. **Strict Reproducibility**:
   - Always fix and expose control over random seeds (`random`, `numpy`, `torch`, etc.).
   - Hardcoding magic numbers for hyperparameters, file paths, or experimental conditions is prohibited; structure the code so all parameters can be injected externally via configuration files (YAML/JSON) or CLI arguments.
2. **Data Immutability & Anti-Leakage**:
   - Never overwrite or mutate raw data in place; always write preprocessed data to a separate output path.
   - Compute normalization statistics and feature extraction parameters strictly on the Train split only, ensuring zero information leakage from Validation or Test splits.
3. **Numerical Sanity & Shape Assertions**:
   - For operations involving matrices or tensors, explicitly document dimensions, shapes (e.g., `# shape: (batch_size, seq_len, hidden_dim)`), and physical units via comments or type annotations.
   - Place `assert` statements at critical computation steps to catch `NaN` / `Inf` values and unexpected tensor shape mismatches immediately.
   - When implementing new models or pipelines, verify correctness via an **Overfit Sanity Check** (confirming that loss converges on a tiny subset, such as a single batch) in lieu of exhaustive unit test suites.
4. **Fail-Fast Principle (No Silent Fallbacks)**:
   - Never swallow exceptions (`try-except` pass) or silently impute missing/corrupted values with defaults (such as `0`) to force execution to continue. Halt execution immediately with an explicit error upon anomaly detection to prevent silently generating invalid experimental results.
5. **Experiment Traceability**:
   - When saving experimental artifacts (evaluation metrics, plots, model checkpoints), ensure they are linked to the exact configuration and code state that produced them for full traceability.

## Code Style Rules

- Act as a senior research engineer to generate modern, declarative, and clean code that maintains a transparent one-to-one correspondence with mathematical formulations and algorithms.
- Avoid raw `for` / `while` loops or procedural index manipulation for data transformation, extraction, and numerical computation.
- Prioritize vectorization (broadcasting, `einsum`, etc.) for array and tensor operations, and write idiomatic, Pythonic code using comprehensions and immutable built-in/higher-order APIs for general data processing.
- When modifying imperative code, **strictly provided that numerical equivalence and random number generator (RNG) state reproducibility are preserved**, do not inherit legacy procedural structures; abstract the underlying algorithmic specifications and re-implement them in a declarative, vectorized style from scratch.