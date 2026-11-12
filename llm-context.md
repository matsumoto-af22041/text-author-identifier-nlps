# Research Technical Context

## Target Computational Stack

- **Language / Environment**: [e.g., Python 3.11 / uv or Poetry / CUDA 12.x]
- **Core Libraries**: [e.g., PyTorch 2.x, NumPy, SciPy, scikit-learn, Matplotlib]
- **Experiment Management / Config Tools**: [e.g., Hydra, Weights & Biases (wandb), MLflow]
- **Computational Resource Constraints**: [e.g., Must run within a single GPU (24GB VRAM) / 32GB system RAM]

## Directory Structure & Asset Placement Rules

- Define the separation of responsibilities across code, configurations, data, and experimental outputs.
- Example:
  - `/configs`: Hyperparameter and experimental condition configuration files (YAML, etc.)
  - `/src`: Reusable core logic (model definitions, data loaders, evaluation metrics)
  - `/scripts` (or `/experiments`): Entry-point scripts for running experiments
  - `/notebooks`: Exploratory Data Analysis (EDA) and visualization only (implementing core logic here is prohibited)
  - `/data`: `raw/` (read-only raw data) and `processed/` (preprocessed data)
  - `/outputs` (or `/results`): Experiment logs, model checkpoints, and publication figures

## Domain Glossary & Mathematical Notation

- Define domain-specific terminology and the exact mapping between mathematical symbols in the paper and variable names in the codebase.
- Example:
  - $N$ (`num_samples`): Total number of samples in the dataset
  - $d$ (`embed_dim`): Dimensionality of the feature embedding space
  - "Episode": A single trajectory from an initial state to a terminal state in reinforcement learning