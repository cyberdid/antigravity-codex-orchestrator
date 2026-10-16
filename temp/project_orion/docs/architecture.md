# Project Architecture

## Module Boundaries
- `core/`: Contains the primary data generation logic. Modules here are strictly focused on the mathematical models of the data.
- `processing/`: Handles data cleaning, normalization, and transformation.
- `analytics/`: Contains evaluation metrics and statistical analysis tools.
- `utils/`: Common utilities (logging, configuration parsing, file handling).

## Reproducibility & Determinism
- **Seed Handling**: A global seed is established at the entry point. All libraries (e.g., `numpy`, `random`, `torch` if applicable) are seeded to ensure identical output across runs.
- **Deterministic Operations**: Where possible, non-deterministic algorithms are replaced with deterministic alternatives.

## Runtime Constraints
- **Memory**: The system is designed for bounded-memory execution. Large datasets are processed in chunks or via generators.
- **Hardware**: GPU acceleration is optional and requires manual environment validation.
- **Offline Operation**: The system is designed to run in air-gapped environments.

## Synthetic Assumptions
- The data generator assumes a "perfect" distribution for the initial synthetic population.
- Noise models are assumed to be Gaussian unless otherwise specified.
- Temporal correlations are modeled as first-order Markov processes.

## Environment Validation
- **Python Version**: 3.13.x
- **NumPy**: Required (Status: Pending local verification)
- **Pytest**: Available (v9.1.1)
- **CUDA/ROCm**: Not assumed; requires manual check for GPU-bound tasks.
