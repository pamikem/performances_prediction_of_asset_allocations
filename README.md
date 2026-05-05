# Performances prediction of asset allocations

This project focuses on predicting the short-term performance of financial asset allocations using historical behavior data. Each allocation represents a dynamic portfolio strategy described by past returns, trading activity, and liquidity signals. The goal is to determine whether an allocation will yield a positive or negative return in the next period.

---

## Requirements

Before setting up the project, make sure you have the following installed on your system:

- [pixi](https://pixi.sh) — environment and dependency manager
- [NVIDIA drivers](https://www.nvidia.com/drivers) — required for CUDA support (GPU only)
- `cmake`, `gcc`, `ninja` — will be managed automatically by pixi

---

## Environment setup

This project uses **[pixi](https://pixi.sh)** to manage the virtual environment, Python dependencies, and system-level packages (including CUDA for GPU support).

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Restart your terminal after installation.

### 2. Clone the repository

```bash
git clone https://github.com/pamikem/performances_prediction_of_asset_allocations.git
cd performances_prediction_of_asset_allocations
```

### 3. Install the environment

```bash
pixi install
```

This command reads `pixi.toml` and `pixi.lock` and installs all dependencies (Python packages, CUDA toolkit, system tools) into a local `.pixi/` directory. No manual activation needed.

### 4. Activate the environment shell

```bash
pixi shell
```

You are now inside the project virtual environment.

---

## Installing packages

All dependencies are declared in `pixi.toml`. To add new packages:

```bash
# Add a conda package (e.g. from conda-forge)
pixi add numpy

# Add a PyPI package (not available on conda)
pixi add --pypi some-package

# Add a package only for a specific platform (e.g. CUDA on Linux)
pixi add --platform linux-64 cuda-toolkit
```

After adding packages, `pixi.lock` is automatically updated to ensure reproducibility across machines.

To sync your environment after pulling changes from the repository:

```bash
pixi install
```

---

## Running tasks

Common tasks are defined in `pixi.toml` and can be run without activating the shell:

```bash
# Launch JupyterLab
pixi run notebook

# Lint the source code
pixi run lint

# Format the source code
pixi run format

# Train a model
pixi run train
```

---

## Working with Jupyter notebooks

### Launch JupyterLab

```bash
pixi run notebook
```

This opens JupyterLab pointing to the `notebooks/` directory.

### Select the correct kernel

When opening a notebook, select the kernel **"Python (your-project-name)"** from the kernel selector in the top-right corner. This ensures your notebook uses the pixi virtual environment and has access to all installed packages.

### Verify the kernel

Run this in a notebook cell to confirm the correct environment is active:

```python
import sys
print(sys.executable)
# Should point to: .pixi/envs/default/bin/python
```

### Notebook naming convention

```
<step>.<version>-<initials>-<short-description>.ipynb
```

Example: `1.0-jd-initial-data-exploration.ipynb`

---

## Project organization

```
├── LICENSE                         <- Open-source license (BSD-3-Clause)
├── Makefile                        <- Convenience commands (e.g. `make data`, `make train`)
├── README.md                       <- This file
├── pixi.toml                       <- Environment and dependency declaration (pixi)
├── pixi.lock                       <- Locked dependency versions for reproducibility
├── pyproject.toml                  <- Python package metadata and tool configuration (ruff)
│
├── data
│   ├── external                    <- Data from third-party sources
│   ├── interim                     <- Intermediate, transformed data
│   ├── processed                   <- Final datasets for modeling
│   └── raw                         <- Original, immutable data dump
│
├── docs                            <- MkDocs project documentation
│
├── models                          <- Trained models, predictions, summaries
│
├── notebooks                       <- Jupyter notebooks (see naming convention above)
│
├── references                      <- Data dictionaries, manuals, explanatory materials
│
├── reports
│   └── figures                     <- Generated graphics and figures
│
└── predict-perf-allocation         <- Source code (Python module)
    ├── __init__.py
    ├── config.py                   <- Paths, constants, project-wide settings
    ├── dataset.py                  <- Data loading and generation scripts
    ├── features.py                 <- Feature engineering
    ├── plots.py                    <- Visualization helpers
    └── modeling
        ├── __init__.py
        ├── train.py                <- Model training
        └── predict.py             <- Model inference
```

---

## Code quality

This project uses **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting (replaces flake8 + black + isort).

```bash
# Check for issues
pixi run lint

# Auto-format code
pixi run format
```

Ruff is configured in `pyproject.toml` under `[tool.ruff]`.