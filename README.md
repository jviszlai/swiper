# SWIPER
Speculative Window Decoding for Fault-Tolerant Quantum Programs.

## ISCA 2025 artifact evaluation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15102955.svg)](https://doi.org/10.5281/zenodo.15102955)

See `artifact/` directory (and `artifact/README.md`) for information on
reproducing the results presented in the publication "SWIPER: Minimizing
Fault-Tolerant Quantum Program Latency via Speculative Window Decoding"

## Installing dependencies
Prerequisites:
- git
- Python 3.10 or 3.11
- CMake and gcc (needed to install Python dependencies)

Clone the repository, then install the required dependencies
using `pip` (with the `requirements.txt`) file or using `poetry` (with the
`pyproject.toml` file).

If you run into issues installing the `tweedledum` package, note that this
depends on some C++ libraries that may need to be manually reinstalled if the
version on your PC is too old.

## Organization
```
.
+-- artifact/               # ISCA 2025 publication data and plots
|   +-- data/               # pre-generated data from publication
|   +-- figures/            # figures from publication
|   +-- README.md
|   +-- run_*.py            # scripts to generate simulation data
+-- benchmarks/
|   +-- cached_schedules    # Compiled benchmark programs
|   ...
+-- notebooks/              # Interactive notebooks with examples and plots
+-- slurm/                  # Scripts and data from slurm jobs
+-- swiper/                 # SWIPER-SIM codebase
+-- tests/                  # Unit tests, run via pytest
...
+-- pyproject.toml          # Use to install dependencies via poetry
+-- requirements.txt        # Use to install dependencies via pip
+-- README.md               # This file
```

## Getting Started
See [getting_started.ipynb](doc/getting_started.ipynb) to try out the simulator!
