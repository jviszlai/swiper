# SWIPER
Speculative windowed decoding.

## Getting started
Easiest way: install dependencies via Poetry using the pyproject.toml file.

## Organization
- `swiper` contains all SWIPER-SIM code.
- `tests` contains unit tests for SWIPER-SIM, runnable via `pytest`.
- `benchmarks` contains code to generate benchmark lattice surgery schedules, as well as cached compiled schedules in a custom lattice surgery format (.lss).
- `slurm` contains scripts to run evaluations via SLURM, and contains data used to generate figures in this work.
- `notebooks` contains Jupyter notebooks used to analyze and plot data.
