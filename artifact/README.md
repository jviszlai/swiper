# SWIPER artifact (ISCA 2025)

This directory contains the necessary scripts and data to generate the figures
in **SWIPER: Minimizing Fault-Tolerant Quantum Program Latency via Speculative
Window Decoding**.

## Installing Python dependencies
This repository requires Python 3.10-3.12. You can install the required dependencies
using `pip` with the `requirements.txt` file, or using `poetry` with the
`pyproject.toml` file.

## Generating plots
Plots can be generated using either the provided data or re-generated data by
running `python artifact/run_analysis.py`. This will generate all data-related
figures that appear in the paper. This involves reading pre-generated data from
`artifact/data`, as well as performing some simpler simulations/calculations
within the script. This script is a combination of
plotting code from multiple notebooks in the `notebooks/` directory.

## Re-generating data
Some of the figures in the paper rely on data that is computationally expensive
to generate. For these datasets, we have provided our data (in `artifact/data/`)
along with scripts to re-generate this data if desired. The figures can all be
created using either the provided data or any re-generated data, if desired.

Note that the SWIPER-SIM
benchmark evaluations require significant computational resources (>3000
single-core jobs, some of which take multiple days to complete) and a
slurm-enabled compute cluster.

The following scripts will re-generate various datasets:

- `analysis/run_benchmark_evals.py`: runs ~3000 SWIPER-SIM slurm jobs to evaluate
  SWIPER on various application benchmarks. Relevant for Figs. 14 and 15.
- `analysis/run_predictor_accuracy.py`: simulates 1-, 2-, and 3-step predictor
  accuracy for different code distances. Relevant for Fig. 4.
- `analysis/run_pymatching_latencies.py`: evaluates latency of Pymatching on
  random decoding problems for different code distances and decoding volumes.
  Relevant for Fig. 3, and as an input to SWIPER-SIM.
- `analysis/run_reaction_time_evals.py`: runs ~300 SWIPER-SIM slurm jobs to
  evaluate SWIPER on a simple "random-T" schedule with different decoder
  latencies. Relevant for Fig. 12.