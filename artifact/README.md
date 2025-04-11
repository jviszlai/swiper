# SWIPER artifact (ISCA 2025)

This directory contains the necessary scripts and data to generate the figures
in **SWIPER: Minimizing Fault-Tolerant Quantum Program Latency via Speculative
Window Decoding**.

## Installing dependencies
Prerequisites:
- git
- Python 3.10 or 3.11
- CMake and gcc (needed to install Python dependencies)

Clone the repository, then install the required dependencies
using `pip` (with the `requirements.txt`) file or using `poetry` (with the
`pyproject.toml` file).

### Potential issues

- Some users have reported issues installing the `tweedledum` dependency with the
following error message: `Could NOT find nlohmann_json: Found unsuitable version
"X.X.X", but required is at least "3.9.0"`. This occurs if you already have
`nlohmann_json` installed on your system. If you encounter this, you need to
manually install 3.9.0+ and retry.

## Generating plots
*Note: run all scripts from the base directory of the repository, not from within
`artifacts/`.*

Plots can be generated using either the provided data or re-generated data by
running `python artifact/run_analysis.py`. This will generate all data-related
figures that appear in the paper. This involves reading pre-generated data from
`artifact/data`, as well as performing some simpler simulations/calculations
within the script. This script is a combination of
plotting code from multiple notebooks in the `notebooks/` directory.

## Re-generating data
*Note: run all scripts from the base directory of the repository, not from within
`artifacts/`.*

Note that the SWIPER-SIM
benchmark evaluations require significant computational resources (>3000
single-core jobs, some of which take multiple days to complete) and a
slurm-enabled compute cluster.

The data generated for this publication is provided in `artifact/data`. The
results can be reproduced using the following Python scripts (note that two
require a SLURM cluster). The final script, `artifact/run_analysis.py`,
generates the figures based on the (original or new) data in `artifact/data`.

- `analysis/run_pymatching_latencies.py`: evaluates latency of Pymatching on
  random decoding problems for different code distances and decoding volumes.
  Relevant for Fig. 3, and as an input to SWIPER-SIM. Expected runtime: about 1
  hour.
- `analysis/run_predictor_accuracy.py`: simulates 1-, 2-, and 3-step predictor
  accuracy for different code distances. Relevant for Fig. 4. Expected runtime:
  about 1 hour.
- `analysis/run_mispredict_strategy.py`: simulates different strategies for
  recovering from mispredictions. Relevant for Fig. 8. Expected runtime: about 5
  minutes.
- `analysis/run_reaction_time_evals.py`: runs ~300 SWIPER-SIM slurm jobs to
  evaluate SWIPER on a simple "random-T" schedule with different decoder
  latencies. Relevant for Fig. 12. Expected runtime: one minute to submit jobs,
  and several hours for SLURM jobs to complete (assuming sufficient
  parallelization).
- `analysis/run_benchmark_evals.py`: runs ~3000 SWIPER-SIM slurm jobs to evaluate
  SWIPER on various application benchmarks. Relevant for Figs. 14 and 15.
  Expected runtime: one minute to several hours to submit jobs (depending on
  configured delay between submitting sets of jobs; this can be configured), and
  up to several days for all SLURM jobs to complete.
- `artifact/run_analysis.py`: **Generates all data-related
figures that appear in the paper.** This involves reading (original or new) data from
`artifact/data`, as well as performing some simpler simulations/calculations
within the script. Expected runtime: 5 minutes.