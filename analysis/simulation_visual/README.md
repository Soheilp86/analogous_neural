# Simulation visual

This directory contains code for running analogous cycles on simulated visual system (V1 simple cells, orientation cells, direction cells)

### Repository structure
* `simulation_code`
    * Contains notebooks that simulate the visual system.
    * All code are in Python. 
* The following notebooks are written in Julia.
* `1_analogous_bars_encoding_study.ipynb`: Computes analogous cycles between the stimulus and simulated V1 simple cells.
* `2_analogous_bars_propagation_study.ipynb`: Computes analogous cycles between V1 simple cells, orientation cells, and direction cells.
* `3_null_model_analogous_stats.ipynb`: Computes the null model statistics. Uses distributed computation.

<mark>Caution: </mark> Running the analogous cycles in this particular simulated data takes a very long time. We recommend running on a server and letting it run for a couple days. 