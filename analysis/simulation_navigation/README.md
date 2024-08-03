# Simulation navigation

This directory contains code for running analogous cycles on simulated navigation system (HD, grid, conjunctive cells)

### Repository structure
* `simulation_navigation`
    * Contains code for simulating the HD, grid, and conjunctive cells. 
    * References: "Toroidal topology of population activity in grid cells" (Gardner et al, 2022). The specific simulation model is based on "Recurrent inhibitory circuitry as a mechanism for grid formulation" (Couey et al, 2013).
    * In particular, see notebook `simulate_grid_HD_conj_cells.ipynb`. The code is in Python. 
* `1_analogous_bars_grid_conj.ipynb`: Computes analogous cycles between grid cells and conjunctive cells.
* `2_analogous_bars_HD_conj.ipynb`: Computes analogous cycles between head-direction cells and conjunctive cells.
* `3_analogous_stats_from_null.ipynb`: Computes the null model statistics. Uses distributed computation.