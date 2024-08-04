# Experimental visual

This directory contains code for running analogous cycles on the experimental visual system (V1 and AL cells)

### Repository structure
* `preprocessing_code`
    * Contains notebook for preprocessing data. Code written in Python.
* `PD_significance`
    * Contains notebook for computing significance threshold from random spike trains.
* The following notebooks are written in Julia.
* `1_analogous_cycles.ipynb`: Computes analogous cycles between V1 and AL cells.
* `2_null_model_stats.ipynb`: Computes the null model statistics. Uses distributed computation.
