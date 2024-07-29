# Analogous cycles 

This repository contains code accompanying the following paper: Yoon et al 2024 "Tracking the topology of neural manifolds across populations".

* We recommend running the code on a compute server rather than a laptop.
* One should first install Julia. 
* Due to the large size of data, not all data files are included in this repository. Please contact Iris Yoon (hyoon@wesleyan.edu) for copies of the data. 

# Install
Download & install <a href="https://julialang.org/downloads/">Julia</a>. 


# Instructions on running Jupyter notebooks
<u>It is important that one activates the virutal environment associated with this project.</u>

## Using Julia REPL
* Open Julia REPL
* Using the `cd` command, navigate to the root of this directory.
* Activate \& instantiate the virtual environment as follows.
```
using Pkg
Pkg.activate( "/env/.")
Pkg.instantiate()
```
* Open a notebook using the following command.
```
using IJulia
notebook()
```
* You can now run the Jupyter notebooks in this repository.

## Using Jupyter notebook/lab or VS studio code 
* The 1st cell in every Jupyter notebook of this repository contains the above command that activates the virtual environment. Please make sure that you run this command. 


# Instructions on running code on your data 
When running the analogous cycles script on your own data, please make sure to include the following command. 
```
using Pkg
Pkg.activate( "/env/.")
Pkg.instantiate()

include("src/Eirene_var.jl")
include("src/analogous_bars.jl")

using .Eirene_var
using .analogous_bars
```