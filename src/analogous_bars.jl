"""
Implementation of the component-wise extension method and analogous bars method in F2. 

@author Iris Yoon
irishryoon@gmail.com
"""
module analogous_bars

include("Eirene_var.jl")
using Combinatorics
using Distances
using .Eirene_var
using Interact
using IJulia
using JLD
using Measures
using Plots
using Plots.PlotMeasures
using Random
using Statistics
using StatsBase
#using LaTeXStrings
using Printf
export 
    compute_Witness_persistence,
    #run_extension_VR_to_VR_bar,
    #run_extension_VR_to_VR,
    run_extension_W_to_VR_bar,
    run_extension_W_to_VR,
    #run_extension_VR_to_W_bar,
    #run_extension_VR_to_W,
    run_lazy_extension_VR,
    run_baseline_similarity_analogous,
    run_similarity_analogous,
    find_Dowker_cycle_correspondence,
    find_CE_BE, 
    find_CE_BE_at_param,
    find_BE_fixed_int_dec_at_param,
    find_alt_BE,
    find_alt_BE_at_param,
    find_alternative_bar_extension,
    compute_distance,
    compute_distance_square_torus,
    sample_torus,
    plot_pY,
    return_extension_results_at_parameter,
    find_cycle_death_in_Witness,
    plot_barcode,
    plot_analogous_bars,
    plot_P_Q,
    plot_cycle_single,
    plot_cycle,
    plot_3D,
    plot_PD,
    plot_Dowker_complex,
    plot_cycle_square_torus,
    select_intervals,
    select_persistent_intervals_IQR,
    get_multiclass_cyclerep,
    get_Witness_cyclerep,
    get_cyclerep_Eirene,
    find_all_homology_classes,
    plot_cycle_single_square_torus,
    run_cycle_extension_VR_to_VR,
    run_cycle_extension_VR_to_VR_bar,
    run_cycle_extension_VR_to_W,
    run_cycle_extension_VR_to_W_bar,
    run_cycle_extension_W_to_VR,
    run_cycle_extension_W_to_VR_bar,
    select_significant_subcollection,
    find_lazy_Ybar_representations_at_parameter,
    find_single_connected_cycle,
    convert_classrep,
    save_classreps,
    vector_to_symmetric_matrix,
    organize_baseline_offset_extensions,
    analogous_cycles_significant
gr()

#############################################################################################
# MAIN FUNCTION 
#############################################################################################
# """
#     run_extension_VR_to_VR_bar(; <keyword arguments>)
# Runs the bar-to-bar extension method to compare two Vietoris-Rips filtrations. Given a selected bar `Z_bar` in `barcode(C_Z, dim = dim)`, finds its cycle extension and bar extension in `C_Y` and `barcode(C_Y, dim = dim)`.

# Using paper notation: given a selected bar in ``\\text{BC}_k(Z^{\\bullet})``, finds the cycle extension in ``Y^{\\bullet}`` and bar extension in ``\\text{BC}_k(Y^{\\bullet})``.

# ### Arguments
# - `C_Z::Dictionary`: Eirene output of filtration `Z`
# - `D_Z::Array`: Distance matrix used to compute `C_Z`
# - `C_Y::Dictionary`: Eirene output of filtration `Y`
# - `D_Y::Array`: Distance matrix used to compute `C_Y`
# - `Z_bar::Integer`: Index of selected bar from `barcode(C_Z, dim = dim)`
# - `dim::Integer=1`: Dimension. Defaults to 1

# ### Returns
# - `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_VR_to_VR`
# """
# function run_extension_VR_to_VR_bar(;
#     C_Z::Dict{String, Any} = Dict{String, Any}(),
#     D_Z::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
#     C_Y::Dict{String, Any} = Dict{String, Any}(),
#     D_Y::Array{Float64,2} = Array{Float64}(undef, 0, 0),
#     Z_bar::Int64 = 0,
#     dim::Int64 = 1)
    
#     # find terminal class rep "tau" and parameter "delta"
#     tau, psi = find_terminal_class_in_VR(C_Z, D_Z, bar = Z_bar, dim = dim)
    
#     # run extension method
#     extension = run_extension_VR_to_VR(C_Z = C_Z, Z_cycle = tau, psi = psi, C_Y = C_Y, D_Y = D_Y, dim = dim)

#     return extension
# end

# """
#     run_extension_VR_to_VR(; <keyword arguments>)
# Runs the cycle-to-bar persistent extension method to compare two Vietoris-Rips filtrations. Given a cycle `Z_cycle` and a parameter `psi`, finds its cycle and bar extensions in `C_Y` and `barcode(C_Y, dim = dim)`.

# Using paper notation: given a cycle and a parameter ``[\\tau] \\in H_k(Z^{\\psi})``, finds its representations in ``Y^{\\bullet}`` and ``\\text{BC}_k(Y^{\\bullet})``.

# ### Arguments
# - `C_Z::Dictionary`: Eirene output of filtration `Z`
# - `Z_cycle`: Selected cycle of interest in complex `Z`. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
# - `psi`: parameter
# - `C_Y::Dictionary`: Eirene output of filtration `Y`
# - `D_Y::Array`: Distance matrix used to compute `C_Y`
# - `dim::Integer=1`: Dimension. Defaults to 1

# ### Outputs
# Function outputs a dictionary with the following keys. 
# **For a more detailed explanation, see the document "code_details.pdf".**

# - `comparison`: Indicates the types of filtrations under comparison. Value will be "VR to VR".
# - `C_Z`: copy of input `C_Z`
# - `C_Y`: copy of input `C_Y`
# - `dim`: dimension of interest. copy of input `dim`
# - `selected_cycle`: copy of input `Z_cycle`
# - `C_auxiliary_filtration`: Eirene output of filtration ``Z^{\\psi} \\cap Y^{\\bullet}``
# - `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
# - `p_Y`: collection of parameter values for ``p_Y``
# - `epsilon_0`: minimum value in `p_Y`
# - `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
# - `nontrivial_pY_dict`: A dictionary of parameter index and values.
# - `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
# - `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
# - `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
# - `cycle_extensions`: A dictionary summarizing cycle extensions at various parameters.
# - `bar_extensions`: A dictionary summarizing bar extensions at various parameters. 
# """
# function run_extension_VR_to_VR(;
#     C_Z::Dict{String, Any} = Dict{String, Any}(),
#     Z_cycle::Array{Array{Int64,1},1} = [[0,0]],
#     psi::Float64 = -1.0, 
#     C_Y::Dict{String, Any} = Dict{String, Any}(),
#     D_Y::Array{Float64,2} = Array{Float64}(undef, 0, 0),
#     dim::Int64 = 1)

#     ##### check input #####
#     if C_Z == Dict()
#         throw(UndefKeywordError(:C_Z))
#     end
#     if C_Y == Dict()
#         throw(UndefKeywordError(:C_Y))
#     end
#     if D_Y == Array{Float64}(undef, 0, 0)
#         throw(UndefKeywordError(:D_Y))
#     end
#     if Z_cycle == 0
#         throw(UndefKeywordError(:Z_cycle))
#     end
#     if psi == -1.0
#         throw(UndefKeywordError(:psi))
#     end

#     # build auxiliary filtration 
#     C_aux, Zpsi_index2simplex, Zpsi_simplex2index = build_auxiliary_filtration(C_Z, psi, D_Y, format = "VR to VR", dim = dim)
    
#     # find epsilon_0, Fbar_representation_tau
#     epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau(Z_cycle, C_aux, Zpsi_simplex2index, dim = dim)

#     # find p_Y and BARS_short
#     #max_pY = maximum(barcode(C_Y, dim = dim)[:,2])
#     p_Y, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
#     # find cyclereps of Fbar_representation_tau and BARS_short using vertices
#     cycle_reps = find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex)

#     # find component-wise bar-representations
#     Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short = find_all_bar_representations(C_aux, p_Y, cycle_reps, C_Y, D_Y, dim = dim)
    
#     # create summary (by parameter)
#     nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions = create_summary_by_parameter(cycle_reps, Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short, epsilon_0, C_aux, C_Y)
    
#     ### create output dictionary
#     extension = Dict()
#     # input data
#     extension["comparison"] = "VR to VR"
#     extension["C_Z"] = C_Z
#     extension["C_Y"] = C_Y
#     extension["dim"] = dim 
#     extension["selected_cycle"] = Z_cycle 
#     # auxiliary filtration & cyclerep
#     extension["C_auxiliary_filtration"] = C_aux
#     extension["aux_filt_cyclerep"] = cycle_reps
#     # parameters pY
#     extension["p_Y"] = p_Y
#     extension["epsilon_0"] = epsilon_0
#     # component-wise Ybar representations
#     extension["Ybar_rep_tau"] = Ybar_rep_tau
#     extension["Ybar_rep_short_epsilon0"] = Ybar_rep_short_epsilon0
#     extension["Ybar_rep_short"] = Ybar_rep_short
#     # exploring results by parameter
#     extension["nontrivial_pY"] = nontrivial_pY
#     extension["nontrivial_pY_dict"] = nontrivial_pY_dict
#     extension["cycle_extensions"] = cycle_extensions
#     extension["bar_extensions"] = bar_extensions
    
#     return extension
# end


"""
    run_extension_W_to_VR_bar(<keyword arguments>)   
Runs the bar-to-bars extension method to compare a Witness barcode and a VR barcode. Given a bar `W_bar` of a Witness filtration, finds its cycle and bar extensions in the VR barcode `barcode(C_VR, dim = dim)`.

### Arguments
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `W_bar::Int64`: Selected bar in Witness barcode
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Returns
- `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_W_to_VR` 
"""
function run_extension_W_to_VR_bar(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_bar::Int64 = 0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "delta"
    tau, psi = find_terminal_class_in_W(W, bar = W_bar, dim = dim)
    
    # run extension method from W to VR
    extension = run_extension_W_to_VR(W = W, 
                                      W_cycle = tau, 
                                      psi = psi, 
                                      C_VR = C_VR, 
                                      D_VR = D_VR, 
                                      dim = dim)
    
    # add key to 'extension' dic
    extension["selected_bar"] = W_bar
    return extension
end

"""
    run_extension_W_to_VR(; <keyword arguments>)
Runs the cycle-to-bar extension method to compare a Witness barcode to a Vietoris-Rips barcode. Given a cycle `W_cycle` in the Witness filtration, finds its cycle and bar extensions in `C_VR` and `barcode(C_VR, dim = dim)`. Note: The point cloud of the VR filtration and the landmark of the Witness filtration must be the same.

### Arguments
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `W_cycle`: cycle of interest. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
- `psi`: parameter
- `W_class::Int64`: Interval of interest in Witness barcode
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Outputs
Function outputs a dictionary with the following keys. 
**For a more detailed explanation, see the document "code_details.pdf".**

- `comparison`: Indicates the types of filtrations under comparison. Value will be "W to VR".
- `C_W`: copy of input `C_W`
- `C_VR`: copy of input `C_VR`
- `dim`: dimension of interest. copy of input `dim`
- `selected_cycle`: copy of input `W_cycle`
- `C_auxiliary_filtration`: Eirene output of filtration ``W^{\\psi} \\cap VR^{\\bullet}``
- `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
- `p_Y`: collection of parameter values for ``p_Y``
- `epsilon_0`: minimum value in `p_Y`
- `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
- `nontrivial_pY_dict`: A dictionary of parameter index and values.
- `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
- `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
- `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
- `cycle_extensions`: A dictionary summarizing cycle extensions at various parameters.
- `bar_extensions`: A dictionary summarizing bar extensions at various parameters. 
"""
function run_extension_W_to_VR(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_cycle::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex = build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
    
    # find epsilon_0, Fbar_representation_tau
    
    epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau_WtoVR_large_limit(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)
    
    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    p_Y, BARS_short = find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim) 
#                                            max_pY = max_pY)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps = find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)
    
    # check the minimum parameter at which one can find the boundary matrix from Eirene -- resolved with Eirene_var
    #min_epsilon_with_boundary = analogous_bars.return_min_parameter_with_boundary_matrix(D_VR, p_Y)
    #pY_new = [i for i in p_Y if i >= min_epsilon_with_boundary]
    
     # find component-wise bar-representations
    Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short = find_all_bar_representations(C_aux, 
                                                                                         p_Y, 
                                                                                         cycle_reps, 
                                                                                         C_VR, 
                                                                                         D_VR, 
                                                                                         dim = dim)
    
    # create summary (by parameter)
    nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions = create_summary_by_parameter(cycle_reps, 
                                                                                        Ybar_rep_tau, 
                                                                                        Ybar_rep_short_epsilon0, 
                                                                                        Ybar_rep_short, 
                                                                                        epsilon_0, 
                                                                                        C_aux,
                                                                                        C_VR)
       
    ### create output dictionary
    extension = Dict()
    # input data
    extension["comparison"] = "W to VR"
    extension["C_W"] = C_W
    extension["C_VR"] = C_VR
    extension["dim"] = dim 
    extension["selected_cycle"] = W_cycle 
    # auxiliary filtration & cyclerep
    extension["C_auxiliary_filtration"] = C_aux
    extension["aux_filt_cyclerep"] = cycle_reps
    # parameters pY
    extension["p_Y"] = p_Y
    extension["epsilon_0"] = epsilon_0
    #extension["min_epsilon_with_boundary"] = min_epsilon_with_boundary
    # component-wise Ybar representations
    extension["baseline_bar_extension"] = Ybar_rep_tau
    extension["offset_bar_extension_epsilon0"] = Ybar_rep_short_epsilon0
    extension["offset_bar_extension"] = Ybar_rep_short
    # exploring results by parameter
    extension["nontrivial_pY"] = nontrivial_pY
    extension["nontrivial_pY_dict"] = nontrivial_pY_dict
    extension["cycle_extensions"] = cycle_extensions
    extension["bar_extensions"] = bar_extensions

    return extension
end


# """
#     run_extension_VR_to_W_bar(<keyword arguments>)
# Runs the bar-to-bars extension method to compare a Vietoris-Rips barcode and a Witness barcode. Given a bar `VR_bar` in the VR barcode, finds the cycle and bar extensions in `W` and `barcode(W, dim = dim)`. 

# ### Arguments
# - `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
# - `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
# - `VR_bar::Int64`: Selected bar in the VR barcode
# - `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
# - `D_W::Array{Float64, 2}`: Distance matrix used to compute `C_W`
# - `dim::Int64 = 1`: Dimension. Defaults to 1

# ### Returns
# - `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_VR_to_W` 
# """
# function run_extension_VR_to_W_bar(;
#     C_VR::Dict{String, Any} = Dict{String, Any}(),
#     D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
#     VR_bar::Int64 = 0,
#     W::Dict{Any, Any} = Dict{Any, Any}(),
#     D_W::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
#     dim::Int64 = 1)
    
#     # find terminal class rep "tau" and parameter "psi"
#     tau, psi = find_terminal_class_in_VR(C_VR, D_VR, bar = VR_bar, dim = dim)
    
#     # run extension method from W to VR
#     extension = run_extension_VR_to_W(C_VR = C_VR, D_VR = D_VR, VR_cycle = tau, psi = psi, W = W, D_W = D_W, dim = dim)
#     return extension
# end

# """
#     run_extension_VR_to_W(; <keyword arguments>)
# Runs the cycle-to-bars extension method to compare a Vietoris-Rips barcode and a Witness barcode. Given a cycle `VR_cycle` in the VR barcode, finds the cycle and bar extensions in `W` and `barcode(W, dim = dim)`. 
# Note: The 0-simplices of the Vietoris-Rips filtration and the Witness filtration must be the same.

# ### Arguments
# - `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
# - `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
# - `VR_cycle`: cycle of interest in the VR filtration. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
# - `psi`: parameter
# - `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
# - `D_W::Array{Float64, 2}`: Distance matrix used to compute the Witness filtration
# - `dim::Int64 = 1`: Dimension. Defaults to 1    

# ### Outputs
# Function outputs a dictionary with the following keys. 
# **For a more detailed explanation, see the document "code_details.pdf".**

# - `comparison`: Indicates the types of filtrations under comparison. Value will be "W to VR".
# - `C_VR`: copy of input `C_VR`
# - `C_W`: copy of input `C_W`
# - `dim`: dimension of interest. copy of input `dim`
# - `selected_cycle`: copy of input `VR_cycle`
# - `C_auxiliary_filtration`: Eirene output of filtration ``VR^{\\psi} \\cap W^{\\bullet}``
# - `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
# - `p_Y`: collection of parameter values for ``p_Y``
# - `epsilon_0`: minimum value in `p_Y`
# - `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
# - `nontrivial_pY_dict`: A dictionary of parameter index and values.
# - `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
# - `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
# - `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
# - `cycle_extensions`: A dictionary summarizing cycle extensions at various parameters.
# - `bar_extensions`: A dictionary summarizing bar extensions at various parameters. 
# """
# function run_extension_VR_to_W(;
#     C_VR::Dict{String, Any} = Dict{String, Any}(),
#     D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
#     VR_cycle::Array{Array{Int64,1},1} = [[0,0]],
#     psi::Float64 = -1.0, 
#     W::Dict{Any, Any} = Dict{Any, Any}(),
#     D_W::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
#     dim::Int64 = 1)
      
#     ##### check input #####
#     if W == Dict()
#         throw(UndefKeywordError(:W))
#     end
#     if C_VR == Dict()
#         throw(UndefKeywordError(:C_VR))
#     end
#     if D_VR == Array{Float64}(undef, 0, 0)
#         throw(UndefKeywordError(:D_VR))
#     end
#     if D_W == Array{Float64}(undef, 0, 0)
#         throw(UndefKeywordError(:D_W))
#     end
#     if psi == -1.0
#         throw(UndefKeywordError(:psi))
#     end
    
    
#     # unpack variables
#     C_W = W["eirene_output"]
    
#     # build the auxiliary filtration
#     C_aux, Zpsi_index2simplex, Zpsi_simplex2index = build_auxiliary_filtration(C_VR, psi, D_W, format = "VR to W", dim = dim)

#     # find epsilon_0, Fbar_representation_tau
#     epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau(VR_cycle, C_aux, Zpsi_simplex2index, dim = dim)

#     # find p_Y and BARS_short
#     #max_pY = maximum(barcode(C_W, dim = dim)[:,2])
#     p_Y, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
#     # find cyclereps of Fbar_representation_tau and BARS_short using vertices
#     cycle_reps = find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex)

#     # find component-wise bar-representations
#     Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short = find_all_bar_representations_W(C_aux, p_Y, cycle_reps, Zpsi_index2simplex, W, dim = dim)
    
#     # create summary (by parameter)
#     nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions = create_summary_by_parameter(cycle_reps, Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short, epsilon_0, C_aux, C_W)
    
 
#     ### create output dictionary
#     extension = Dict()
#     # input data
#     extension["comparison"] = "VR to W"
#     extension["C_VR"] = C_VR
#     extension["C_W"] = C_W
#     extension["dim"] = dim 
#     extension["selected_cycle"] = VR_cycle 
#     # auxiliary filtration & cyclerep
#     extension["C_auxiliary_filtration"] = C_aux
#     extension["aux_filt_cyclerep"] = cycle_reps
#     # parameters pY
#     extension["p_Y"] = p_Y
#     extension["epsilon_0"] = epsilon_0
#     # component-wise Ybar representations
#     extension["Ybar_rep_tau"] = Ybar_rep_tau
#     extension["Ybar_rep_short_epsilon0"] = Ybar_rep_short_epsilon0
#     extension["Ybar_rep_short"] = Ybar_rep_short
#     # exploring results by parameter
#     extension["nontrivial_pY"] = nontrivial_pY
#     extension["nontrivial_pY_dict"] = nontrivial_pY_dict
#     extension["cycle_extensions"] = cycle_extensions
#     extension["bar_extensions"] = bar_extensions
    
#     return extension
# end

"""
    run_similarity_analogous(; <keyword arguments>)
Run the similarity-centric analogous bars method to compare Vietoris-Rips filtrations on two point clouds `P` and `Q`.
In particular, given a bar `W_PQ_bar` in barcode(W(P,Q)), the function finds all bar extensions of `W_PQ_bar` in barcode(VR(P)) and barcode(VR(Q)). Any bar extensions in barcode(VR(P)) and barcode(VR(Q)) can be considered as analogous bars. 

### Arguments
- `VR_P::Dict`: Dictionary output of Eirene on Vietoris-Rips filtration on P
- `D_P::Dict`: Distance matrix used to compute `VR_P`
- `VR_Q::Dict`: Dictionary output of Eirene on Vietoris-Rips filtration on Q
- `D_Q::Dict`: Distance matrix used to compute `VR_Q`
- `W_PQ::Dict`: Dictionary output of function `compute_Witness_persistence`. Represents the Witness filtration with P as landmark and Q as witness.
- `W_PQ_bar::Int64`: Selected bar of interest in the Witness barcode of `W_PQ`.
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Outputs
- `extension_P::Dict`: Dictionary output of bar-to-bars extension to `VR_P`.
- `extension_Q::Dict`: Dictionary output of bar-to-bars extension to `VR_Q`. 
"""
function run_similarity_analogous(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bar::Int64 = 0,
    dim::Int64 = 1)
    
    ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bar == 0
        throw(UndefKeywordError(:W_PQ_class))
    end
    
    ############################################################
    # 1. apply the extension method between W(P,Q) and VR(P)
    ############################################################
    extension_to_VR_P = run_extension_W_to_VR_bar(W = W_PQ, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)
    println("Extension to P complete.")

    ############################################################
    # 2. Compute the dual Witness complex
    ############################################################
    W_PQ_barcode = barcode(W_PQ["eirene_output"], dim = dim)
    W_death_param = find_max_death_param(W_PQ_barcode, [W_PQ_bar])

    W_QP, P_to_Q = compute_Dowker_dual(W_PQ, param_max = W_death_param)

    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    W_QP_bar = P_to_Q[W_PQ_bar]

    println("computed dual Witness filtration.")
    
    ############################################################
    # 3. apply the extension method between W(Q,P) and VR(Q)
    ############################################################

    ##### apply the extension method between W(P,Q) and VR(Q) #####
    extension_to_VR_Q = run_extension_W_to_VR_bar(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
    println("Extension to Q complete.")
    # ##### apply the extension method between W(P,Q) and VR(P) #####
    # extension_to_VR_P = run_extension_W_to_VR_bar(W = W_PQ, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)

    # ##### apply the extension method between W(Q,P) and VR(Q) #####
    # # get W(Q,P) info
    # D_P_Q = W_PQ["distance_matrix"]
    # D_Q_P = collect(transpose(D_P_Q))
    # W_QP = compute_Witness_persistence(D_Q_P, maxdim = 1)
    
    # # find the bar in W(Q,P) that corresponds to W_PQ_bar
    # P_to_Q = apply_Dowker(W_PQ, W_QP, dim = dim)
    # W_QP_bar = P_to_Q[W_PQ_bar]
        
    # # extension in VR(Q)
    # extension_to_VR_Q = run_extension_W_to_VR_bar(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
    
    return extension_to_VR_P, extension_to_VR_Q
end


#############################################################################################
# FUNCTIONS FOR RUNNING SIMILARITY-CENTRIC ANALOGOUS BARS AT BIRTHTIME OF WITNESS CYCLE
#############################################################################################


function find_classrep_in_W(
    W::Dict; # dictionary output of "compute_Witness_persistence" 
    bar::Int64 = 0,
    dim = 1)
    
    C_W = W["eirene_output"]
    D = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_vertex_to_default_vertex = W["W_vertex_to_default_vertex"]
    
    # find class rep of bar
    rep = classrep(C_W, class = bar, dim = dim)
    
    # express as a collection of simplices
    tau = [W_index2simplex[(item, dim)] for item in rep]
        
    if W_vertex_to_default_vertex != nothing
        tau = [[W_vertex_to_default_vertex[item] for item in simplex] for simplex in tau]
    end
    
    return tau
end

function run_extension_W_to_VR_bar_birthtime(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_bar::Int64 = 0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find class rep "tau" of Witness filtration
    tau = find_classrep_in_W(W, bar = W_bar, dim = dim)

    # find the birth time of the selected bar in the Witness barcode
    W_barcode = barcode(W["eirene_output"], dim = dim)
    birth_param = W_barcode[W_bar,1]
    
    # run extension method from W to VR
    extension = run_extension_W_to_VR(W = W, 
                                      W_cycle = tau, 
                                      psi = birth_param, 
                                      C_VR = C_VR, 
                                      D_VR = D_VR, 
                                      dim = dim)
    
    # add key to 'extension' dic
    extension["selected_bar"] = W_bar
    return extension
end

"""
Modified version of `run_similarity_analogous`. 
Uses the birthtime of the Witness cycle. 

Original name: run_similarity_analogous_birthtime
"""
function run_birthtime_similarity_analogous(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bar::Int64 = 0,
    dim::Int64 = 1)

     ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bar == 0
        throw(UndefKeywordError(:W_PQ_class))
    end

    ##### apply the extension method between W(P,Q) and VR(P) #####
    extension_to_VR_P = run_extension_W_to_VR_bar_birthtime(W = W_PQ, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)

    ##### apply the extension method between W(Q,P) and VR(Q) #####
    # get W(Q,P) info
    D_P_Q = W_PQ["distance_matrix"]
    D_Q_P = collect(transpose(D_P_Q))
    W_QP = compute_Witness_persistence(D_Q_P, maxdim = 1)
    
    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = analogous_bars.apply_Dowker(W_PQ, W_QP, dim = dim)
    W_QP_bar = P_to_Q[W_PQ_bar]
        
    # extension in VR(Q)
    extension_to_VR_Q = run_extension_W_to_VR_bar_birthtime(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
    
    return extension_to_VR_P, extension_to_VR_Q
end


```
Variation of the function `run_similarity_analogous_birthtime`.
Build the Witness complex only upto the birth time of the bar we are interested in. 
```
function run_similarity_analogous_birthtime_fast(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bar::Int64 = 0,
    dim::Int64 = 1)

     ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bar == 0
        throw(UndefKeywordError(:W_PQ_class))
    end

    ##### apply the extension method between W(P,Q) and VR(P) #####
    extension_to_VR_P = analogous_bars.run_extension_W_to_VR_bar_birthtime(W = W_PQ, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)
    
    ##### apply the extension method between W(Q,P) and VR(Q) #####
    # get W(Q,P) info
    D_P_Q = W_PQ["distance_matrix"]
    D_Q_P = collect(transpose(D_P_Q))

    # get max parameter for witness filtraiton
    barcode_W_PQ = barcode(W_PQ["eirene_output"], dim = 1)
    birth_param = barcode_W_PQ[W_PQ_bar,1]

    W_QP = compute_Witness_persistence(D_Q_P, maxdim = 1, param_max = birth_param)
    
    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = apply_Dowker_birth(W_PQ, W_QP, dim = dim)
    W_QP_bar = P_to_Q[W_PQ_bar]

    # extension in VR(Q)
    extension_to_VR_Q = analogous_bars.run_extension_W_to_VR_bar_birthtime(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
    
    return extension_to_VR_P, extension_to_VR_Q
end


#############################################################################################
# FUNCTIONS FOR EXPLORING EXTENSION RESULTS
#############################################################################################

"""
    return_extension_results_at_parameter
An interactive function that summarizes the bar extensions at a specific parameter of ``p_Y``.
All bar extensions are computed under the default interval decomposition of ``barcode(C_Y)`` used by Eirene. 

1. User selects parameter of interest from ``p_Y``
2. Function prints the baseline bar extension and offset bar extensions at selected parameter.
3. User selects offset bar extensions.
4. Function prints the resulting bar extension (baseline + offset bar extensions).
5. Function returns a plot object for the final bar extension. 

### Arguments
- `extension::Dictionary` : Output of `run_extension_VR_to_VR_bar()`, `run_extension_VR_to_VR()`, `run_extension_W_to_VR_bar()`, `run_extension_W_to_VR()`, `run_extension_VR_to_W_bar()`, or `run_extension_VR_to_W()`

### Returns
- `p::plots` :plot of the selected baseline bar extension + offset bar extension highlighted 
"""
function return_extension_results_at_parameter(extension; size = (500, 400))
    # get variables
    pY_dict = extension["nontrivial_pY_dict"]
    dim = extension["dim"]
    if extension["comparison"] == "VR to VR"
        C_target = extension["C_Y"]
        title = "barcode(VR(Y))"
    elseif extension["comparison"] == "VR to W"
        C_target = extension["C_W"]
        title = "barcode(W)"
    elseif extension["comparison"] == "W to VR"
        C_target = extension["C_VR"]
        title = "barcode(VR)"
    end
    
    ### 1. User selects parameter 
    # print parameter key and parameter values in p_Y
    print("*** Parameter key, value pair *** \n")
    for i = 1:length(pY_dict)
        @printf("key: %i parameter: %.6f \n", i,pY_dict[i] )
    end
   
    # ask user to select key for parameter
    key = IJulia.readprompt("\nSelect a key for parameter")
    key = parse(Int64, key)
    parameter = pY_dict[key]
    
    # print selected parameter
    print("Selected parameter: ", parameter, "\n")
    
    ### 2. Print the baseline and offset bar extensions at selected parameter
    # print the baseline bar-extension
    baseline_ext = extension["bar_extensions"][parameter]["baseline"]
    print("\nBaseline bars extension at selected parameter: ", baseline_ext, "\n")
    
    # find the offset bar-extensions
    offset_ext = unique(values(extension["bar_extensions"][parameter]["offset"]))
    print("\n*** Offset bar extensions at selected parameter *** \n")
    for i =1:length(offset_ext)
       @printf("key: %i offset bar extension: ", i)
        print(offset_ext[i], "\n") 
    end
    
    ### 3. User selects offset bar extensions
    offset_key = IJulia.readprompt("\nSelect keys for offset bar extensions. \nLeave blank to select none. \nTo select multiple keys, separate keys with space. ex) 1 2 3 : ")
    
    ### 4. Find the resulting bar extension and print.
    if offset_key == ""
        bar_ext = baseline_ext
    else
        offset_key = split(offset_key, " ")
        offset_key = [parse(Int64, item) for item in offset_key]
        offset_selected = []
        for item in offset_key
            append!(offset_selected, offset_ext[item] )
        end
        bar_ext = select_odd_count(vcat(baseline_ext, offset_selected))       
    end
    
    # print the baseline, offset, and final bar extensions
    print("\nBaseline bars extension at selected parameter: ", baseline_ext, "\n")
    if offset_key != ""
        print("Selected offset bar extension: ", [i for i in offset_selected], "\n")
        print("Final bar extensions: ", bar_ext)
    end
    
    ### 5. plot final bar extension
    barcode_target = barcode(C_target, dim = extension["dim"])
    p = plot_barcode(barcode_target, 
                        title = title,
                        epsilon = parameter,
                        v_line = [parameter],
                        selected_bars = bar_ext)

    return p

end



"""
    find_CE_BE(extension)
Finds the collection of cycle extensions ``E(\\tau, Y^{\\bullet})``
and the bar extensions ``S^{\\mathcal{D}} = \\{S^{\\mathcal{D}}_{[y]} | \\ell \\in p_Y, [y] \\in \\mathfrak{E}_{\\ell} \\}`` 
from Algorithm 3 of paper. Note that the bar extension is computed for a **fixed interval decomposition** of ``PH_k(Y^{\\bullet})``.

### Arguments
- extension: (dict) output of the extension method

### Returns
- CE: (dict) of cycle extensions. Corresponds to ``E(\\tau, Y^{\\bullet})`` in Algorithm 3 of paper. 
    - Given a parameter `param`, CE[param] is a dictionary whose values are distinct (non-homologous) cycle extensions.
- BE: (dict) of bar extensions. Corresponds to ``S^{\\mathcal{D}} = \\{S^{\\mathcal{D}}_{[y]} | \\ell \\in p_Y, [y] \\in \\mathfrak{E}_{\\ell} \\}`` of paper.
    - Given a parameter `param`, BE[param] is a dictionary whose values are distinct bar-extensions.
"""
function find_CE_BE(extension)

    p_Y = extension["nontrivial_pY"]
    
    CE = Dict()
    BE = Dict()
    
    for param in p_Y
        CE_param, BE_param = find_CE_BE_at_param(extension, param)

        CE[param] = CE_param
        BE[param] = BE_param
    end
    return CE, BE
end

"""
    find_CE_BE_at_param(extension, param)
Find the cycle extensions and bar extensions at given parameter. Note: the bar extension is computed under a **fixed interval decomposition** of ``PH_k(Y^{\\bullet})``.
"""
function find_CE_BE_at_param(extension, param)
    
    BE_param = Dict()
    CE_param = Dict()

    bar_extensions = extension["bar_extensions"]
    cycle_extensions =extension["cycle_extensions"]
    
    ### (1) From component-wise offset bar extensions, select only the short bars (of the auxiliary filtration) 
    ### with distinct bar-representations in barcode(Y)        
    # note: any short bars with the same bar representation will result in homologous cycle extensions
    selected_short_bars = []
    selected_bar_reps = []

    for (short_bar, bar_rep) in bar_extensions[param]["offset"]
        if bar_rep ∉ selected_bar_reps
            push!(selected_bar_reps, bar_rep)
            push!(selected_short_bars, short_bar)
        end
    end

    # consider all combinations of the selected short bars
    short_bar_comb = collect(combinations(selected_short_bars))

    ### (2) Save the baseline bar extension and cycle extensions as "0"
    BE_param[0] = copy(bar_extensions[param]["baseline"]) #baseline bar extension
    CE_param[0] = copy(cycle_extensions[param]["baseline"]) # baseline cycle extension

    ### (3) Find all possible bar extensions = "baseline bar extension" + linear combination ("offset bar extension")
    # ONLY save distinct bar extensions & their corresponding cycle extensions
    offset_bar_ext = bar_extensions[param]["offset"]
    j = 1
    for bars in short_bar_comb
        ### Get bar extension ###
        # get baseline bar extension
        bar_ext = copy(bar_extensions[param]["baseline"])

        # get the offset bar extension corresponding to the selected "bars"
        for k in bars
            append!(bar_ext, offset_bar_ext[k])
        end

        # combine baseline and offset bars
        bar_ext = select_odd_count(bar_ext)
        sort!(bar_ext)

        ### save if bar_ext is not already in BE_param
        if bar_ext ∉ values(BE_param)
            # save corresponding cycle extension
            baseline_ce = copy(cycle_extensions[param]["baseline"])
            offset_ce = vcat([extension["aux_filt_cyclerep"][i] for i in bars]...)
            cycle_ext = vcat(baseline_ce, offset_ce)
            cycle_ext = select_odd_count([sort(item) for item in cycle_ext])
            CE_param[j] = cycle_ext

            # save bar extension
            BE_param[j] = bar_ext
            j += 1
        end
    end
    return CE_param, BE_param
end

"""find_nontrivial_pY()
Finds the parameter values `epsilon` in pY in which a cycle (of the auxiliary filtration) born at `epsilon` has nontrivial cycle extension. 
"""
function find_nontrivial_pY(C_aux, Ybar_rep_short, epsilon_0; dim = 1)
    barcode_aux = barcode(C_aux, dim = dim)
    
    # collect birth times of bars in Ybar_rep_short
    nontrivial_pY = [barcode_aux[item,1] for item in keys(Ybar_rep_short)]
    
    # include epsilon_0
    append!(nontrivial_pY, epsilon_0)
    sort!(nontrivial_pY)
    
    # create dictionary of parameter index and parameter vaue
    nontrivial_pY_dict = Dict(i => nontrivial_pY[i] for i =1:size(nontrivial_pY,1))
    
    return nontrivial_pY, nontrivial_pY_dict
end

"""
    find_bar_extensions_at_parameter
Create a dictionary with key: parameter and values: baseline, offset bar-extensions. Note that all bar extensions are computed using the default interval decomposition of the target barcode used by Eirene.

### Arguments
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `Ybar_rep_tau`: (Array) representing the Y-bar representation of `Fbar_representation_tau`
- `Ybar_rep_short_epsilon0`: (Dictionary) representing the short bars of `C_aux` (that are present at epsilon0) and their Y-bar representations at epsilon0
- `Ybar_rep_short`: (Dictionary) representing the remaining short bars of `C_aux` and their Y-bar representations at their birth times. 
- `nontrivial_pY`: output of `find_nontrivial_pY()`
- `C_Y`: (dict) output of Eirene.
- `dim`: (int) dimension. Defaults to 1

### Outputs
- `bar_extensions`: (dict)
    - keys: parameters
    - values: (dict)
        - keys: "baseline", "offset"
        - values: if key == "baseline", then array representing the baseline bar-extension
                  if key == "offset", then another dictionary of the following form
                        - key: BARS_{short}
                        - values: baseline bar-extension corresponding to the short bar in key

### Example output and its interpretation
be["0.123"]["baseline"] = [1, 2, 3]
be["0.123"]["offset"][5] = [4, 5]
- at parameter 0.123, the baseline bar extension is [1,2,3]
- at parameter 0.123, the offset bar extension corresponding to bar 5 of BAR_{short} is [4, 5]
"""
function find_bar_extensions_at_parameter(
        C_aux,
        Ybar_rep_tau, 
        Ybar_rep_short_epsilon0, 
        Ybar_rep_short,
        nontrivial_pY,
        C_Y;
        dim = 1)
    barcode_aux = barcode(C_aux, dim = dim)
    bar_extensions = Dict()
    
    ### at epsilon_0
    # baseline bar: Y-bar extension Fbar_representation_tau
    # offset bars: Y-bar extensions of short bars. 
    # Note: offset bars is a dictionary. Keys: bar numbers in auxiliary filtration. values: bar extensions in barcode(Y) 
    be0 = Dict()
    be0["baseline"] = Ybar_rep_tau
    be0["offset"] = Ybar_rep_short_epsilon0
    bar_extensions[nontrivial_pY[1]] = be0
    
    ### at epsilon > epsilon_0
    # baseline bar: Fbar_representation_tau + cycle born at epsilon
    # offset bar: all other short cycles present at epsilon
    barcode_Y = barcode(C_Y, dim = dim)
    for i = 2:size(nontrivial_pY,1)
        be_epsilon = Dict()
        epsilon = nontrivial_pY[i]

        ### baseline bar extension: The non-trivial bars of baseline bar extension at epsilon0
        if Ybar_rep_tau == nothing
            be_epsilon["baseline"] = nothing
        else
            be_epsilon["baseline"] = [i for i in Ybar_rep_tau if epsilon < barcode_Y[i,2]]
        end
        
        ### offset bar: find bars of C_aux that (1) exist at given epsilon and (2) whose bar representation in Y^{\epsilon} is nontrivial
        # (1) bars that exist at epsilon 
        bars1 =  filter(kv -> barcode_aux[kv.first,1] <= epsilon < barcode_aux[kv.first,2], Ybar_rep_short_epsilon0)
        bars2 =  filter(kv -> barcode_aux[kv.first,1] <= epsilon < barcode_aux[kv.first,2], Ybar_rep_short)
        bars = merge(bars1, bars2)
        
        # (2) delete any bars of C_aux that are trivial in Y^{\epsilon}
        # note: A Y-bar representation of rho may be nontrivial, but it may be a trivial bar at the given parameter
        for key in keys(bars)
            new_val = [i for i in bars[key] if barcode_Y[i,1] <= epsilon < barcode_Y[i,2]]
            if new_val != []
                bars[key] = new_val
            else
                delete!(bars, key)
            end
        end
    
        be_epsilon["offset"] = bars
        bar_extensions[epsilon] = be_epsilon
    end
    return bar_extensions
    
end

"""find_cycle_extensions_at_parameter
Create a dictionary with key: parameter and values: baseline, offset cycle-extensions
"""
function find_cycle_extensions_at_parameter(
    C_aux,
    cycle_reps,
    Ybar_rep_short_epsilon0, 
    bar_extensions,
    nontrivial_pY,
    C_Y;
    dim = 1)
    
    cycle_extensions = Dict()
    barcode_aux = barcode(C_aux, dim = dim)

    ### at epsilon_0
    # baseline cycle: cycle extension of Fbar_representation_tau
    # offset cycles: cycle extensions of short bars at epsilon0
    ce0 = Dict()
    ce0["baseline"] = cycle_reps["Fbar_rep_tau"]
    ce0_offset = Dict()
    for key in keys(Ybar_rep_short_epsilon0)
        ce0_offset[key] = cycle_reps[key]
    end
    ce0["offset"] = ce0_offset
    cycle_extensions[nontrivial_pY[1]] = ce0

    ### at epsilon > epsilon_0
    # baseline cycle: cycle extensions of Fbar_representation_tau
    # offset cycles: cycle extensions of short bars (of C_aux) present at epsilon
    for i = 2:size(nontrivial_pY,1)
        ce_epsilon = Dict()
        epsilon = nontrivial_pY[i]
        
        # baseline cycle: find short bars (of C_aux) born at epsilon
        #b = find_birth_intervals_at_param(C_aux, epsilon, dim = dim)[1]
        #b = [item for item in b if barcode_aux[item, 2] < Inf]
        #baseline = select_odd_count(vcat(cycle_reps[b], cycle_reps["Fbar_rep_tau"]))
        
        # if bar extension is trivial, then cycle extension is trivial as well.
        if bar_extensions[epsilon]["baseline"] == []
            ce_epsilon["baseline"] = []
        else
            ce_epsilon["baseline"] = cycle_reps["Fbar_rep_tau"]
        end

        # offset cycle: cycle extensions of short bars that (1) exist at parameter epsilon and (2) is nontrivial in Y^{epsilon}
        bars = keys(bar_extensions[epsilon]["offset"])
        ce_epsilon["offset"] = Dict(key => cycle_reps[key] for key in bars)
        
        cycle_extensions[epsilon] = ce_epsilon
    end
    return cycle_extensions
end    


"""create_summary_by_parameter()
Given all Y-bar representations, organizes the result so that the baseline and offset cycle / bar extensions can be explored by parameter.
"""
function create_summary_by_parameter(
    cycle_reps,
    Ybar_rep_tau, 
    Ybar_rep_short_epsilon0, 
    Ybar_rep_short, 
    epsilon_0,
    C_aux,
    C_Y;
    dim = 1)
    
    # 1. find nontrivial_pY
    nontrivial_pY, nontrivial_pY_dict = find_nontrivial_pY(C_aux, Ybar_rep_short, epsilon_0, dim = dim)
    
    # 2. Create dictionary with parameter as key and baseline and offset bar-extensions as values
    bar_extensions = find_bar_extensions_at_parameter(C_aux,
                                                      Ybar_rep_tau, 
                                                      Ybar_rep_short_epsilon0, 
                                                      Ybar_rep_short,
                                                      nontrivial_pY,
                                                      C_Y;
                                                      dim = 1)
    
    # 3. Create dictionary with parameter as key and baseline cycles and offset cycles as values
    cycle_extensions = find_cycle_extensions_at_parameter(C_aux,
                                                          cycle_reps,
                                                          Ybar_rep_short_epsilon0, 
                                                          bar_extensions,
                                                          nontrivial_pY,
                                                          C_Y;
                                                          dim = 1)
    
    return nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions
    
end


"""
Given the output of function `run_similarity_analogous`, organize the baseline and offset extensions
using significant cycles. 
"""
function organize_baseline_offset_extensions(analogous_P_full, significant_P, analogous_Q_full, significant_Q)
    # gather baseline extensions
    P_baseline = [i for i in analogous_P_full["baseline_bar_extension"] if i in significant_P]
    Q_baseline = [i for i in analogous_Q_full["baseline_bar_extension"] if i in significant_Q]

    # organize offset extensions
    P_offset_dict = select_significant_subcollection(analogous_P_full["offset_bar_extension"], significant_P)
    P_offset_bars = [v for (k,v) in P_offset_dict]

    Q_offset_dict = select_significant_subcollection(analogous_Q_full["offset_bar_extension"], significant_Q)
    Q_offset_bars = [v for (k,v) in Q_offset_dict]

    extensions = Dict()
    extensions["P_baseline"] = P_baseline
    extensions["P_offset"] = P_offset_bars
    extensions["Q_baseline"] = Q_baseline
    extensions["Q_offset"] = Q_offset_bars

    return extensions

end

#################################################################################
# functions for visualizations
#################################################################################

function plot_barcode(barcode; 
    color = :grey56, # default bar color
    selected_bars = [], # index of bars to highlight
    epsilon = missing, # if provided, only highlight the portion of bars on the right side of epsilon
    selection_color = :deeppink2,  # highlight color
    v_line = [], # if provided, draw vertical lines at values
    return_perm = false, # whether to return the permutation index or not
    kwargs...)

    # adjust linewidth, depending on number of intervals
    n = size(barcode,1)

    # find ordering according to birth time
    perm = sortperm(barcode[:,1])

    # non-inf maximum death time
    if filter(!isinf,barcode[:,2]) != []
        death_max = maximum(filter(!isinf,barcode[:,2])) 
    else
        death_max = maximum(barcode[:,1]) * 2
    end

    p = plot(framestyle = :box,
            top_margin = 5 * Plots.PlotMeasures.mm, 
            bottom_margin = 5 * Plots.PlotMeasures.mm, 
            yaxis = nothing;
            kwargs...)
    
    # plot all bars
    idx = 1
    for i in perm
        birth = barcode[i,1]
        death = barcode[i,2]
        
        # assign a death time if bar has infinite death time 
        if isinf(death)
            death = death_max * 1.2
        end
        if i in selected_bars
            
            # if epsilon is missing, highlight the entire bar
            if ismissing(epsilon)
                plot!(p,[birth, death], [idx, idx], legend = false, linecolor = selection_color, hover = "class " *string(i); kwargs...)
            
            # if epsilon is provided, only highlight the portion of the bar on the right side of epsilon    
            else 
                if birth <= epsilon
                    plot!(p,[birth, epsilon], [idx, idx], legend = false, linecolor = color, hover = "class " *string(i); kwargs...)
                    plot!(p,[epsilon, death], [idx, idx], legend = false, linecolor = selection_color, hover = "class " *string(i); kwargs...)
                else
                    plot!(p,[birth, death], [idx, idx], legend = false, linecolor = selection_color, hover = "class " *string(i); kwargs...)
                end
            end
        else
            plot!(p,[birth,death],[idx,idx], legend = false, linecolor = color, hover = "class " *string(i); kwargs...)
        end
        idx += 1
    end

    # plot vertical lines 
    if v_line != []
        plot!(v_line, seriestype="vline", linestyle = :dot, linecolor = :red)
    end

    ylims!((-1, n+1))
    
    if return_perm == true
        return p, perm
    else
        return p
    end
end


"""
    plot_analogous_bars()
Plots summary of the analogous bars method.
1. plots 4 barcodes: Barcode(VR(P)), Barcode(W(P,Q)), Barcode(W(Q,P)), Barcode(VR(Q)).
2. Highlights the selected Witness bar in Barcode(W(P,Q)) and Barcode(W(Q,P))
3. Highlights the **BASELINE** bar extension to Barcode(VR(P)) and Barcode(VR(Q)) at **epsilon_0**
"""
function plot_analogous_bars(extension_P, extension_Q; 
                            titlefontsize = 10,
                            lw_VR_P = nothing,
                            lw_W = nothing,
                            lw_VR_Q = nothing,
                            title_VR_P = "Baseline bar extension in Barcode(VR(P))",
                            title_W_PQ = "Selected bar in Barcode(W(P,Q))",
                            title_W_QP = "Selected bar in Barcode(W(Q,P))",
                            title_VR_Q = "Baseline bar extension in Barcode(VR(Q))",
                            kwargs...)

    # get variables
    d = extension_P["dim"]   
    barcode_VR_P = barcode(extension_P["C_VR"], dim = d)
    barcode_W_PQ = barcode(extension_P["C_W"], dim = d)
    barcode_W_QP = barcode(extension_Q["C_W"], dim = d)
    barcode_VR_Q = barcode(extension_Q["C_VR"], dim = d)
    
    # get bar extension to P
    if extension_P["Ybar_rep_tau"] == nothing
        bar_ext_P = []
        epsilon0_P = missing
        v_line_P = []
    else
        bar_ext_P = extension_P["Ybar_rep_tau"]
        epsilon0_P = extension_P["epsilon_0"]
        v_line_P = [epsilon0_P]
       
    end
    
    # get bar extension to Q
    if extension_Q["Ybar_rep_tau"] == nothing
        bar_ext_Q = []
        epsilon0_Q = missing
        v_line_Q = []
    else
        bar_ext_Q = extension_Q["Ybar_rep_tau"]
        epsilon0_Q = extension_Q["epsilon_0"]
        v_line_Q = [epsilon0_Q]
    end
    
    # plot
    p_VR_P = plot_barcode(barcode_VR_P,
                            selected_bars = bar_ext_P, title = title_VR_P,
                            titlefontsize = titlefontsize,
                            lw = lw_VR_P,
                            epsilon = epsilon0_P,
                            v_line = v_line_P;
                            kwargs...)
    p_W_PQ = plot_barcode(barcode_W_PQ,
                            selected_bars = extension_P["selected_bar"], 
                            selection_color = :deepskyblue2,
                            title = title_W_PQ,
                            titlefontsize = titlefontsize,
                            lw = lw_W;
                            kwargs...)
    p_W_QP = plot_barcode(barcode_W_QP,
                            selected_bars = extension_Q["selected_bar"], 
                            selection_color = :deepskyblue2,
                            title = title_W_QP,
                            titlefontsize = titlefontsize,
                            lw = lw_W;
                            kwargs...)
    p_VR_Q = plot_barcode(barcode_VR_Q,
                            selected_bars = bar_ext_Q, 
                            title = title_VR_Q,
                            titlefontsize = titlefontsize,
                            lw = lw_VR_Q,
                            epsilon = epsilon0_Q,
                            v_line = [epsilon0_Q];
                            kwargs...)
    
    title = plot(title = "Analogous bar", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    p = plot(title, p_VR_P, p_W_PQ, p_W_QP, p_VR_Q, layout = @layout([A{0.05h}; [B; C; D; E]]), size = (500, 1000))
    return p
end


"""
    plot_pY
Plot all parameter values of nontrivial pY on target barcode.
"""
function plot_pY(extension; lw = nothing, title = "", size = (500, 400))
    # get variables
    pY = extension["nontrivial_pY"]
    dim = extension["dim"]
    epsilon_0 = extension["epsilon_0"]
    if extension["comparison"] == "VR to VR"
        C_target = extension["C_Y"]
        barcode_title = "barcode(VR(Y))"
    elseif extension["comparison"] == "W to VR"
        C_target = extension["C_VR"]
        barcode_title = "barcode(VR)"
    elseif extension["comparison"] == "VR to W"
        C_target = extension["C_W"]
        barcode_title = "barcode(W)"
    end    
    
    if title == ""
        title = barcode_title
    end
    p_ext = plot_barcode(barcode(C_target, dim = dim), v_line = pY, lw = lw, title = title)
    plot(p_ext, size = size)
end


"""
    plot_cycle_single()
Plots a single point cloud P in 2-dimensions and a 1-dimensional cycle. 
"""
function plot_cycle_single(P; cycle = [], cycle_color = "black", cycle_lw = 5, kwargs...)
    # P: array of size (2, n) or (3,n)
    # cycle: [[v1, v2], [v3, v4], ...  ]
    
    # plot points P
    p = plot(P[1,:], P[2,:], 
            seriestype = :scatter, 
            label = "",
            framestyle = :box,
            xaxis = nothing,
            yaxis = nothing;
            kwargs...)
    
    # plot cycle
    for item in cycle
        p1, p2 = item
        p1_x, p1_y = P[:,p1]
        p2_x, p2_y = P[:,p2]
        plot!(p, [p1_x, p2_x], [p1_y, p2_y], color = cycle_color, lw = cycle_lw, label ="")
    end
    
    return p
end



"""
    plot_P_Q()
Plots point clouds P and Q in 2-dimensions
"""
function plot_P_Q(P, # array of size (m, 2)
                  Q; # array of size (n, 2)
                  P_color = "#008181", 
                  P_label = "P",
                  P_markersize = 5,
                  P_marker = :circle,
                  Q_color = "#ff8d00",
                  Q_label = "Q",
                  Q_markersize = 5,
                  Q_marker = :xcross,
                  kwargs...)
   # plot points P and Q on a square
    
    p = plot(framestyle = :box, yaxis = nothing, xaxis = nothing; kwargs...)
    
    # plot P
    scatter!(p, P[:,1], P[:,2], color = P_color, label = P_label, markersize = P_markersize, marker = P_marker)
    
    # plot Q
    scatter!(p, Q[:,1], Q[:,2], color = Q_color, label = Q_label, markersize = Q_markersize, marker = Q_marker)
    return p
end


"""
    plot_cycle
Plots both points P, Q and a 1-dimensional cycle. User can specify whetheer the cycle exists among P or Q.
"""
function plot_cycle(P, # array of size (m,2)
                    Q; # array of size (n,2)
                  cycle = [],
                  cycle_loc = "P",
                  cycle_color = :deeppink,
                  cycle_linewidth = 5,
                  P_color = "#008181", 
                  P_label = "P",
                  P_markersize = 5,
                  P_marker = :circle,
                  Q_color = "#ff8d00",
                  Q_label = "Q",
                  Q_markersize = 5,
                  Q_marker = :xcross,
                  kwargs...)
# plot one-dimensional cycle 
        
    p = plot_P_Q(P, Q, 
                 P_color = P_color, P_label = P_label, P_markersize = P_markersize, P_marker = P_marker,
                 Q_color = Q_color, Q_label = Q_label, Q_markersize = Q_markersize, Q_marker = Q_marker;
                 kwargs...)

    # specifiy the 0-simplices
    if cycle_loc == "P"
        PC = P
    else
        PC = Q
    end
    
    for simplex in cycle
        v1, v2 = simplex 
        v1_theta, v1_phi = PC[v1,1], PC[v1,2]
        v2_theta, v2_phi = PC[v2,1], PC[v2,2]
        
        plot!(p, [v1_theta, v2_theta], [v1_phi, v2_phi], label = "", color = cycle_color, lw = cycle_linewidth)

    end
    return p
end



"""
    plot_3D(<keyword arguments>)
Plots point clouds P and Q in 3-dimensions. 
"""
function plot_3D(P,
                 Q;
                 P_color = "#008181", 
                 P_label = "P",
                 P_markersize = 3,
                 P_marker = :circle,
                 Q_color = "#ff8d00",
                 Q_label = "Q",
                 Q_markersize = 3,
                 Q_marker = :xcross,
                 kwargs...)
  
    
    # plot P
    p = scatter3d(P[:,1],P[:,2],P[:,3], label = P_label, color = P_color, marker = P_marker, markersize = P_markersize; kwargs...)
 
    # plot Q
    scatter3d!(Q[:,1], Q[:,2], Q[:,3], label = Q_label, color = Q_color, marker = Q_marker, markersize = Q_markersize; kwargs...)

    return p 
end


"""
    plot_cycle_square_torus()
Plots a 1-dimensional cycle on a square torus. The function plots a 1-simplex only if it doesn't cross the square boundary
"""
function plot_cycle_square_torus(P, # array of size (m,2)
                                 Q; # array of size (n,2)
                                  cycle = [],
                                  cycle_loc = "P",
                                  cycle_color = :deeppink,
                                  cycle_linewidth = 5,
                                  P_color = "#008181", 
                                  P_label = "P",
                                  P_markersize = 5,
                                  P_marker = :circle,
                                  Q_color = "#ff8d00",
                                  Q_label = "Q",
                                  Q_markersize = 5,
                                  Q_marker = :xcross,
                                  kwargs...)
# plot one-dimensional cycle on square torus
        
    p = plot_P_Q(P, Q, 
                 P_color = P_color, P_label = P_label, P_markersize = P_markersize, P_marker = P_marker,
                 Q_color = Q_color, Q_label = Q_label, Q_markersize = Q_markersize, Q_marker = Q_marker;
                 kwargs...)

    # specifiy the 0-simplices
    if cycle_loc == "P"
        PC = P
    else
        PC = Q
    end
    
    for simplex in cycle
        v1, v2 = simplex 
        v1_theta, v1_phi = PC[v1,1], PC[v1,2]
        v2_theta, v2_phi = PC[v2,1], PC[v2,2]
        # If the simplex doesn't cross the square boundary, plot it:
        if (abs(v1_theta - v2_theta) <= 4) && (abs(v1_phi - v2_phi) <= 4)
            plot!(p, [v1_theta, v2_theta], [v1_phi, v2_phi], label = "", color = cycle_color, lw = cycle_linewidth)
        end
    end
    return p
end

function plot_cycle_single_square_torus(P; # array of size (m,2)
    cycle = [],
    cycle_color = :deeppink,
    cycle_linewidth = 5,
    kwargs...)
    # plot one-dimensional cycle on square torus
    p = plot(framestyle = :box, yaxis = nothing, xaxis = nothing)

    # plot P
    scatter!(p, P[:,1], P[:,2]; kwargs...)    

    PC = P
    for simplex in cycle
        v1, v2 = simplex 
        v1_theta, v1_phi = PC[v1,1], PC[v1,2]
        v2_theta, v2_phi = PC[v2,1], PC[v2,2]
        # If the simplex doesn't cross the square boundary, plot it:
        if (abs(v1_theta - v2_theta) <= 4) && (abs(v1_phi - v2_phi) <= 4)
            plot!(p, [v1_theta, v2_theta], [v1_phi, v2_phi], label = "", color = cycle_color, lw = cycle_linewidth)
        end
    end
    return p
end



#################################################################################
# Functions implementing Dowker's Theorem
#################################################################################
matchrow(a,B) = findfirst(i->all(j->a[j] == B[i,j],1:size(B,2)),1:size(B,1))

"""
    apply_Dowker(W_PQ, W_QP; <keyword arguments>)
Given Witness filtrations `W_PQ` (landmark: P, witness: Q) and `W_QP` (landmark: Q, witness: P), 
use Dowker's Theorem to find the correspondence between bars in barcode(W(P,Q)) and barcode(W(Q,P)).

### Arguments
- `W_PQ`(dict): Output of `compute_Witness_persistence(D_P_Q, maxdim = dim)`
- `W_QP`(dict): Output of `compute_Witness_persistence(D_Q_P, maxdim = dim)`
- dim(int): dimension. Defaults to 1

### Outputs
- `P_to_Q` (dict): of correspondence between barcode(W(P,Q)) and barcode(W(Q,P)).
        `P_to_Q[i] = j` implies that bar `i` in barcode(W(P,Q)) matches bar `j` in barcode(W(Q,P))
"""
function apply_Dowker(
    W_PQ,
    W_QP;
    dim = 1)

    P_to_Q = Dict()
    barcode_W_PQ = barcode(W_PQ["eirene_output"], dim = dim)
    barcode_W_QP = barcode(W_QP["eirene_output"], dim = dim)
    n = size(barcode_W_PQ)[1]
    for i=1:n

        birth_rows = matchrow(barcode_W_PQ[i,:], barcode_W_QP)
        #if size(birth_rows)[1] > 1
        #    print("ERROR: multiple bars with same birth and death time")
        #    break
        #else
        P_to_Q[i] = birth_rows
        #end
    end
    return P_to_Q
end


"""
variation of the function "apply_dowker".
Match bars using birth time alone.
Assume that the birth times of the Witness filtration are all unique.
If W_QP is created with a maximum parameter ("param_max"), 
then barcode_W_PQ differs from barcode_W_QP, and that's okay.
We only need this function for the following purpose: Given a bar in barcode_W_PQ, 
find the corresponding bar in barcode_W_QP. 
"""
function apply_Dowker_birth(
    W_PQ,
    W_QP;
    dim = 1)

    P_to_Q = Dict()
    barcode_W_PQ = barcode(W_PQ["eirene_output"], dim = dim)
    barcode_W_QP = barcode(W_QP["eirene_output"], dim = dim)

    # check that all birthtimes are unique
    birthtimes = barcode_W_PQ[:,1]
    if length(unique(birthtimes)) != length(birthtimes)
        print("Error: The Witness barcode must have unique birth times.")
        return nothing
    else
        n = size(barcode_W_PQ)[1]
        for i=1:n
            birth_rows = findfirst(item -> item == barcode_W_PQ[i,1], barcode_W_QP[:,1])
            P_to_Q[i] = birth_rows
        end
    end
    return P_to_Q
end

"""
    find_barycentric_subdivision(simplex)
Finds the barycentric subdivision of a 1-dimensional simplex

### Arguments
- simplex: 1-dimensional simplex of form [i, j], consisting of the i-th and j-th vertices

### Returns
- a list containing the vertices of the barycentric subdivision
"""
function find_barycentric_subdivision(simplex)
    return [[simplex[1]], simplex, [simplex[2]]]
    
end

"""
    find_witness_column(relations, rows)
Given rows in a relations matrix, return a witness column. In particular, return witness column with smallest index

### Arguments
- relations: (array) binary relations matrix
- rows: (list) of rows

### Returns
- col: (int) that witnesses the given rows
"""
function find_witness_column(relations, rows)
    sub_relations = relations[rows, :]
    n_cols = size(relations, 2)
    n_rows = size(rows, 1)

    # compute the sum of all rows in sub matrix
    S = sum(sub_relations, dims = 1)
    idx = findfirst(x -> x == n_rows, S)
    
    if idx == nothing
        print("There is no witness for selected rows")
    else
        col = idx[2]
        return col
    end
end

"""
    barycentric_to_column_complex(barycentric_simplex, relations)
Given a list of vertices in the barycentric subdivision, return 1-simplices in the column complex it maps to.
That is, given a simplex in the barycentric subdivision, return all witnesses of the vertices.

### Arguments
- relations: (array) relations matrix

### Returns
"""
function barycentric_to_column_complex(barycentric_simplex, relations)
    
    mapped_cols = [find_witness_column(relations, vertex) for vertex in barycentric_simplex]
    mapped_1simplices = [[mapped_cols[1], mapped_cols[2]], [mapped_cols[2], mapped_cols[3]]]
    mapped_1simplices = [item for item in mapped_1simplices if item[1] != item[2]]
    
    return mapped_1simplices
end

"""
    find_column_cycle_via_Dowker(row_cycle, relations)
Given a cycle in the row complex, find the corresponding cycle in the column complex using Dowker's Theorem
"""
function find_column_cycle_via_Dowker(row_cycle, relations)
    # check that input is a cycle
    col_cycle = []
    for row_simplex in row_cycle
        barycentric = find_barycentric_subdivision(row_simplex)
        col_simplex = barycentric_to_column_complex(barycentric, relations)
        append!(col_cycle, col_simplex)
    end
    col_cycle = [sort(item) for item in col_cycle]
    return col_cycle
end



"""
    find_Dowker_cycle_correspondence(cycle_W_PQ, param, D_P_Q)
Given a cycle in the Witness complex W(P,Q) and a parameter at which the cycle is nontrivial, 
find its corresponding cycle in W(Q,P) via Dowker's Theorem. 
"""
function find_Dowker_cycle_correspondence(cycle_W_PQ, param, D_P_Q)
    # find the binary relations matrix at parameter epsilon
    relations = replace(x -> x <= param, D_P_Q)

    # find corresponding cycle in W(Q,P) using Dowker Theorem
    cycle_W_QP = find_column_cycle_via_Dowker(cycle_W_PQ, relations)
    
    return cycle_W_QP
end



#################################################################################
# Functions related to Witness complexes
#################################################################################

"""
    select_vertices_of_Witness_complex(D, threshold)
Given a cross-distance matrix D, select the vertices that exist at W^{threshold}. Since W^{threshold} may have fewer vertices than given (rows of D), we need to index the vertices of W^{threshold}.

### Arguments
- `D`: (array) of dross-distance matrix
- `threshold`: (float) parameter to build W^{threshold}

### Returns
- if all vertices are selected, return nothing, nothing
- if a subset of vertices are selected, then return the following
    - `Wpsi_vertex_to_default_vertex`: (dict)
        `Wpsi_vertex_to_default_vertex[i] = j` means vertex i of Wpsi corresponds to vertex j in default vertex (or row j in matrix D)
    - `default_vertex_to_Wpsi_vertex`: (dict) 
        `default_vertex_to_Wpsi_vertex[i] = j` means vertex i (or row i of matrix D) corresponds to vertex j in Wpsi
"""
function select_vertices_of_Witness_complex(D; threshold = Inf)
    
    # Find rows of D that has at least one entry <= threshold
    D_binary = replace(x -> x <= threshold, D)
    idx = findall(x -> x >0 , [(sum(D_binary, dims = 2)...)...])
    

    # Need to index the vertices of Wpsi
   
    if size(idx, 1) == size(D, 1)
        return nothing, nothing
    else
        # create correspondence between vertices of Wepsilon and default vertices corresponding to rows of D
        Wpsi_vertex_to_default_vertex = Dict(i => idx[i] for i=1:size(idx,1))
        default_vertex_to_Wpsi_vertex = Dict(val => key for (key, val) in Wpsi_vertex_to_default_vertex)
    
        return Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex
    end
end


function build_Witness_complex(
    D::Array, 
    threshold::Float64;
    maxdim::Int64=2)
    
    # select vertices that exist at given threshold
    Wepsilon_vertex_to_default_vertex, default_vertex_to_Wepsilon_vertex = select_vertices_of_Witness_complex(D, threshold = threshold)    
    if Wepsilon_vertex_to_default_vertex == nothing
        D_sub = D
        n = size(D, 1)
    else
        W_idx = sort(collect(keys(default_vertex_to_Wepsilon_vertex)))
        D_sub = D[W_idx, :]
        n = size(W_idx,1)
    end
        
    Witness_complex = []
    fv = []
    for d = 0:maxdim
        simplices = []
        fv_dim = []

        candidates = collect(combinations(1:n, d+1))
        for item in candidates
            t = minimum(maximum(D_sub[item,:], dims = 1))
            if t <= threshold
                push!(simplices, item)
                push!(fv_dim, t)
            end
        end

        push!(Witness_complex, simplices)
        push!(fv, fv_dim)
    end
    
    # turn the list of simplices to dictionary
    # Create dictionaries
    W_simplex2index = Dict([i] => i for i =1:n)

    # higher dimensional simplices
    for d = 1:maxdim
        for (idx, val) in enumerate(Witness_complex[d+1])
            W_simplex2index[val] = idx
        end
    end
    
    # rverse simplex2index
    W_index2simplex = Dict((val, size(key,1)-1) => key for (key, val) in W_simplex2index)
    
    return W_index2simplex, W_simplex2index, fv, Wepsilon_vertex_to_default_vertex, default_vertex_to_Wepsilon_vertex
end


"""
    compute_Witness_persistence(D; <keyword arguments>)
Given a cross-distance matrix `D`, build the Witness filtration using rows as landmarks (0-simplices) and columns as witnesses.

### Arguments
- `D`: Distance matrix between landmarks and witnesses.
    rows:landmarks
    columns: witnesses
- `maxdim`: Maximum dimension of interest. Defaults to 1
- `param_max`: maximum parameter to build the Witness filtration.

### Outputs 
- `W`: Dictionary containing the following information.
    - `param_max`: threshold value used in `build_Witness_complex`
    - `index2simplex`: indexing of simplices in the Witness complex
    - `simpelx2index`: referse of index2simplex
    - `distance_matrix`: D 
    - `fv`: birth times of simplices in the Witness complex
    - `eirene_output`: dictionary output of Eirene on the Witness filtration. 
    - `W_vertex_to_default_vertex`: Output of `select_vertices_of_Witness_complex``
    - `default_vertex_to_W_vertex`: Output of `select_vertices_of_Witness_complex`
"""
function compute_Witness_persistence(
    D::Array;
    maxdim::Int64 = 1,
    param_max = Inf)
    
    # select max parameter
    if param_max == Inf
        param_max = minimum(maximum(D, dims = 1))
    end
        
    # Index simplices in the Witness complex and find face value (birth time) in filtration.
    W_index2simplex, W_simplex2index, W_fv, W_vertex_to_default_vertex, default_vertex_to_W_vertex = build_Witness_complex(D, param_max, maxdim = maxdim + 1)
    
    # prepare input for Eirene
    W_rv, W_cp, W_ev, W_fv = create_CSC_input_from_fv(W_fv, W_index2simplex, W_simplex2index)
    
    # run Eirene
    C_W = eirene(rv = W_rv, cp = W_cp, ev = W_ev, fv = W_fv, record = "all", maxdim = maxdim)
    
    # save relevant info
    W = Dict()
    W["param_max"] = param_max
    W["index2simplex"] = W_index2simplex
    W["simplex2index"] = W_simplex2index
    W["distance_matrix"] = D
    W["fv"] = W_fv
    W["eirene_output"] = C_W
    W["W_vertex_to_default_vertex"] = W_vertex_to_default_vertex
    W["default_vertex_to_W_vertex"] = default_vertex_to_W_vertex
    
    return W
end

"""
    get_Witness_cyclerep(W, class_num)
Find the classrep of a Witness filtration in vertex format.
"""
function get_Witness_cyclerep(W;class_num = nothing, dim = 1)
    
    cyclerep_idx = classrep(W["eirene_output"], dim = dim, class = class_num)
    
    cyclerep = []
    for i in cyclerep_idx
       push!(cyclerep, W["index2simplex"][(i, dim)]) 
    end
    return cyclerep
end

function get_cyclerep_Eirene(C, classnum)
    cyclerep = classrep(C, class = classnum, dim = 1)
    cyclerep = [col for col in eachcol(cyclerep)]
    return cyclerep
end
        

########################################################################################
# functions for working with Eirene's CSC format
########################################################################################
function create_CP(
    ev)
    # create `cp` for CSC format input to Eirene 
    """
    --- input ---
    ev: (array)
        ev[d]: number of (d-1) dimensional simplices
    --- output ---
    cp: (array) colptr to be used as input to Eirene
    """
    cp = []
    for d = 1:size(ev,1)
        if d == 1
            cp_dim = ones(Int64, ev[d]+1)
            push!(cp, cp_dim)
        else
            cp_dim = collect(StepRange(1, Int8(d), ev[d]*d+ 1))
            cp_dim = convert(Array{Int64,1}, cp_dim)
            push!(cp, cp_dim)
        end
    end
    return cp
end

function create_rv(
    ev,
    index2simplex,
    simplex2index
    )
    # create "rv" for CSC format input to Eirene
    
    """
    --- input ---
    ev: (array)
        ev[d]: number of (d-1) dimensional simplices
    --- output ---
    rv: (array) rv vector to be used as input to Eirene
    """
    maxdim = size(ev, 1)
    rv = []
    for d = 0:maxdim - 1
        # 0-simplices
        if d == 0
            push!(rv, Array{Int64,1}(undef, 0))

        # 1-simplices
        elseif d == 1
            rv_d = Array{Int64,1}(undef, 0)
            for i = 1:ev[d+1]
                append!(rv_d, index2simplex[(i,d)])
            end
            push!(rv, rv_d)

        # higher-dim simplices
        else
            rv_d = Array{Int64,1}(undef, 0)
            for i = 1:ev[d+1]
                boundary_idx = [simplex2index[item][1] for item in combinations(index2simplex[i,d], d)]
                append!(rv_d, boundary_idx)
            end
            push!(rv, rv_d)
        end
    end
    return rv
end

function create_CSC_input_from_fv(
    fv,
    index2simplex,
    simplex2index)
    
    # Find input for Eirene in CSC format (rv, cp, ev, fv), given the face values
    
    ##### ev: number of simplices in each dimension
    ev = [size(item,1) for item in fv]
    
    ##### boundary matrix CSC format: columns
    cp = create_CP(ev)
    
    ##### boundary matrix CSC format: rows
    rv = create_rv(ev, index2simplex, simplex2index)
    return rv, cp, ev, fv
    
end

function create_CSC_input_from_distance(
    n_simplices::Array{Int64,1}, 
    index2simplex::Dict, 
    simplex2index::Dict, 
    D_Y::Array{Float64,2})
    # Find input for Eirene in CSC format (rv, cp, ev, fv).
    # Use distance matrix D_Y to find the birth time of each simplex.
    """
    --- input ---
    n_simplices: output of function 'index_simplices_Zpsi'
    index2simplex: output of function 'index_simplices_Zpsi'
    simplex2index: output of function 'index_simplices_Zpsi'
    D_Y: distance matrix used for C_Y
    --- output ---
    rv: rv for Zpsi
            row values of the boundary matrix
    cp: cp for Zpsi
    ev: ev for Zpsi
            number of simplices in each dimension
    fv: fv for Zpsi
            birth time of simplices according to D_Y
    """

    ##### ev: number of simplices in each dimension
    maxdim = size(n_simplices,1)
    ev = n_simplices
    
    ##### boundary matrix CSC format: columns
    cp = create_CP(ev)
    
    ##### boundary matrix CSC format: rows
    rv = create_rv(ev, index2simplex, simplex2index)

    ##### fv: birth time of each simplex using D_X
    fv = []

    # 0-simplices have birth time of zero.
    fv_0 = zeros(ev[1])
    push!(fv, fv_0)

    # higher dimensional simplices
    for d = 1:maxdim-1
        fv_d = []
        for i = 1:ev[d+1]
            # find vertex representation of simplex using original indexing
            simplex = index2simplex[(i,d)]

            # find its birth time according to D_Y
            push!(fv_d, find_birth_time(simplex, D_Y))
        end

        push!(fv, fv_d)
    end
    return rv, cp, ev, fv
end

"""
    create_CSC_input_from_W_distance(;<keyword arguments>)
Given a complex, create Witness complex based on a given distance matrix. That is, use the distance matrix of the Witness complex to find the birth time. 
Return the input for Eirene in CSC format (rv, cp, ev, fv).

### Arguments
- `n_simplices`: number of simplices in complex
- `index2simplex` (dict): simplex index to simplex
- `simplex2index` (dict): simplex (as a list of vertices) to simplex
- `D_W`: the Witness distance matrix used to determine the birth time of each simplex

### Outputs
- `rv`: rv for W
    row values of the boundary matrix
- `cp`: cp for W
- `ev`: ev for W
    number of simplices in each dimension
- `fv`: fv for W
    birth time of simplices according to `D_W`
"""
function create_CSC_input_from_W_distance(
    n_simplices::Array{Int64,1}, 
    index2simplex::Dict, 
    simplex2index::Dict, 
    D_W::Array{Float64,2})

    ##### ev: number of simplices in each dimension
    maxdim = size(n_simplices,1)
    ev = n_simplices
    
    ##### boundary matrix CSC format: columns
    cp = create_CP(ev)
    
    ##### boundary matrix CSC format: rows
    rv = create_rv(ev, index2simplex, simplex2index)

    ##### fv: birth time of each simplex 
    fv = []

    # 0-simplices have birth time of zero.
    fv_0 = zeros(ev[1])
    push!(fv, fv_0)

    # higher dimensional simplices
    for d = 1:maxdim-1
        fv_d = []
        for i = 1:ev[d+1]
            # find vertex representation of simplex using original indexing
            simplex = index2simplex[(i,d)]

            # find its birth time according to D_W
            push!(fv_d, minimum(maximum(D_W[simplex,:], dims = 1)))
        end

        push!(fv, fv_d)
    end
    return rv, cp, ev, fv
end


#################################################################################
# Other helper functions
#################################################################################
function get_vertex_perm(C::Dict)
    # Eirene permutes the vertex indices.
    # Get vertex permutation information from Eirene.
    # note C["nvl2ovl"][i] = k, where i is the Eirene index and k is the original vertex index. 

    """
    --- input ---
    C: (dict) output from Eirene
    --- output ---
    v_perm: (arr) v_perm[k] = i, where k: original vertex index, 
            and i: corresponding Eirene index.
    
            ex) If C = Eirene(distance_matrix), 
            then k corresponds to the kth row/column of distance_matrix
    """

    n_vertices = size(C["nvl2ovl"],1)
    v_perm = zeros(Int64,n_vertices)
    for i=1:n_vertices
        idx = findall(x->x==i, C["nvl2ovl"])[1]
        v_perm[i] = idx
    end
    return v_perm
end


function select_simplices_at_psi(
    C::Dict, 
    psi::Float64;  
    maxdim::Int64 = 2)
    # Select the simplices in C that exist at parameter psi and return their indices. 
    # Note: Indexing is according to C, which is Eirene's internal indexing of simplices 

    """
    --- input ---
    C: output of running eirene
    psi: parameter
    maxdim: maximum dimension of simplices
    --- output ---
    simplices: array of simplices that exist at parameter delta 
            simplices[i] is an array of (i-1) dimensional simplices that exist at parameter delta
    """
    rv, cp, fv = Eirene_var.eirened2complex(C)
    simplices = []
    for d = 1:maxdim + 1
        push!(simplices, findall(x -> x <= psi, fv[d]))
    end
    return simplices
end

function create_idx_to_simplex_V(
    C::Dict; 
    maxdim::Int64 = 2)
    # Creates a dictionary from index of a simplex to its vertex representation, 
    # where the vertices are indexed according to C. 
    """
    --- input ---
    C: (Dict) output of Eirene_var. The maximum dimension of C must be at least "maxdim"-1 
    maxdim: (int) maximum dimension of simplices to consider
    --- output ---
    idx_smplx: (Dict) with (index, dimension) as key and its vertex representation as value
                ex: idx_smplx[(idx, dim)] = [v0, v1, ..., v_dim]

    --- note ---
    note: dim >= 1
    !!!!! NOTE !!!!! : The vertices are indexed according to Eirene's internal representation in C.
        (not the user-specified order)
    """
    # check that Eirene dictionary C has enough dimensions
    if C["input"]["maxdim"] < maxdim - 1
        throw(error("maxdim of Eirene output is too small."))
    end

    # get boundary matrix
    rv, cp, fv = Eirene_var.eirened2complex(C)
        
    idx_smplx = Dict()
    for d = 1:maxdim
        # 1-simplices
        if d == 1
            n = size(fv[d+1], 1)
            for i = 1:n
            # find vertex representation [v0, v1]
            idx_smplx[(i, 1)] = rv[2][i * 2 - 1:i * 2] 
            end
            
        # higher dimensional simplices
        else
            n = size(fv[d+1],1)
            for i = 1:n
                # find simplex as a list of its boundary
                # simplex = [b_1, b_2, ... ,b_{d+1}], where each b_i is the index 
                # of the i-th boundary simplex.
                simplex = rv[d+1][i * (d + 1) - d: i*(d+1)] 
                
                # find vertices of the simplex
                vertices = []
                # it suffices to just consider two of the boundary cells
                for item in simplex[1:2]
                    append!(vertices, idx_smplx[(item, d-1)])
                end
                idx_smplx[(i,d)] = unique(vertices)
            end
        end
    end
    return idx_smplx
end

function index_simplices_Zpsi(
    C_Z::Dict, 
    psi::Float64;
    maxdim::Int64 = 1)
    # Find simplices in the fixed complex Z_psi and devise an indexing scheme. 
    """
    --- input ---
    C_Z: Dictionary output of Eirene
    psi: parameter for building Z_psi
    maxdim: maximum dimension of homology of interest 

    --- output ---
    n_simplices: (array)
            n_simplices[i] is the number of (i-1) dimensional simplices of C_Z.
    Zpsi_index2simplex: (dict) given an index and dimension, returns the vertex representation of the simplex.
            Zpsi_index2simplex[(idx, dim)] = [v0, ... , vn]
    Zpsi_simplex2index: (dict) reverse of Zpsi_index2simplex
            Zpsi_simplex2index[[v0, ... , vn]] = idx

    !!!!! NOTE !!!!!
    In Zpsi_index2simplex and Zpsi_simplex2index, the vertices are ordered according to the original indexing, and not C_Z.
    That is, if C_Z was obtained from distance matrix D_Z, vertex "i" corresponds to the i-th row and column of D_Z
    """

    ##### 1. Find all simplices of C_Z, their index (according to C_Z), and their vertex representation (using original indexing)
    Z_index2simplexV_eirene = create_idx_to_simplex_V(C_Z; maxdim = maxdim + 1)

    # express each simplex in C_Z using original vertices
    v_perm = C_Z["nvl2ovl"]
    Z_index2simplexV_orig = Dict(key => v_perm[value] for (key, value) in Z_index2simplexV_eirene)

    # reverse
    Z_simplex2indexV_orig = Dict(val => key for (key, val) in Z_index2simplexV_orig)

    ##### 2. create list of simplices that exist in Zpsi
    # note: "simplices_psi" is a list of indices, where the index corresponds to Eirene's indexing of C_Z
    simplices_psi = select_simplices_at_psi(C_Z, psi, maxdim = maxdim + 1)

    ##### 3. Create indexing of simplices in Z_psi

    # 0-simplices
    n0 = size(simplices_psi[1],1)
    Zpsi_simplex2index = Dict( [i] => v_perm[i] for i =1:n0)

    # higher dimensional simplices
    for d = 1:size(simplices_psi, 1)-1
        for (index, value) in enumerate(simplices_psi[d+1])
            simplex = sort(Z_index2simplexV_orig[(value, d)])
            Zpsi_simplex2index[simplex] = index
        end
    end

    # reverse dictionary
    Zpsi_index2simplex = Dict((val, size(key,1)-1) => key for (key, val) in Zpsi_simplex2index)

    # find number of simplices in each dimension
    n_simplices = [size(item,1) for item in simplices_psi]

    return n_simplices, Zpsi_index2simplex, Zpsi_simplex2index
end


function find_birth_time(
    simplex::Array, 
    D::Array{Float64,2})
    # Given a simplex (written as a list of vertices), find its birth time using distnace matrix D
    """
    --- input ---
    simplex: array of vertices.
            ex) simplex = [1,2,3]
    D: distance matrix
    --- output ---
    time: (float) birth time of given simplex. 
    """
    time = 0
    for item in combinations(simplex, 2)
        time = max(time, D[item[1], item[2]])
    end
    return time
end

"""find_cycle_death
Find the death time of a given cycle in a persistence module

### Arguments
- `cycle`: form [[v^1_1, v^1_2], [v^2_1, v^2_2], ..., ]
- `C`: (dict) output from Eirene
- `D`: (array) distance matrix used for C
"""
function find_cycle_death(cycle, C, D)
    dim = size(cycle[1],1)-1
    
    # find bar-representations of given cycle
    cycle_birth = maximum([find_birth_time(item, D) for item in cycle])
    bar_rep = find_bar_representation(cycle, C, D, cycle_birth)
    
    # find death time from the bar-representations
    cycle_death = maximum(barcode(C, dim = dim)[bar_rep,2])
    
    return cycle_death
end

"""find_cycle_birth_in_Witness
Find the birth time of a given cycle in a Witness compelx
"""
function find_cycle_birth_in_Witness(cycle, W)
    dim = size(cycle[1],1)-1
    C_W = W["eirene_output"]
    D = W["distance_matrix"]
    # find bar-representations of given cycle
    cycle_birth = maximum([minimum(maximum(D[item,:], dims = 1)) for item in cycle])
    bar_rep = find_bar_representation_W(cycle, W, cycle_birth)
    
    # find death time from the bar-representations
    cycle_birth = maximum(barcode(C_W, dim = dim)[bar_rep,1])
    
    return cycle_birth
end


"""find_cycle_death_in_Witness
Find the death time of a given cycle in a Witness compelx
"""
function find_cycle_death_in_Witness(cycle, W)
    dim = size(cycle[1],1)-1
    C_W = W["eirene_output"]
    D = W["distance_matrix"]
    # find bar-representations of given cycle
    cycle_birth = maximum([minimum(maximum(D[item,:], dims = 1)) for item in cycle])
    bar_rep = find_bar_representation_W(cycle, W, cycle_birth)
    
    # find death time from the bar-representations
    cycle_death = maximum(barcode(C_W, dim = dim)[bar_rep,2])
    
    return cycle_death 
end
    

"""
    get_multiclass_cyclerep
Finds the cycle rep of one or more classes. 

### Arguments
- `C`: eirene dictionary
- `cycles`: array of class numbers. ex) [1] or [1,2,3]

### Outputs
- `multi_cyclerep`: cycle representative of form [[x, y], [z, w], ... ]
"""
function get_multiclass_cyclerep(C, cycles; dim = 1)
    multi_cyclerep = []
    for i in cycles
        cycle_rep = classrep(C, dim = dim, class = i, format = "vertex x simplex")
        append!(multi_cyclerep, [sort(cycle_rep[:,i]) for i = 1:size(cycle_rep,2)])
    end
    
    # select simplices that occur odd number of times
    multi_cyclerep = select_odd_count(multi_cyclerep)
    
    return multi_cyclerep
end


function find_linear_combinations(
    candidates)
    # Given an array of integers (corresponding to vectors), find all non-trivial linear combinations
    """
    --- input ---
    candidates: (array) of integers
    --- output ---
    linear_comb: (array) of possible non-trivial linear combinations of candidates
    """

    linear_comb = []
    for i = 1:size(candidates, 1)
        comb = collect(combinations(candidates, i))
        append!(linear_comb, comb)
    end
    return linear_comb
end

function select_odd_count(
    orig_list::Array)
    # given an array, return the elements that occur odd number of times.
    """
    --- input ---
    orig_list: (N-element array)
    --- output ---
    new_list: (M-element array) containing only the elements that occur odd number of times
            in orig_list.
    """

    count = countmap(orig_list)
    new_list = [item for item in orig_list if count[item] % 2 != 0]
    return unique(new_list)
end

function express_x_CZpsi(
    C_Z::Dict, 
    class_num::Int64, 
    Z_psi_simplex2index::Dict;
    dim::Int64=1)
    # Express [x] in H_n(C_Z) as a chain (list of indices) using the indexing of C_Z_psi)
    """
    --- input ---
    C_Z: (dict) output of Eirene
    class_num: (int) class number of cycle [x] of interest
    Z_psi_simplex2index: (dict) simplex indexing of Z_psi.
                            output of function 'index_simplices_Zpsi'
    dim: (int) dimension of cycle [x]
    --- output ---
    """
    # express [x] using vertices.
    # note: cycle_rep format is "vertex x simplex", where vertices are indexed using the original indexing
    cycle_rep = classrep(C_Z, class = class_num, dim = dim)

    chain = []
    #  for each simplex in [x]
    for i = 1:size(cycle_rep, 2)
        # express simplex using original vertices
        simplex = cycle_rep[:,i]

        # find index in C_Z_filt
        idx = Z_psi_simplex2index[sort(simplex)]
        append!(chain, idx)
    end    
    return chain
end
    

function bounding_chain(C;chain=zeros(Int64,0),dim=1)
    # Check if a given chain is a boundary
    
    ##### CREDITS: This function was written by Greg Henselman-Petrusek. To be included in a future version of Eirene #####
    ##### https://github.com/Eetion/Eirene.jl #####
    
	if isempty(chain)
		return zeros(Int64,0)
	elseif !isempty(Eirene_var.chainboundary(C,chain=chain,dim=dim))
		print("there is no bounding chain"); return nothing
	else
		sd 			= 	dim+1;
		Lrv 		= 	C["Lrv"][sd+1]
		Lcp 		= 	C["Lcp"][sd+1]
		Rrv 		= 	C["Rrv"][sd+1]
		Rcp 		= 	C["Rcp"][sd+1]
		numrows 	= 	Eirene_var.boundarycorank(C,dim=dim)
		translator 	= 	zeros(Int64,Eirene_var.complexrank(C,dim=dim))
		translator[C["tid"][sd+1]] = 1:numrows
		rv 			= 	findall(!iszero,translator[chain])
		rv 			=  	translator[chain[rv]]
		cp 			= 	[1,length(rv)+1]
		rv,cp 		=  	Eirene_var.spmmF2silentLeft(Lrv,Lcp,rv,cp,numrows)
		rv,cp 		=  	Eirene_var.spmmF2silentLeft(Rrv,Rcp,rv,cp,numrows)
		#
		# recall that plo = the (ordered) subvector consisting of
		# the first (rank(boundary operator)) elements of tid
		#
		if maximum(rv) > length(C["plo"][sd+1])
			#print("there is no bounding chain");  # NOTE: Iris removed print
			return nothing
		else
			ocg2rad = 	C["ocg2rad"]
			grain 	= 	C["grain"][sd+1]
			names 	= 	C["phi"][sd+1]
			return 	names[rv]
		end
	end
end


function check_homologous_cycles(
    v_chain::Array, 
    w_chain::Array, 
    C::Dict)
    # check if two given cycles [v], [w] are homologous in C
    """
    --- input ---
    v_chain: (array) of indices of simplices of [v].
            Note that simplex indexing must be consistent with C.
    w_chain: (array) of indices of simplices of [w].
            Note that simplex indexing must be consistent with C. 
    C: (dict) output of eirene. When running, eirene, must use the argument `record = "all"`
    --- output ---
    (bool) true if [v] = [w] in C
            flase if [v] != [w] in C
    """
    # chain of [v] + [w]
    chain_vw = vcat(v_chain, w_chain)

    # only keep simplices that occur odd number of times
    chain_vw = select_odd_count(chain_vw)

    # if chain of v+w is trivial:
    if chain_vw == []
        return true
    else
        # check if [v] + [w] = 0 in H_n(C)
        bounder = bounding_chain(C, chain = chain_vw)
        if bounder == nothing
            return false
        else
            return true
        end
    end
end


"""
    find_terminal_class_in_VR(C, D; <keyword arguments>)
Given a bar in barcode(VR), find the terminal class and parameter psi. 

### Arguments
- `C`: (dict) Eirene output 
- `D`: (array) distance matrix used for computing persistence C
- `bar`: (Int64) Selected bar in barcode(C, dim = dim)
- `dim`: (Int) defaults to 1

### Returns
- `tau`: (array) terminal class (cycle representative of selected bar) expressed in vertex notation. 
    format: [[v0, v1], [v1, v2], ... , [v_i, v_n]]
- `psi`: (float) parameter immediately prior to the death time of selected bar in barcode. Corresponds to delta(tau)-1 in paper. 
"""
function find_terminal_class_in_VR(
    C::Dict, 
    D ::Array{Float64, 2}; 
    bar::Int64 = 0,
    dim = 1)

    rep = classrep(C, class = bar, dim = dim, format = "vertex x simplex")
    tau = [sort(rep[:,i]) for i = 1:size(rep,2)]
        
    # find parameter delta(tau)-1
    d = barcode(C, dim = dim)[bar, 2]
    psi = maximum(D[D.< d])
    
    return tau, psi
end


"""
    find_terminal_class_in_W(W; <keyword arguments>)
Given a bar in barcode(W), find the terminal class and parameter psi. Express cycle representative using DEFAULT VERTEX INDEXING 
(and not the Witness vertex indexing)

### Arguments
- `W`: (dict) output of `compute_Witness_persistence`
- `bar`: (Int64) Selected bar in barcode(W)
- `dim`: (Int) defaults to 1

### Returnsx
- `tau`: (array) terminal class (cycle representative of selected bar) expressed in vertex notation. 
    format: [[v0, v1], [v1, v2], ... , [v_i, v_n]]
- `psi`: (float) parameter immediately prior to the death time of selected bar in barcode. Corresponds to delta(tau)-1 in paper. 
"""
function find_terminal_class_in_W(
    W::Dict; # dictionary output of "compute_Witness_persistence" 
    bar::Int64 = 0,
    dim = 1)
    
    C_W = W["eirene_output"]
    D = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_vertex_to_default_vertex = W["W_vertex_to_default_vertex"]
    
    # find class rep of bar
    rep = classrep(C_W, class = bar, dim = dim)
    
    # express as a collection of simplices
    tau = [W_index2simplex[(item, dim)] for item in rep]
        
    if W_vertex_to_default_vertex != nothing
        tau = [[W_vertex_to_default_vertex[item] for item in simplex] for simplex in tau]
    end
    
    # find parameter delta(tau)-1
    d = barcode(C_W, dim = dim)[bar, 2]
    psi = maximum(D[D.< d])
    
    return tau, psi
end


"""build_auxiliary_filtration
Build the auxiliary filtration ``Z^{\\psi} \\cap Y^{\\bullet}``

### Arguments
- `C_Z`: (dict) Eirene output of Z
- `psi`: (float) parameter
- `D_Y`: (array) Distance matrix used in Y.
    If format == "VR to W", then D_Y is the distance matrix used to build the Witness complex Y. 
- `format`: (str) defaults to "VR to VR"
    * If format == "VR to VR", both `Z` and `Y` are Vietoris-Rips complexes.  
    * If format == "VR to W", then `Z` is a Vietoris-Rips complex and `Y` is a Witness complex.  
- `dim`: (int) defaults to 1
- `W_index2simplex`: (dict) Required when format is "W to VR".
    When Z is the Witness complex, `W_index2simplex` stores all indexing of simplices in Z.
- `W_fv`: (array) Required when format is "W to VR"
    When Z is the Witness complex, `W_fv` stores all face values of simplices in the Witness complex. 

### Returns
- `C_aux`: (dict) Eirene output of auxiliary filtration
- `Zpsi_index2simplex`: (dict) indexing of simplices in Zpsi
"""
function build_auxiliary_filtration(
        C_Z::Dict{String, Any}, 
        psi::Float64, 
        D_Y::Array{Float64, 2}; 
        format = "VR to VR", 
        dim = 1,
        W_index2simplex::Dict=Dict{Any, Any}(),
        W_fv::Array{Any, 1}=Array{Any, 1}(undef, 0))
    
    if format == "VR to VR"
        n_simplices, Zpsi_index2simplex, Zpsi_simplex2index = index_simplices_Zpsi(C_Z, psi)
        
        rv, cp, ev, fv = create_CSC_input_from_distance(n_simplices, Zpsi_index2simplex, Zpsi_simplex2index, D_Y)
        # run eirene
        C_aux = eirene(rv = rv, cp = cp, ev = ev, fv = fv, record = "all", maxdim = dim)
       
    elseif format == "VR to W"
        # create Z^{ψ}
        n_simplices, Zpsi_index2simplex, Zpsi_simplex2index = index_simplices_Zpsi(C_Z, psi)

        # Build the auxiliary filtration 
        # note D_Y here is the distance matrix D_P_Q or D_Q_P used to build the Witness complex
        rv, cp, ev, fv = create_CSC_input_from_W_distance(n_simplices, Zpsi_index2simplex, Zpsi_simplex2index, D_Y)

        # run Eirene
        C_aux = eirene(rv = rv, cp = cp, ev = ev, fv = fv, record = "all", maxdim = dim)  
   
    end
    
    return C_aux, Zpsi_index2simplex, Zpsi_simplex2index

end   

"""
    create_CSC_for_auxiliary_filtration_W_to_VR
Creates CSC inputs (rv, cp, ev, fv) for the auxiliary filtration ``W^{\\psi} \\cap VR^{\\bullet}``
"""
function create_CSC_for_auxiliary_filtration_W_to_VR(
    n_simplices,
    Wpsi_index2simplex,
    Wpsi_simplex2index,
    Wpsi_vertex_to_default_vertex,
    D_VR)
    
    ##### ev: number of simplices in each dimension
    maxdim = size(n_simplices,1)
    ev = n_simplices
    
    ##### boundary matrix CSC format: columns
    cp =  create_CP(ev)
    
    ##### boundary matrix CSC format: rows
    rv =  create_rv(ev, Wpsi_index2simplex, Wpsi_simplex2index)

    ##### fv: birth time of each simplex using D_X
    fv = []

    # 0-simplices have birth time of zero.
    fv_0 = zeros(ev[1])
    push!(fv, fv_0)

    # higher dimensional simplices
    for d = 1:maxdim-1
        fv_d = []
        for i = 1:ev[d+1]
            # find vertex representation of simplex using Wpsi indexing
            simplex = Wpsi_index2simplex[(i,d)]
            
            # find vertex representation using original vertex indexing (if needed)
            if Wpsi_vertex_to_default_vertex != nothing
                simplex = [Wpsi_vertex_to_default_vertex[v] for v in simplex]
            end
                
            # find its birth time according to D_Y
            push!(fv_d,  find_birth_time(simplex, D_VR))
        end

        push!(fv, fv_d)
    end
    return rv, cp, ev, fv
end

function build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR; dim = 1)
    
    # build W^{psi} and index the simplices
    Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_fv, Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex =  build_Witness_complex(D_W, psi)
    
    # convert simplices from vertex notation in Wpsi to default vertex (if needed)
    if Wpsi_vertex_to_default_vertex == nothing
        Wpsi_index2simplex_default_vertex = Wpsi_index2simplex
        Wpsi_default_vertex_simplex2index = Wpsi_simplex2index
    else
        Wpsi_index2simplex_default_vertex = Dict(key => [Wpsi_vertex_to_default_vertex[v] for v in value] for (key, value) in Wpsi_index2simplex)
        Wpsi_default_vertex_simplex2index = Dict(val => key for (key, val) in Wpsi_index2simplex_default_vertex)
    end
    
    # compute persistence
    n_simplices = [size(d, 1) for d in Wpsi_fv]
    rv, cp, ev, fv = create_CSC_for_auxiliary_filtration_W_to_VR(n_simplices, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_vertex_to_default_vertex, D_VR)
    C_aux = eirene(rv = rv, cp = cp, ev = ev, fv = fv, record = "all", maxdim = dim)   
    
    return C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index, Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex
end

"""
    find_epsilon0_Fbar_representation_tau()
Finds the parameter `epsilon0` and the F-bar representation of ``[\\tau_*]``. 
Recall the F-bar representation of ``[\\tau^{\\mathcal{B}}_*]``: Bars ``\\rho_1, \\dots, \\rho_m`` of `barcode(C_aux, dim)`
such that ``[\\tau_*] = [\\rho_1] + \\dots + [\\rho_m]`` in ``H_n(Z^{\\psi})``.
Performs steps 2(a)-2(c) of Algorithm 1 in paper.

### Arguments
- `cycle`
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `Z_psi_simplex2index`: (dict) of simplex and index of Z_psi.
                            output of function 'index_simplices_Zpsi'
- `dim`: (int) dimension of cycle

### Outputs
- `epsilon_0`: (float) `epsilon_0` from algorithm step 3
    `Fbar_representation_tau`: (arr) of integers, where each integer corresponds to interval of C_aux
                            in the intervals representing x.
                    
    ex) `Fbar_representation_tau = [1, 3, 5]`
                                1, 3, 5 refers to intervals 1, 3, 5 of `barcode(C_aux, dim)`.
                                Then `[x] = [b_1] + [b_3] + [b_5]` in ``H_n(Z^{\\psi})``.
"""
function find_epsilon0_Fbar_representation_tau( 
    cycle,  
    C_aux, 
    Zpsi_simplex2index; 
    dim::Int64=1)
    
    # express cycle [τ] in H_k(Z^ψ) as a chain (list of indices) using the indexing of the auxiliary filtration Z^{ψ} cap Y^{\bullet}
    τ_aux_chain = [Zpsi_simplex2index[item] for item in cycle]
        
    # Find bars of H_k(Z^{ψ} \cap Y^{\bullet}) that persist until the last filtration step
    barcode_aux = barcode(C_aux, dim = dim)
    candidates = findall(x -> x == Inf, barcode_aux[:,2])

    # find cycle representatives (as chains) of each candidate intervals
    candidates_rep = Dict(key => classrep(C_aux, class = key, dim = dim) for key in candidates)
    
    # find all possible linear combinations candidate bars
    candidate_comb = find_linear_combinations(candidates)

    # check if any of the combinations satisfy [τ_*] = [ρ_1] + ... + [ρ_m] in H_n(Z^ψ)
    Fbar_representation_tau = nothing

    # for each possible linear combination (1, ..., k)
    for comb in candidate_comb
        # express combination [ρ_1 + ... + ρ_m] as a chain 
        comb_chain = vcat([candidates_rep[item] for item in comb]...)
        
        # check if comb_chain and [x] are homologous
        homologous = check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
        if homologous == true
            Fbar_representation_tau = comb
            break
        end
    end

    # Error handling: return error if there are no F-bar representation of τ
    if Fbar_representation_tau == nothing
        throw(ErrorException("Error!!! Cannot find F-bar representations of τ"))
    end   

    # find epsilon_0
    birth = [barcode_aux[x,1] for x in Fbar_representation_tau]
    epsilon_0 = maximum(birth)

    return epsilon_0, Fbar_representation_tau
end

"""
    find_epsilon0_Fbar_representation_tau()
Finds the parameter `epsilon0` and the F-bar representation of ``[\\tau_*]``. 
Recall the F-bar representation of ``[\\tau^{\\mathcal{B}}_*]``: Bars ``\\rho_1, \\dots, \\rho_m`` of `barcode(C_aux, dim)`
such that ``[\\tau_*] = [\\rho_1] + \\dots + [\\rho_m]`` in ``H_n(Z^{\\psi})``.
Performs steps 2(a)-2(c) of Algorithm 1 in paper.

### Arguments
- `cycle`
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `Z_psi_simplex2index`: (dict) of simplex and index of Z_psi.
                            output of function 'index_simplices_Zpsi'
- `dim`: (int) dimension of cycle

### Outputs
- `epsilon_0`: (float) `epsilon_0` from algorithm step 3
    `Fbar_representation_tau`: (arr) of integers, where each integer corresponds to interval of C_aux
                            in the intervals representing x.
                    
    ex) `Fbar_representation_tau = [1, 3, 5]`
                                1, 3, 5 refers to intervals 1, 3, 5 of `barcode(C_aux, dim)`.
                                Then `[x] = [b_1] + [b_3] + [b_5]` in ``H_n(Z^{\\psi})``.
"""
function find_epsilon0_Fbar_representation_tau2( 
    cycle,  
    C_aux, 
    barcode_C_VR,
    Zpsi_simplex2index; 
    dim::Int64=1)
    
    # express cycle [τ] in H_k(Z^ψ) as a chain (list of indices) using the indexing of the auxiliary filtration Z^{ψ} cap Y^{\bullet}
    τ_aux_chain = [Zpsi_simplex2index[item] for item in cycle]
        
    # Find bars of H_k(Z^{ψ} \cap Y^{\bullet}) that persist until the last filtration step
    barcode_aux = barcode(C_aux, dim = dim)
    #candidates = findall(x -> x == Inf, barcode_aux[:,2]) OLD
    max_param = maximum(barcode_C_VR[:,2])
    candidates = [i for i =1:size(barcode_aux, 1) if (barcode_aux[i,2] == Inf)&(barcode_aux[i,1] < max_param)]

    # find cycle representatives (as chains) of each candidate intervals
    candidates_rep = Dict(key => classrep(C_aux, class = key, dim = dim) for key in candidates)
    
    # find all possible linear combinations candidate bars
    candidate_comb = find_linear_combinations(candidates)

    # check if any of the combinations satisfy [τ_*] = [ρ_1] + ... + [ρ_m] in H_n(Z^ψ)
    Fbar_representation_tau = nothing

    # for each possible linear combination (1, ..., k)
    for comb in candidate_comb
        # express combination [ρ_1 + ... + ρ_m] as a chain 
        comb_chain = vcat([candidates_rep[item] for item in comb]...)
        
        # check if comb_chain and [x] are homologous
        homologous = check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
        if homologous == true
            Fbar_representation_tau = comb
            break
        end
    end

    # Error handling: return error if there are no F-bar representation of τ
    if Fbar_representation_tau == nothing
        throw(ErrorException("Error!!! Cannot find F-bar representations of τ"))
    end   

    # find epsilon_0
    birth = [barcode_aux[x,1] for x in Fbar_representation_tau]
    epsilon_0 = maximum(birth)

    return epsilon_0, Fbar_representation_tau
end

function find_epsilon0_Fbar_representation_tau_WtoVR( 
    cycle,  
    C_aux, 
    Wpsi_default_vertex_simplex2index; 
    dim::Int64=1)
    
    # express cycle [τ] in H_k(W^psi) as a chain (list of indices) using the indexing of the auxiliary filtration W^psi cap VR^{\bullet}
    τ_aux_chain = [Wpsi_default_vertex_simplex2index[item][1] for item in cycle]
        
    # Find bars of H_k(W^psi \cap Y^{\bullet}) that persist until the last filtration step
    barcode_aux = barcode(C_aux, dim = dim)
    candidates = [i for i =1:size(barcode_aux, 1) if (barcode_aux[i,2] == Inf)]
    
     # find cycle representatives (as chains) of each candidate intervals
    candidates_rep = Dict(key => classrep(C_aux, class = key, dim = dim) for key in candidates)

    count = 0
    count_lim = 10000
    for i = 1:size(candidates,1)
        for comb in combinations(candidates,i)
            if count < count_lim
                # express combination [ρ_1 + ... + ρ_m] as a chain 
                comb_chain = vcat([candidates_rep[item] for item in comb]...)

                # check if comb_chain and [x] are homologous
                homologous =  check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
                count += 1

                if homologous == true
                    #print("count: ", count)
                    #print("\ncomb: ", comb)
                    Fbar_representation_tau = comb

                    # find epsilon_0
                    birth = [barcode_aux[x,1] for x in Fbar_representation_tau]
                    epsilon_0 = maximum(birth)

                    return epsilon_0, Fbar_representation_tau

                end
            else
                
                # Error handling: return error if there are no F-bar representation of τ (or run out of trials)
                print("Error!!! Cannot find F-bar representations of τ with limited iterations " * string(count_lim))
                return nothing, nothing

    
                
            end
        end

    end

    
    """
    # find all possible linear combinations candidate bars
    candidate_comb =  find_linear_combinations(candidates)

    # check if any of the combinations satisfy [τ_*] = [ρ_1] + ... + [ρ_m] in H_n(Z^ψ)
    Fbar_representation_tau = nothing

    # for each possible linear combination (1, ..., k)
    for comb in candidate_comb
        print(comb)
        # express combination [ρ_1 + ... + ρ_m] as a chain 
        comb_chain = vcat([candidates_rep[item] for item in comb]...)
        
        # check if comb_chain and [x] are homologous
        homologous =  check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
        if homologous == true
            Fbar_representation_tau = comb
            break
        end
    end
    """
    # Error handling: return error if there are no F-bar representation of τ
    print("Error!!! Cannot find F-bar representations of τ")
    return nothing, nothing
   
    
    
end

function find_epsilon0_Fbar_representation_tau_WtoVR2( 
    cycle,  
    C_aux, 
    barcode_VR,
    Wpsi_default_vertex_simplex2index; 
    dim::Int64=1)
    
    # express cycle [τ] in H_k(W^psi) as a chain (list of indices) using the indexing of the auxiliary filtration W^psi cap VR^{\bullet}
    τ_aux_chain = [Wpsi_default_vertex_simplex2index[item][1] for item in cycle]
        
    # Find bars of H_k(W^psi \cap Y^{\bullet}) that persist until the last filtration step
    barcode_aux = barcode(C_aux, dim = dim)
    max_param = maximum(barcode_VR[:,2])
    candidates = [i for i =1:size(barcode_aux, 1) if (barcode_aux[i,2] == Inf)&(barcode_aux[i,1] < max_param)] ### DIFFERS FROM `find_epsilon0_Fbar_representation_tau_WtoVR`

    # find cycle representatives (as chains) of each candidate intervals
    candidates_rep = Dict(key => classrep(C_aux, class = key, dim = dim) for key in candidates)
    
    # find all possible linear combinations candidate bars
    candidate_comb =  find_linear_combinations(candidates)

    # check if any of the combinations satisfy [τ_*] = [ρ_1] + ... + [ρ_m] in H_n(Z^ψ)
    Fbar_representation_tau = nothing

    # for each possible linear combination (1, ..., k)
    for comb in candidate_comb
        # express combination [ρ_1 + ... + ρ_m] as a chain 
        comb_chain = vcat([candidates_rep[item] for item in comb]...)
        
        # check if comb_chain and [x] are homologous
        homologous =  check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
        if homologous == true
            Fbar_representation_tau = comb
            break
        end
    end

    # Error handling: return error if there are no F-bar representation of τ
    if Fbar_representation_tau == nothing
        throw(ErrorException("Error!!! Cannot find F-bar representations of τ"))
    end   

    # find epsilon_0
    birth = [barcode_aux[x,1] for x in Fbar_representation_tau]
    epsilon_0 = maximum(birth)

    return epsilon_0, Fbar_representation_tau
end

function find_epsilon0_Fbar_representation_tau_WtoVR_large_limit( 
    cycle,  
    C_aux, 
    Wpsi_default_vertex_simplex2index; 
    dim::Int64=1)
    
    # express cycle [τ] in H_k(W^psi) as a chain (list of indices) using the indexing of the auxiliary filtration W^psi cap VR^{\bullet}
    τ_aux_chain = [Wpsi_default_vertex_simplex2index[item][1] for item in cycle]
        
    # Find bars of H_k(W^psi \cap Y^{\bullet}) that persist until the last filtration step
    barcode_aux = barcode(C_aux, dim = dim)
    candidates = [i for i =1:size(barcode_aux, 1) if (barcode_aux[i,2] == Inf)]
    
     # find cycle representatives (as chains) of each candidate intervals
    candidates_rep = Dict(key => classrep(C_aux, class = key, dim = dim) for key in candidates)

    count = 0
    count_lim = 1000000
    for i = 1:size(candidates,1)
        for comb in combinations(candidates,i)
            if count < count_lim
                # express combination [ρ_1 + ... + ρ_m] as a chain 
                comb_chain = vcat([candidates_rep[item] for item in comb]...)

                # check if comb_chain and [x] are homologous
                homologous =  analogous_bars.check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
                count += 1

                if homologous == true
                    #print("count: ", count)
                    #print("\ncomb: ", comb)
                    Fbar_representation_tau = comb

                    # find epsilon_0
                    birth = [barcode_aux[x,1] for x in Fbar_representation_tau]
                    epsilon_0 = maximum(birth)

                    return epsilon_0, Fbar_representation_tau

                end
            else
                
                # Error handling: return error if there are no F-bar representation of τ
                print("Error!!! Cannot find F-bar representations of τ with limited iterations " * string(count_lim))
                return nothing, nothing  
            end
        end

    end

    
    """
    # find all possible linear combinations candidate bars
    candidate_comb =  find_linear_combinations(candidates)

    # check if any of the combinations satisfy [τ_*] = [ρ_1] + ... + [ρ_m] in H_n(Z^ψ)
    Fbar_representation_tau = nothing

    # for each possible linear combination (1, ..., k)
    for comb in candidate_comb
        print(comb)
        # express combination [ρ_1 + ... + ρ_m] as a chain 
        comb_chain = vcat([candidates_rep[item] for item in comb]...)
        
        # check if comb_chain and [x] are homologous
        homologous =  check_homologous_cycles(comb_chain, τ_aux_chain, C_aux)
        if homologous == true
            Fbar_representation_tau = comb
            break
        end
    end
    """
    # Error handling: return error if there are no F-bar representation of τ
    print("Error!!! Cannot find F-bar representations of τ")
    return nothing, nothing
   
    
    
end


# find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
function find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex; dim = 1)
    
    aux_cycles = Dict()
    
    # find cyclerep of Fbar_representation_tau
    Fbar_rep_chain = vcat(classrep(C_aux, dim = dim, class = Fbar_representation_tau)...)    
    Fbar_rep_v = [Wpsi_index2simplex_default_vertex[(item, dim)] for item in Fbar_rep_chain]
    aux_cycles["Fbar_rep_tau"] = Fbar_rep_v
    
    # find cyclerep of bars in BARS_short
    for bar in BARS_short
        cyclerep_chain = classrep(C_aux, dim = dim, class = bar)  
        cyclerep_v = [Wpsi_index2simplex_default_vertex[(item, dim)] for item in cyclerep_chain]
        aux_cycles[bar] = cyclerep_v
    end
    
    return aux_cycles
end


"""
    find_pY_and_BARSshort
Given parameter epsilon_0, find the collection `pY` and BARSshort 

### Arguments
- `C_aux`: (dict) output of Eirene
- `epsilon_0`: (float) epsilon_0 parameter found
- `max_pY`: maximum parameter value to consider. defaults to Inf. 
- `dim`: dimension. defaults to 1

### Returns
- `pY`: collection of parameters pY
- `BARSshort`: collection of bars in the auxiliary filtration that exist at, or after, epsilon_0 that dies before the final parameter
"""
function find_pY_and_BARSshort(
    C_aux::Dict, 
    epsilon_0::Float64; 
    max_pY = Inf,
    dim::Int64 = 1)
    
    birth_times = barcode(C_aux, dim = dim)[:,1]
    death_times = barcode(C_aux, dim = dim)[:,2]

    # note x < Inf so that we exclude the long bars
    BARSshort_d = findall(x -> epsilon_0 <= x < Inf, death_times)
    BARSshort_b = findall(x -> x <= max_pY, birth_times)
    BARSshort = collect(intersect(Set(BARSshort_b), Set(BARSshort_d)))
    pY = sort(append!([epsilon_0],birth_times[BARSshort]))
    pY = [item for item in pY if item >= epsilon_0]

    return pY, BARSshort
end


"""
    find cyclereps_auxiliary
Find the cycle representatives of `Fbar_representation_tau` and `BARS_short` from the auxiliary filtration, using the vertex indexing
"""
function find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex; dim = 1)
    
    aux_cycles = Dict()
    
    # find cyclerep of Fbar_representation_tau
    Fbar_rep_chain = vcat(classrep(C_aux, dim = dim, class = Fbar_representation_tau)...)    
    Fbar_rep_v = [Zpsi_index2simplex[(item, dim)] for item in Fbar_rep_chain]
    aux_cycles["Fbar_rep_tau"] = Fbar_rep_v
    
    # find cyclerep of bars in BARS_short
    for bar in BARS_short
        cyclerep_chain = classrep(C_aux, dim = dim, class = bar)  
        cyclerep_v = [Zpsi_index2simplex[(item, dim)] for item in cyclerep_chain]
        aux_cycles[bar] = cyclerep_v
    end
    
    return aux_cycles
end

function simplex_to_index(simplex, C)
    # Given a n-simplex [u_0, u_1, ... , u_n], where each u_i is Eirene-indexed,
    # find the index of simplex in C_n according to eirene output C.
    
    """
    --- input ---
    simplex (arr) : [u_0, u_1, ... , u_k], where each u_i is indexed 
                according to Eirene.
                u_0 < u_1 < ... < u_k.
    C (dict): output from Eirene
    --- output ---
    index (int): index of given simplex in C_n according to C.

    --- example use ---
    # 1. get permutation info
    v_perm = get_vertex_perm(C)
    # 2. rewrite simplex in terms of 0-simplices (indexed according to eirene)
    simplex = [96, 183, 188]
    simplex_eirene = sort([v_perm[i] for i in simplex])
    # 3. find simplex index
    simplex_to_index(simplex_eirene, C)
    """ 
    dim = size(simplex,1) - 1

    # given simplex_eirene =[u_0, u_1, ... , u_n], find the portion of row-indices for column u0
    rowvals = Eirene_var.crows(C["firstv"][dim+1], C["farfaces"][dim+1], simplex[1])

    # find location of [u_1, ..., u_n] in rowvals.
    # that is, find the index of [u_1, ..., u_n] among C_{n-1}.
    
    # if simplex is 1-dimensional
    if dim == 1
        rowloc = findall(x->x==simplex[2], rowvals)[1]
        
    # if simplex is higher dimensional
    else
        rowloc0 = simplex_to_index(simplex[2:end], C)
        rowloc = findall(x->x==rowloc0, rowvals)[1]
    end

    # index of simplex according to Eirene
    index = C["firstv"][dim+1][simplex[1]] + rowloc - 1
    
    return index
end



function chain_to_index(
    chain, 
    C::Dict)
    # Given an n-dimensional chain expressed as a list of simplices 
    # chain = [simplex_1, simplex_2, ... , simplex_k],
    # where each simplex_j = [v_0, ... ,v_n], a list of its (n+1) vertices,
    # rewrite the chain as a list of integers [i_1, ..., i_k], 
    # where each $i_j$ is the index of simplex_j in C_n according to Eirene. 

    """
    --- input ---
    chain (arr) : [s1, s2, ... , sn], 
                where each s_j is a n-simplex of form [v_0, v_1, ..., v_n],
                a list of its (n+1) vertices
    C (dict): output from Eirene

    --- output ---
    chain_idx (arr): [i_1, ... ,i_n], where each i_j is the index of s_j according to C

    --- example ---
    test_chain =[[96, 183, 188],[99, 111, 188]]
    chain_idx = chain_to_index(test_chain, C)
    print(chain_idx)

    # check
    dim = size(test_chain[1],1)-1
    for item in chain_idx
        simplex = EireneVar.incidentverts(C["farfaces"], C["firstv"], dim+1, [item])
        simplex = sort(C["nvl2ovl"][simplex])
        print(simplex)
    end
    """
    # get permutation of vertices
    v_perm = get_vertex_perm(C)

    chain_idx = []
    for simplex in chain
        simplex_eirene = sort([v_perm[i] for i in simplex])
        try 
            simplex_idx = simplex_to_index(simplex_eirene, C)
            append!(chain_idx, simplex_idx)
        catch
            print("chain doesn't exist in eirene output")
            print(simplex)
            return nothing
        end
    end

    return chain_idx
end


# ----------------------------------------------------------------------------------------------------
# functions for finding Y-bar-representations
# ----------------------------------------------------------------------------------------------------

"""
    prep_bar_representation()
Prepare for finding bar-representations of a cycle at given parameter ``\\epsilon``. Builds the complex at ``Y^{\\epsilon}`` 
and returns the indices of bars of barcode(Y) that are present at parameter ``\\epsilon``.

### Arguments
- `C_Y`: (dict) Eirene output of complex Y
- `D_Y`: (array) distance matrix on Y
- `epsilon`: (float) parameter
- `dim`: (float) dimension

### Returns
- `C_Y_epsilon`: (dict) output of Eirene on `D_Y` with `maxrad = epsilon`
- `bars_Y_cyclerep`: (dict)
    keys: indices of `barcode(C_Y, dim = dim)` that exist at parameter epsilon
    vals: cycle representatives of bars of `barcode(C_Y, dim = dim)` expressed as a list of indices in `C_Y_epsilon`
"""
function prep_bar_representation(
    C_Y::Dict, 
    D_Y::Array,
    epsilon::Float64; 
    dim::Int64=1)

    # find bars of barcode(Y) that exist at parameter "epsilon"
    barcode_Y = barcode(C_Y, dim = dim)
    bars_Y = [i for i = 1:size(barcode_Y,1) if (barcode_Y[i,1] <= epsilon) & (barcode_Y[i,2] > epsilon)]

    # build the complex "Y_epsilon" by running Eirene on D_Y up to epsilon 
    C_Y_epsilon = eirene(D_Y, maxrad = epsilon, record = "all");

    # express each bar as a chain (list of indices) in C_Y_epsilon.
    # note: indexing is according to C_Y_epsilon
    bars_Y_cyclerep = Dict()
    for i in bars_Y
        rep = classrep(C_Y, dim = dim, class = i, format = "vertex x simplex") 
        rep_v = [rep[:,j] for j =1:size(rep,2)]
        bars_Y_cyclerep[i] = chain_to_index(rep_v, C_Y_epsilon)
    end
    return C_Y_epsilon, bars_Y_cyclerep
end


"""
    prep_bar_representation_W(W, D, epsilon; <keyword arguments>)
Prepare for finding bar-representations of a cycle at given parameter ``\\epsilon``. Builds the complex at ``W^{\\epsilon}`` and returns the indices of bars of barcode(W) that are present at parameter ``\\epsilon``.

### Arguments
- `W` (dict): Summary of Witness filtration information. Output of running `compute_Witness_persistence(D, maxdim = dim)`. 
- `epsilon` (float): Parameter to build W(epsilon)
- `dim` (int): dimension

### Outputs
- `bars_W_cyclerep`: (dict)
    keys: indices of bars in `barcode(C_W, dim = dim)` that are present at parameter `epsilon`
    vals: cycle representatives of the bars in `barcode(C_W, dim = dim)`,  expressed as a list of indices in `C_W_epsilon`.
- `W_epsilon`: (dict) summarizing the Witness complex W_epsilon
"""
function prep_bar_representation_W(
    W::Dict,
    epsilon::Float64; 
    dim::Int64=1)
    
    C_W = W["eirene_output"]
    D = W["distance_matrix"]

    # find bars of barcode(W) that exist at parameter "epsilon"
    barcode_W = barcode(C_W, dim = dim)
    bars_W = [i for i = 1:size(barcode_W,1) if (barcode_W[i,1] <= epsilon) & (barcode_W[i,2] > epsilon)]

    # build the complex "W_epsilon" and run persistence
    W_epsilon = compute_Witness_persistence(D, maxdim = dim, param_max = epsilon)
    
    # express each bar of barcode(C_W, dim) as a chain (list of indices) in C_W_epsilon (using indexing of C_Wepsilon)
    bars_W_cyclerep = Dict()
    Wepsilon_simplex2index = W_epsilon["simplex2index"]
    default_vertex_to_Wepsilon_vertex = W_epsilon["default_vertex_to_W_vertex"]
    
    for i in bars_W
        
        cr = classrep(C_W, dim = dim, class = i)
        
        # express classrep using vertex representation
        cr_vertex = [W["index2simplex"][(item, dim)] for item in cr]
        
        if default_vertex_to_Wepsilon_vertex == nothing
            cr_vertex_Wepsilon = cr_vertex
        else
            # convert vertex index from default to vertex indexing in C_Wepsilon
            cr_vertex_Wepsilon = [[default_vertex_to_Wepsilon_vertex[v] for v in simplex] for simplex in cr_vertex] 
        end
        
        # express classrep using indexing of W_epsilon
        cr_idx = [Wepsilon_simplex2index[item] for item in cr_vertex_Wepsilon]
        bars_W_cyclerep[i] = cr_idx
    end

    return bars_W_cyclerep, W_epsilon
end

"""
    find_bar_representation
Given a cycle, find its bar-representation in ``Y^{\\epsilon}``

### Arguments
- `cycle`: cycle representative given in the format of [simplex1, simplex2, ... ]
    ex) cycle = [[v^1_1, v^1_2], [v^2_1, v^2_2], ... ,]
- `C_Y`: (dict) Eirene output of complex Y
- `D_Y`: (array) Distance matrix on Y
- `epsilon`: (float) selected paramter to build ``Y^{\\epsilon}``
- `dim`: (int) dimension

### Returns
- nothing if cycle is trivial in ``Y^{\\epsilon}``. 
    Otherwise, return the collection of bars of Y that represent the given cycle.
    That is, if `S_{[cycle]} = {gamma_1, ... , gamma_n}`, then return `[1, ..., n]`
"""
function find_bar_representation(
    cycle,
    C_Y,
    D_Y,
    epsilon;
    dim = 1)

    # preparation
    C_Y_epsilon, bars_Y_cyclerep = prep_bar_representation(C_Y, D_Y, epsilon, dim = dim)
    
    # express given cycle as a list of indices in C_Y_epsilon, using the indexing of C_Y_epsilon
    cycle_index = chain_to_index(cycle, C_Y_epsilon)
    
    ### check if cycle is trivial
    homologous = check_homologous_cycles([], cycle_index, C_Y_epsilon)
    if homologous == true
        # cycle is trivial
        return nothing
    end
    
    ### If cycle is nontrivial
    # Find linear combinations of candidate bars in barcode(Y)
    bars_Y = collect(keys(bars_Y_cyclerep))
    n_bars = size(bars_Y,1)

    # For each linear combination of bars_Y, check if the cycle rep of the combination is homologous given cycle in C_Y_epsilon
    representation = nothing 
    i = 1 
    while (representation == nothing) & (i <= n_bars) 
        # find combinations of "i" number of items 
        combination = collect(combinations(bars_Y,i))
    
        for comb in combination
            comb_chain = vcat([bars_Y_cyclerep[item] for item in comb]...)
            comb_chain = select_odd_count(comb_chain)
            homologous = check_homologous_cycles(comb_chain, cycle_index, C_Y_epsilon)
            if homologous == true
                representation = comb
                return representation
            end
        end
        i += 1
    end
    return representation
    
end
"""
    find_bar_representation_W
Given a cycle, find its bar-representation in ``W^{\\epsilon}``

### Arguments
- `cycle`: cycle representative given in the format of [simplex1, simplex2, ... ], 
    where each simplex is a list of vertices.
    ex) cycle = [[v^1_1, v^1_2], [v^2_1, v^2_2], ... ,]
- `W`: (dict) Summary of Witness filtration information. Output of running `compute_Witness_persistence(D, maxdim = dim)`. 
- `epsilon`: (float) selected paramter to build ``W^{\\epsilon}``
- `dim`: (int) dimension

### Returns
- nothing if cycle is trivial in ``W^{\\epsilon}``. 
    Otherwise, return the collection of bars of barcode(W) that represent the given cycle.
    That is, if S_{[cycle]} = {gamma_1, ... , gamma_n}, then return [1, ..., n]
"""
function find_bar_representation_W(
    cycle,
    W,
    epsilon;
    dim = 1)

    bars_W_cyclerep, W_epsilon = prep_bar_representation_W(W, epsilon, dim = dim)
    C_Wepsilon = W_epsilon["eirene_output"]
    Wepsilon_index2simplex = W_epsilon["index2simplex"]
    Wepsilon_simplex2index = W_epsilon["simplex2index"]
    default_vertex_to_W_vertex = W_epsilon["default_vertex_to_W_vertex"]

    if default_vertex_to_W_vertex != nothing
        cycle = [[default_vertex_to_W_vertex[item] for item in simplex] for simplex in cycle]
    end
    # express given cycle as a list of indices in C_Wepsilon, using the indexing of C_Wepsilon
    cycle_idx = [Wepsilon_simplex2index[item] for item in cycle] 
  
    ### check if cycle is trivial
    homologous = check_homologous_cycles([], cycle_idx, C_Wepsilon)
    if homologous == true
        # cycle is trivial
        return nothing
    end
    
    ### If cycle is nontrivial, find possible combinations of candidate bars in barcode(W)
    bars_W = collect(keys(bars_W_cyclerep))
    n_bars = size(bars_W,1)

    # For each combination of bars_W, check if the cycle rep is homologous to given cycle in C_Wepsilon
    representation = nothing 
    i = 1 
    while (representation == nothing) & (i <= n_bars) 
        # find combinations of "i" number of items 
        combination = collect(combinations(bars_W,i))
    
        for comb in combination
            comb_chain = vcat([bars_W_cyclerep[item] for item in comb]...)
            comb_chain = select_odd_count(comb_chain)
            homologous = check_homologous_cycles(comb_chain, cycle_idx, C_Wepsilon)
            if homologous == true
                representation = comb
                return representation
            end
        end
        i += 1
    end
    return representation   
end

"""
    find_bar_representation_given_CYepsilon

Finds bar representations of a given cycle. Similar to the function `find_bar_representation`, except the input takes `C_Y_epsilon` and `bars_Y_cyclerep` explicitly. This function is useful when we have multiple cycles to examine for the same ``Y^{\\epsilon}``.

### Arguments
- `cycle`: cycle representative given in the format of [simplex1, simplex2, ... ], 
    where each simplex is a list of vertices.
    ex) `cycle = [[v^1_1, v^1_2], [v^2_1, v^2_2], ... ,]`
- `C_Y_epsilon`: (dict) Eirene output of complex Y with maxrad = epsilon for some epsilon.
    First output of function `prep_bar_representation`
- `bars_Y_cyclerep`: (dict) bars that exist in barcode(Y) at some parameter epsilon and the cycle representatives
    Second output of function `prep_bar_representation`

### Returns
- `representation`:
    - nothing if cycle is trivial in ``Y^{\\epsilon}``
    - otherwise, return the collection of bars of Y that represent the given cycle.
    That is, if `S_{[cycle]} = {gamma_1, ... , gamma_n}`, then return [1, ..., n]
"""
function find_bar_representation_given_CYepsilon(
    cycle,
    C_Y_epsilon,
    bars_Y_cyclerep;
    dim = 1)
    
    ### if `bars_Y_cyclerep` is empty, then C_Y_epsilon is trivial at epsilon
    if isempty(bars_Y_cyclerep) == true
        return nothing
    end
    
    
    # express given cycle as a list of indices in C_Y_epsilon, using the indexing of C_Y_epsilon
    cycle_index = chain_to_index(cycle, C_Y_epsilon)
    
    if (dim == 1) & (size(cycle_index,1) == 3)
       return nothing 
    end
    
    
    ### check if cycle is trivial
    homologous = check_homologous_cycles([], cycle_index, C_Y_epsilon)
    if homologous == true
        # cycle is trivial
        return nothing
    end
    
    ### If cycle is nontrivial
    # Find linear combinations of candidate bars in barcode(Y)
    bars_Y = collect(keys(bars_Y_cyclerep))
    n_bars = size(bars_Y,1)

    # For each linear combination of bars_Y, check if the cycle rep of the combination is homologous given cycle in C_Y_epsilon
    representation = nothing 
    i = 1 
    while (representation == nothing) & (i <= n_bars) 
        # find combinations of "i" number of items 
        combination = collect(combinations(bars_Y,i))
    
        for comb in combination
            comb_chain = vcat([bars_Y_cyclerep[item] for item in comb]...)
            comb_chain = select_odd_count(comb_chain)
            homologous = check_homologous_cycles(comb_chain, cycle_index, C_Y_epsilon)
            if homologous == true
                representation = comb
                return representation
            end
        end
        i += 1
    end
    return representation
    
end

"""
    find_bar_representation_given_CWepsilon

Finds bar representations of a given cycle in the Witness barcode. Similar to the function `find_bar_representation_W`, except the input takes C_Wepsilon and bars_W_cyclerep explicitly. This function is useful when we have multiple cycles to examine for the same ``W^{\\epsilon}``.

### Arguments
- `cycle`: cycle representative given in the format of [simplex1, simplex2, ... ], 
    where each simplex is a list of vertices.
    ex) `cycle = [[v^1_1, v^1_2], [v^2_1, v^2_2], ... ,]`
- `W_epsilon`: (dict) summarizing the complex W_epsilon 
- `bars_W_cyclerep`: (dict) bars that exist in barcode(W) at some parameter epsilon and the cycle representatives
    Second output of function `prep_bar_representation`

### Returns
- representation:
    - nothing if cycle is trivial in ``W^{\\epsilon}``
    - otherwise, return the collection of bars of W that represent the given cycle.
    That is, if `S_{[cycle]} = {gamma_1, ... , gamma_n}, then return [1, ..., n]`
"""
function find_bar_representation_given_CWepsilon(
    cycle,
    W_epsilon,
    bars_W_cyclerep)
    
    # get variables
    C_Wepsilon = W_epsilon["eirene_output"]
    Wepsilon_simplex2index = W_epsilon["simplex2index"]
    default_vertex_to_W_vertex = W_epsilon["default_vertex_to_W_vertex"]
    
    # express given cycle as a list of indices in C_Wepsilon, using the indexing of C_Wepsilon
    if default_vertex_to_W_vertex == nothing
            cr_vertex_Wepsilon = cycle
        else
            # convert vertex index from default to vertex indexing in C_Wepsilon
            cr_vertex_Wepsilon = [[default_vertex_to_W_vertex[v] for v in simplex] for simplex in cycle] 
        end
        
    # express classrep using indexing of W_epsilon
    cycle_index = [Wepsilon_simplex2index[item] for item in cr_vertex_Wepsilon]
    
    ### check if cycle is trivial
    homologous =  check_homologous_cycles([], cycle_index, C_Wepsilon)
    if homologous == true
        # cycle is trivial
        return nothing
    end
    
    ### If cycle is nontrivial, find combinations of candidate bars in barcode(W)
    bars_W = collect(keys(bars_W_cyclerep))
    n_bars = size(bars_W,1)

    # For each combination of bars_W, check if the cycle rep is homologous to the given cycle in C_Wepsilon
    representation = nothing 
    i = 1 
    while (representation == nothing) & (i <= n_bars) 
        # find combinations of "i" number of items 
        combination = collect(combinations(bars_W,i))
    
        for comb in combination
            comb_chain = vcat([bars_W_cyclerep[item] for item in comb]...)
            comb_chain =  select_odd_count(comb_chain)
            homologous =  check_homologous_cycles(comb_chain, cycle_index, C_Wepsilon)
            if homologous == true
                representation = comb
                return representation
            end
        end
        i += 1
    end
    return representation
    
end

"""
    find_bar_representations_at_epsilon0()
Find the Y-bar representations for (1) `Fbar_representation_tau` and (2) all short bars of the auxiliary filtration 
that are present at `epsilon_0`.

### Arguments 
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `epsilon_0`: (float) parameter. minimum of p_Y
- `cyclereps`: (dict) of cycle representatives of Fbar_representation_tau and short bars
- `C_Y`: (dict) output of Eirene
- `D_Y`: (array) distance matrix used for C_Y
- `dim`: (int) dimension

### Returns
- `Ybar_Fbar_tau`: Array representing the Y-bar representation of Fbar_representation_tau
- `bar_rep_short`: Dictionary representing the short bars of C_aux and their Y-bar representations
"""
function find_Ybar_representations_at_epsilon0(
    C_aux::Dict,
    epsilon_0::Float64,
    cyclereps,
    C_Y::Dict,
    D_Y::Array;
    dim::Int64=1)
    
    # Build C_Y_epsilon and find bars of barcode(Y) present at epsilon_0
    C_Y_epsilon, bars_Y_cyclerep = prep_bar_representation(C_Y, D_Y, epsilon_0, dim = dim)
    
    # find Y-bar representations of Fbar_representation_tau
    Fbar_rep_cycle = cyclereps["Fbar_rep_tau"]
    Ybar_Fbar_tau = find_bar_representation_given_CYepsilon(Fbar_rep_cycle, C_Y_epsilon, bars_Y_cyclerep)
    
    # find bar representations of short bars that exist at epsilon_0
    intervals_short = find_short_intervals_at_param(C_aux, epsilon_0, dim = dim)
    bar_rep_short = Dict()
    for i in intervals_short
        result = find_bar_representation_given_CYepsilon(cyclereps[i], C_Y_epsilon, bars_Y_cyclerep)
        if result != nothing
            bar_rep_short[i] = result
        end
    end
    
    return Ybar_Fbar_tau, bar_rep_short
    
end

"""
    find_bar_representations_at_epsilon0_W()
Find the W-bar representations for (1) `Fbar_representation_tau` and (2) all short bars of the auxiliary filtration 
that are present at `epsilon_0`.

### Arguments 
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `epsilon_0`: (float) parameter. minimum of `p_Y`
- `cyclereps`: (dict) of cycle representatives of `Fbar_representation_tau` and short bars
- `Zpsi_index2simplex`: (dict) of simplex and index of `C_aux`. Output of function `index_simplices_Zpsi`
- `W`: (dict) Summary of Witness filtration information. Output of running `compute_Witness_persistence(D, maxdim = dim)`. 
- `dim`: (int) dimension

### Returns
- `Ybar_Fbar_tau`: Array representing the Y-bar representation of `Fbar_representation_tau` in barcode(W)
- `bar_rep_short`: Dictionary representing the short bars of `C_aux` and their Y-bar representations
"""
function find_bar_representations_at_epsilon0_W(
    C_aux::Dict,
    epsilon_0::Float64,
    cyclereps, 
    Zpsi_index2simplex::Dict,
    W::Dict;
    dim::Int64=1)
    
    # Build C_W_epsilon and find bars of barcode(W) present at epsilon_0
    bars_W_cyclerep, W_epsilon = prep_bar_representation_W(W, epsilon_0, dim = dim)

    # find Y-bar representations of Fbar_representation_tau
    Fbar_rep_cycle = cyclereps["Fbar_rep_tau"]
    Ybar_Fbar_tau = find_bar_representation_given_CWepsilon(Fbar_rep_cycle, W_epsilon, bars_W_cyclerep)
    
    # find bar representations of short bars that exist at epsilon_0
    intervals_short = find_short_intervals_at_param(C_aux, epsilon_0, dim = dim)
    bar_rep_short = Dict()
    for i in intervals_short
        result = find_bar_representation_given_CWepsilon(cyclereps[i], W_epsilon, bars_W_cyclerep)
        if result != nothing
            bar_rep_short[i] = result
        end
    end
    
    return Ybar_Fbar_tau, bar_rep_short   
end

"""
    find_all_bar_representations
Find bar representations for all relevant bars in the auxiliary barcode `barcode(C_aux, dim = dim)`. 
In particular, find the Y-bar representations of
- the `Fbar_representation_tau` at `epsilon_0`
- the short bars present at `epsilon_0`
- the short bars that are born after epsilon_0 at their respective birth times. 

### Arguments
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `p_Y`: (array) of parameters
- `cyclereps`: (dict) of cycle representatives of `Fbar_representation_tau` and short bars
- `C_Y`: (dict) output of Eirene
- `D_Y`: (array) Distance matrix for Y
- `dim`: (int) dimension

### Returns
- `Ybar_rep_tau`: Array representing the Y-bar representation of `Fbar_representation_tau`
- `Ybar_rep_short_epsilon0`: Dictionary representing the short bars of `C_aux` (that are present at epsilon0) and their Y-bar representations at epsilon0
- `Ybar_rep_short`: Dictionary representing the remaining short bars of `C_aux` and their Y-bar representations at their birth times. 
"""
function find_all_bar_representations(
    C_aux::Dict,
    p_Y,
    cyclereps, 
    C_Y::Dict,
    D_Y::Array;
    dim::Int64=1)
    
    # find bar representation at epsilon_0
    epsilon_0 = minimum(p_Y)
    Ybar_rep_tau, Ybar_rep_short_epsilon0 = find_Ybar_representations_at_epsilon0(C_aux, epsilon_0, cyclereps, C_Y, D_Y, dim = dim)
    
    # for every other epsilon of p_Y, 
    # (1) find the bar rho of barcode(C_aux, dim = dim) that has epsilon as birth time, and
    # (2) find the Y-bar representation of rho
    
    Ybar_rep_short = Dict()
    for epsilon in p_Y[2:end]
        # find bar of barcode(C_aux) with birth time epsilon
        rho =  find_birth_intervals_at_param(C_aux, epsilon, dim = dim)
    
        for i in rho
        
            # find Y-bar representation of rho
            rho_bar_rep =  find_bar_representation(cyclereps[i], C_Y, D_Y, epsilon, dim = dim)
        
            if rho_bar_rep != nothing
                Ybar_rep_short[i] = rho_bar_rep
            end
        end
    end
    
    return Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short
    
end

"""
    find_all_bar_representations_W
Find bar representations for all relevant bars in the auxiliary barcode barcode(C_aux, dim = dim). 
In particular, find the Y-bar representations of
- the `Fbar_representation_tau` at `epsilon_0`
- the short bars present at `epsilon_0`
- the short bars that are born after epsilon_0 at their respective birth times. 

### Arguments
- `C_aux`: (dict) output of Eirene on the auxiliary filtration
- `p_Y`: (array) of parameters
- `cyclereps`: (dict) of cycle representatives of `Fbar_representation_tau` and short bars
- `Zpsi_index2simplex`: (dict) of simplex and index of `C_aux`. Output of function `index_simplices_Zpsi`
- `W`: (dict) Summary of Witness filtration information. Output of running `compute_Witness_persistence(D, maxdim = dim)`. 
- `dim`: (int) dimension

### Returns
- `Ybar_rep_tau`: Array representing the Y-bar representation of `Fbar_representation_tau`
- `Ybar_rep_short_epsilon0`: Dictionary representing the short bars of `C_aux` (that are present at epsilon0) and their Y-bar representations at epsilon0
- `Ybar_rep_short`: Dictionary representing the remaining short bars of `C_aux` and their Y-bar representations at their birth times. 
"""
function find_all_bar_representations_W(
    C_aux::Dict,
    p_Y,
    cyclereps, 
    Zpsi_index2simplex::Dict,
    W::Dict;
    dim::Int64=1)
    
    # find bar representation at epsilon_0
    epsilon_0 = minimum(p_Y)
    Ybar_rep_tau, Ybar_rep_short_epsilon0 = find_bar_representations_at_epsilon0_W(C_aux, epsilon_0, cyclereps, Zpsi_index2simplex, W, dim = dim)

    # for every other epsilon of p_Y, 
    # (1) find the bar rho of barcode(C_aux, dim = dim) that has epsilon as birth time, and
    # (2) find the Y-bar representation of rho
    
    Ybar_rep_short = Dict()
    for epsilon in p_Y[2:end]
        # find bar of barcode(C_aux) with birth time epsilon
        rho =  find_birth_intervals_at_param(C_aux, epsilon, dim = dim)
        
        for i in rho
        
            # find Y-bar representation of rho
            rho_bar_rep =  find_bar_representation_W(cyclereps[i], W, epsilon, dim = dim)
        
            if rho_bar_rep != nothing
                Ybar_rep_short[i] = rho_bar_rep
            end
        end
    end
    
    return Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short
    
end

#---------------------------------------------------------------
# functions for findind alternative bar extensions
# for different interval decomposition barcode(Y)
#---------------------------------------------------------------
"""
    find_possible_nonzero_row_locations(barcode, param, selected_intervals)
For each `c` in `selected_intervals`, find all coordinates `r` at which the entry `L_{r,c}` may be nonzero. 

### Arguments
- `barcode::Array`: barcode
- `param::Float`: selected parameter value 
- `selected_intervals::Array`

### Output
- `locations::Dictionary` 
    - `locations[c] = [r_1, ..., r_k]` indicates that `L_{r_i, c}` may be nonzero for any change of basis matrix `L`.
"""
function find_possible_nonzero_row_locations(barcode, param, selected_intervals)
    
    n = size(barcode, 1)
    locations = Dict()
    for c in selected_intervals
        birth_c, death_c = barcode[c,:]
        indices = []
        for r = 1:n
            birth_r, death_r = barcode[r,:]
            if (c != r) & (birth_r <= birth_c) & (death_r <= death_c) & (birth_c < death_r) & (birth_r <= param < death_r)
                append!(indices, r)
            end
        end
        locations[c] = indices
    end
    return locations
end

"""
    find_possible_column_vectors(nonzero_coordinates)
For each `c` in the key of `nonzero_coordinates`, find all possible column vector `L_{_,c}`  

### Arguments
- `nonzero_coordinates::Dictionary`: Output of function `find_possible_nonzero_row_locations`

### Output
- `possible_columns::Dictionary`
    - `possible_columns[c] = [[c], [c, i_1], ..., [c, i_1, ..., i_k]]`
    - For interval `c`, each element of of the value indicates a possible vector of L_{_, c}.
    - Each element of the value is an array indicating the coordinates at which the vector L_{_, c} is nonzero.
    - First possible column vector has 1 in coordinate `c` and 0 elsewhere.
    - Second possible column vector has a 1 in coordinates `c` and `i_1` and 0 elsewhere.
    - Last possible column vector has a 1 in coordinates `c`, ..., `i_k` and 0 elsewhere.
"""
function find_possible_column_vectors(nonzero_coordinates)
    possible_columns = Dict()
    for (key, val) in nonzero_coordinates
        comb_i = [[]]
        for n =1:size(val,1)
            append!(comb_i, collect(combinations(val, n))) 
        end
        possible_columns[key] = [append!(item, key) for item in comb_i]
    end
    return possible_columns
end


"""
    find_alternative_bar_extension(extension, param; <keyword arguments>)
Given a parameter and a bar extension, find all alternative bar extensions under a different basis of barcode(Y). Using notations in Algorithm 3 or paper:  Given a parameter ``\\ell \\in p_Y`` and a bar extension ``S^{\\mathcal{D}}_{[y]}`` at ``\\ell``, find all alternative bar extensions under a different interval decomposition of ``PH_k(Y^{\\bullet})``. That is, returns ``S^{\\mathcal{D} \\circ L^{-1}}_{[y]}`` for all possible ``L \\in L_Y``.

### Arguments
- `extension::Dictionary`: Output of `run_extension_method_VR` or `run_Extension_method_W_VR` or `analogous_intervals`
- `param::Float`: selected parameter
- `bar_extension::Array`: selected intervals to express using an alternative basis
- `dim::Integer=1`: Dimension. Defaults to 1 

### Outputs
- `alt_extension::Array`: Each intervals[i] is a collection of intervals that represent a specific interval extension via a different choice of basis of barcode. 
"""
function find_alternative_bar_extension(extension, param; bar_extension = [], dim = 1)
    # load barcode(Y)
    if extension["comparison"] == "VR to VR"   
        barcode_Y = barcode(extension["C_Y"], dim = dim)
    elseif extension["comparison"] == "W to VR"
        barcode_Y = barcode(extension["C_VR"], dim = dim)
    elseif extension["comparison"] == "VR to W"
        barcode_Y = barcode(extension["C_W"], dim = dim)
    end
    
    # For each bar 'c' in bar_extension, find all coordinates 'r' such that matrix entry L_{r,c} can be non-zero.
    nonzero_rows = find_possible_nonzero_row_locations(barcode_Y, param, bar_extension)

    # For each 'c' in bar_extension, generate all possible c-th column vector of matrix L
    possible_columns = find_possible_column_vectors(nonzero_rows)
    
    # Find all possible combinations of the column vectors. 
    # note: any such combination is automatically linearly-independent.
    possible_L = vcat(collect(Iterators.product([val for val in values(possible_columns)]...))...)
    
    # For each possible column vectors of L, find the alternative bar-extension
    # under interval decompostion D \circ L^{-1}
    alt_extension = [sort(select_odd_count(vcat(item...))) for item in possible_L]
    
    unique!(alt_extension) # since F2 coefficients

    return alt_extension

end


"""
    find_alt_BE(extension, BE)
Given the collection of all bar-extensions (at every parameter) for a fixed interval decomposition of ``PH_k(Y^{\\bullet})``,
find all possible alternative bar-extensions under a different interval decomposition. 

### Returns
- BE_all: (dict) corresponds to ``S(\\tau, Y^{\\bullet})`` in Algorithm 3 of paper.

"""
function find_alt_BE(extension, BE)
    BE_all = Dict()
    p_Y = extension["nontrivial_pY"]
    
    for param in p_Y
        be_param = collect(values(BE[param]))

        # find all alternative bar extensions
        BE_alt_param = []
        for bar_extension in be_param
            alt_ext = find_alternative_bar_extension(extension, param, bar_extension = bar_extension)
            append!(BE_alt_param, alt_ext)
        end
        unique!(BE_alt_param)
        BE_all[param] = BE_alt_param
    end
    return BE_all
end

"""
    find_BE_fixed_int_dec_at_param(extenison, param)
Finds all bar-extensions (under a fixed interval decomposition of ``PH_k(Y^{\\bullet})``) at a specific parameter.
"""
function find_BE_fixed_int_dec_at_param(extension, param)
    bar_extensions_at_param = []
    bar_ext = copy(extension["bar_extensions"][param]["baseline"]) #baseline bar-extension
    push!(bar_extensions_at_param, bar_ext)
    
    
    # find all possible combinations of offset bars
    offset_bar_comb = collect(combinations(unique(values(extension["bar_extensions"][param]["offset"]))))
    # find all possible bar extensions (baseline bar extension + offset bar extension) at given parameter
    for i=1:size(offset_bar_comb,1)
        bar_ext = copy(extension["bar_extensions"][param]["baseline"]) #baseline bar-extension
        append!(bar_ext, vcat(offset_bar_comb[i]...))
        bar_ext =  select_odd_count(bar_ext)
        sort!(bar_ext)
        # select only bars that appear ODD number of times
        push!(bar_extensions_at_param, bar_ext)
    end
    unique!(bar_extensions_at_param)
    return bar_extensions_at_param
end


"""
    find_alt_BE_at_param(extension, param)
Finds all bar-extension at a specific parameter.
Given a fixed ``\\ell``, finds ``\\{S^{\\mathcal{D} \\circ L^{-1}}_{[y]} | [y] \\in \\mathfrak{E}_{\\ell}, L \\in L_{Y}\\}``
"""
function find_alt_BE_at_param(extension, param)
   # find bar extensions under fixed interval decomposition D
    bar_extensions_at_param = find_BE_fixed_int_dec_at_param(extension, param) 
    
    # find all alternative bar extensions
    S_at_param = []
    for bar_ext in bar_extensions_at_param
        alt_at_param = find_alternative_bar_extension(extension, param, bar_extension = bar_ext)
        append!(S_at_param, alt_at_param)
    end
    unique!(S_at_param)
    
    return S_at_param
end



#----------------------------------------------------------------------------------------
# functions for finding different types of intervals at given parameter
#----------------------------------------------------------------------------------------

function find_short_intervals_at_param(
    C::Dict, 
    param::Float64; 
    dim::Int64= 1)
    # Find the "short" intervals of barcode(C, dim) that exist at given parameter.
    # The intervals must be 'short' intervals that die before reaching the last parameter
    C_barcode = barcode(C, dim = dim)
    candidates = []

    for i = 1:size(C_barcode)[1]
        if (C_barcode[i,1] <= param) & (C_barcode[i,2] > param) & (C_barcode[i,2] < Inf)
            push!(candidates, i)
        end
    end

    return candidates
end

function find_birth_intervals_at_param(
    C::Dict, 
    param::Float64; 
    dim::Int64 = 1)
    # Find intervals of barcode(C, dim) that are born at specific param
    """
    --- input ---
    C: (dict) output of Eirene
    param: (float) parameter in p_Y. 
    --- output ---
    (array) [x], where x is the index of interval with birth time epsilon
    """
    birth_times = barcode(C, dim = dim)[:,1]
    return findall(x -> x == param, birth_times)
end

#----------------------------------------------------------------------------------------
# other functions
#----------------------------------------------------------------------------------------

"""
    sample_torus(R, r, n)
Sample point cloud from a torus in 3-dimensions. The torus is the collection of points (x,y,z) satisfying the following
``x = (R + r \\cos \\theta) \\cos \\phi``
``y = (R + r \\cos \\theta) \\sin \\phi``
``z = r \\sin \\theta``

### Arguments
- R: (float)
- r: (float)
- n: (int) number of points to sample 

### Outputs
- X: (array) rows correspond to data, columns correspond to x,y,z-coordinates
"""
function sample_torus(R, r, n; return_angles = false)
    # sample angles
    theta = rand(n) * 2π
    phi = rand(n) * 2π

    x = (R .+ r .* cos.(theta) ).* cos.(phi)
    y = (R .+ r .* cos.(theta)) .* sin.(phi)
    z = r .* sin.(theta) 

    X = cat(x,y,z, dims = 2)
    
    if return_angles == false
        return X
    elseif return_angles == true
        return X, theta, phi
    end 
end

"""
    compute_distance(P, Q)
Compute the Euclidean distance matrices

### Arguments
- `P`: (array) of points in one region. Rows: points, columns: coordinates.
- `Q`: (array) of points in second region. Rows: points, columns: coordinates.

### Outputs
- `D_P`: (array) distance matrix among points in P
- `D_Q`: (array) distance matrix among points in Q
- `D_P_Q`: (array) distance matrix among points in P (row) and Q (column)
- `D_Q_P`: (array) distance matrix among points in Q (row) and P (column)
"""
function compute_distance(P, Q)
    
    # number of points in P
    n_P = size(P,1)
    
    # gather all points 
    X = vcat(P, Q)

    # compute distance
    D = Distances.pairwise(Euclidean(), X, X, dims = 1)

    # Define submatrices 
    D_P = D[1:n_P, 1:n_P]
    D_Q = D[n_P+1:end, n_P+1:end]
    D_P_Q = D[1:n_P, n_P+1:end]
    D_Q_P = D[n_P+1:end, 1:n_P]
    
    return D_P, D_Q, D_P_Q, D_Q_P
end

function angle_distance(phi, theta)
    s = min(phi, theta)
    l = max(phi, theta)
    distance = min(l-s, s + 2 * π - l)
    return distance
end

"""
    compute_distance_square_torus()
Given a list of theta values and phi values, computes the distance matrix on a square torus
"""
function compute_distance_square_torus(X_theta, X_phi)
    X = hcat(X_theta, X_phi)
    n = size(X)[1]
    d = zeros(n,n)
    for i=1:n
        for j=i+1:n
            d_phi = angle_distance(X[i,1], X[j,1])
            d_theta = angle_distance(X[i,2], X[j,2])
            distance = sqrt(d_phi^2 + d_theta^2)
            d[i,j] = distance
            d[j,i] = distance
        end
    end
    return d
end


#----------------------------------------------------------------------------------------
# functions for returning all possible homology classes corresponding to bars under different interval decompositions
#----------------------------------------------------------------------------------------
"""idx_to_vertex
Given a cycle expressed using chain indices, convert them to an expression using vertex representation.

Consider as inverse to function `chain_to_index`

To test: Make sure that the last output is the same as `test_vertex` (minus column-wise ordering)
test_idx = classrep(C, dim = 1, class = 4, format = "index")
test_vertex = classrep(C, dim = 1, class = 4)
idx_to_vertex(test_idx, 1, C)
"""
function idx_to_vertex(chain_idx, dim, C)
    simplex_list = Array{Int64}(undef, 2,0)
    for item in chain_idx
        simplex = Eirene_var.incidentverts(C["farfaces"], C["firstv"], dim+1, [item])
        simplex_list = hcat(simplex_list, sort(C["nvl2ovl"][simplex]))
    end
    return simplex_list
end


"""find_all_homology_classes
Find all possible homology classes corresponding to selected bars of a barcode. 
If selected_bars = [i,j], then find the homology classes that correspond to bar_i + bar_j

### Arguments
- `C`: (dict) Eirene output
- `selected_bars`: (array) of selected bars. ex) [1] or [1,2,3]
- `dim`: (int) dimension

### Returns
- `alternative_classes`: (dict) of alternative bar representations and their cycle representatives
"""
function find_all_homology_classes(C; selected_bars = [], dim = 1)
    barcode_C = barcode(C, dim = dim)
    
    # check that selected_bars are valid
    b = maximum(barcode_C[selected_bars,1])
    d = minimum(barcode_C[selected_bars,2])
    if d < b
       throw(error("There is no parameter at which the selected bars are simultaneously present.")) 
    end
    
    # select the smallest parameter at which all selected_bars are present
    param = maximum(barcode_C[selected_bars,1])
    
    # Given bar `c` (selected_bar), find all coordinates 'r' such that matrix entry L_{r,c} can be non-zero.
    nonzero_rows =  find_possible_nonzero_row_locations(barcode_C, param, selected_bars)

    # For each 'c' in bar_extension, generate all possible c-th column vector of matrix L
    possible_columns =  find_possible_column_vectors(nonzero_rows)

    # Find all possible combinations of the column vectors. 
    # note: any such combination is automatically linearly-independent.
    possible_L = vcat(collect(Iterators.product([val for val in values(possible_columns)]...))...)

    # For each possible column vectors of L, find the alternative bar-extension
    # under interval decompostion D \circ L^{-1}
    alt_extension = [sort( select_odd_count(vcat(item...))) for item in possible_L]

    unique!(alt_extension) # since F2 coefficients
    
    # for each alt_extension, find the homology class representatives
    alternative_classes = Dict()
    for bars in alt_extension
        combined_classrep = vcat([classrep(C, class = i, dim = dim, format = "index") for i in bars]...)
        combined_classrep =  select_odd_count(combined_classrep)
        combined_classrep_v = idx_to_vertex(combined_classrep, dim, C)
        alternative_classes[bars] = combined_classrep_v
    end

    return alternative_classes
end

"""
Find the minimum death parameter among bars1 in barcode1 and bars2 in barcode2.
"""
function find_min_death_param(barcode1, bars1, barcode2, bars2)
    min1 = minimum(barcode1[bars1,2])
    min2 = minimum(barcode2[bars2,2])

    min_total = minimum([min1, min2])

    return min_total
end


# ------------------------------------------------------------------------------------------------------------
# Additional functions written (not included in github of analogous bars method)
# ------------------------------------------------------------------------------------------------------------
function plot_PD(barcode; 
        highlight = [], highlight_color = :deeppink2, cutoff = nothing, 
        pd_min = nothing, pd_max = nothing, threshold_lw = 2, diagonal_lw = 2,
        kwargs...)
    points = barcode
    

    if size(barcode,1) == 0
        # plot diagonal line
        p = plot([0, 1], [0, 1], 
        labels ="", 
        linewidth = diagonal_lw,
        framestyle = :box,
        xlims = (0,1),
        ylims = (0,1),
        aspect_ratio = :equal,
        color = "grey"; 
        kwargs...)
        return p
    end
    # find index of points with death parameter == death
    idx = findall(x -> x == Inf, points[:,2])
    
    # plot points with death < Inf
    idx2 = [i for i in 1:size(points,1) if i ∉ idx]
    p = scatter(points[idx2,1], points[idx2,2]; kwargs..., color = "grey", labels = "", alpha = 0.5)
    
    # find max death value
    max_death = maximum(points[idx2, 2])
    
    # plot points with death parameter == Inf
    death_plot = ones(size(idx,1)) * max_death
    
    scatter!(p, points[idx,1], death_plot, aspect_ratio = :equal, legend=:bottomright, labels="", color ="red"; kwargs...)

    # plot diagonal line
    if pd_max == nothing
        
        min_birth = minimum(points[:,1]) * 0.8
        max_death = max_death * 1.1
        plot!(p, [min_birth, max_death], [min_birth, max_death], 
            labels ="", 
            linewidth = diagonal_lw,
            framestyle = :box,
            xlims = (min_birth, max_death),
            ylims = (min_birth, max_death),
            color = "grey"; 
            kwargs...)
    else
        max_death = pd_max
        min_birth = pd_min
        plot!(p, [min_birth, max_death], [min_birth, max_death], 
            labels ="", 
            linewidth = diagonal_lw,
            framestyle = :box,
            xlims = (min_birth, max_death),
            ylims = (min_birth, max_death),
            color = "grey"; 
            kwargs...)
    end
    
     # if highlight is provided, color specific points with the given color
    if highlight != []
         scatter!(points[highlight,1], points[highlight,2]; kwargs..., color = highlight_color, labels = "", #hover = highlight
        )
    end
    
    # plot the cutoff (dotted line) if provided
    if cutoff != nothing
        f(x) = x + cutoff
        plot!(f, linestyle = :dash, c = "black", label = "", #hover = false, 
            linewidth = threshold_lw)
    end
    
    return p
end

# function plot_PD_old(barcode; highlight = [], highlight_color = :deeppink2, cutoff = nothing, kwargs...)
#     points = barcode
    
#     # find index of points with death parameter == death
#     idx = findall(x -> x == Inf, points[:,2])
    
#     # plot points with death < Inf
#     idx2 = [i for i in 1:size(points,1) if i ∉ idx]
#     p = scatter(points[idx2,1], points[idx2,2]; 
#                 kwargs..., 
#                 color = "grey", 
#                 labels = "", 
#                 alpha = 0.5, 
#                 hover = idx2,
#                 framestyle = :box)

#     # find max death value
#     max_death = maximum(points[idx2, 2])
    
#     # plot points with death parameter == Inf
#     death_plot = ones(size(idx,1)) * max_death
#     min_vals = minimum(barcode[:,1]) * 0.9
#     max_vals = max_death * 1.1
#     scatter!(p, points[idx,1], death_plot, aspect_ratio = :equal, legend=:bottomright, labels="", color ="red", #hover = idx
#         ; kwargs...)

#     # if highlight is provided, color specific points with the given color
#     if highlight != []
#          scatter!(points[highlight,1], points[highlight,2]; kwargs..., color = highlight_color, labels = "", #hover = highlight
#         )
#     end
    
#     # plot diagonal line
#     g(x) = x
#     plot!(g, c= "black",
#         xlims = (min_vals, max_vals), ylims = (min_vals, max_vals), label = ""; kwargs..., #hover = false
#         )
    
#     # plot the cutoff (dotted line) if provided
#     if cutoff != nothing
#         f(x) = x + cutoff
#         plot!(f, linestyle = :dash, c = "black", label = "", #hover = false
#         )
#     end
    

#     return p
# end


"""
function return_min_parameter_with_boundary_matrix(D_VR, p_Y; dim = 1, skip_interval = 1, maxtry = 10)
    #Returns the minimum parameter (among p_Y) with a valid boundary matrix. 
    #Current version of Eirene (v1.3.6) has an issue where it doesn't return a valid boundary matrix. 
    #This function tests if epsilon_0 returns a valid boundary matrix. If not, it finds the minimum parameter among p_Y at which Eirene returns a valid boundary matrix.
    
    
    for p in p_Y[begin:skip_interval:end]
        
        # try computing the boundary matrix
        PH_param = eirene(D_VR, maxdim = dim, maxrad = p)
        
         try
            rv, cp = Eirene_var.boundarymatrix(PH_param, dim = dim)
            return p
        catch 
        end   
    end
    
    # if we iterated through all of p_Y, then keep increasing the parameter by 10%
    p = p_Y[end] * 1.1
    
    count = 0
    while count <= maxtry
        # try computing the boundary matrix
        PH_param = eirene(D_VR, maxdim = dim, maxrad = p)
        try
            rv, cp = Eirene_var.boundarymatrix(PH_param, dim = dim)
            return p
        catch 
        end   
        
        count += 1
        p = p * 1.1
    end 
end


function run_extension_VR_to_W_bar_part1(;
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_bar::Int64 = 0,
    W::Dict{Any, Any} = Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "psi"
    tau, psi =  find_terminal_class_in_VR(C_VR, D_VR, bar = VR_bar, dim = dim)
    
    ### run first half of extension of VR to W
    VR_cycle = tau
    C_W = W["eirene_output"]
    
    # build the auxiliary filtration
    C_aux, Zpsi_index2simplex, Zpsi_simplex2index =  build_auxiliary_filtration(C_VR, psi, D_W, format = "VR to W", dim = dim)

    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau =  find_epsilon0_Fbar_representation_tau(VR_cycle, C_aux, Zpsi_simplex2index, dim = dim)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_W, dim = dim)[:,2])
    p_Y, BARS_short =  find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using vertices
    cycle_reps =  find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex)
    
    return p_Y, cycle_reps, C_aux, Zpsi_index2simplex
    
end
"""

function run_extension_VR_to_W_bar_part2(epsilon_0, p_Y, C_aux, cycle_reps, W, Zpsi_index2simplex, dim)

    p_Y = vcat([epsilon_0], p_Y)
    C_W = W["eirene_output"]
    # find component-wise bar-representations
    Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short =  find_all_bar_representations_W(C_aux, p_Y, cycle_reps, Zpsi_index2simplex, W, dim = dim)
    
    # create summary (by parameter)
    nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions =  create_summary_by_parameter(cycle_reps, Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short, epsilon_0, C_aux, C_W)
    
    return nontrivial_pY, cycle_extensions, bar_extensions
end



function find_min_epsilon(D_Y, p_Y)
    
    for epsilon in p_Y
        # build the complex "Y_epsilon" by running Eirene on D_Y up to epsilon 
        C_Y_epsilon = eirene(D_Y, maxrad = epsilon, record = "all");
        try
            rv, cp = Eirene_var.boundarymatrix(C_Y_epsilon, dim = 2)
            return epsilon
        catch
        end
    end
end

function run_extension_W_to_VR_bar_at_epsilon0(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_bar::Int64 = 0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)

    # find terminal class rep "tau" and parameter "delta"
    tau, psi =  find_terminal_class_in_W(W, bar = W_bar, dim = dim)
    
    ### run first half of extension W to VR
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    W_cycle = tau
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex =  build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
 
    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau =  find_epsilon0_Fbar_representation_tau_WtoVR(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    
    p_Y, BARS_short =  find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim)
    
    #epsilon_0 = find_min_epsilon(D_VR, p_Y)
    
    # re-run the BARS_short
    _, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps =  find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)

    # find component-wise bar-representations
    Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short =  find_all_bar_representations(C_aux, 
                                                                                         [epsilon_0], 
                                                                                         cycle_reps, 
                                                                                         C_VR, 
                                                                                         D_VR, 
                                                                                         dim = dim)
    
    # create summary (by parameter)
    nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions =  create_summary_by_parameter(cycle_reps, 
                                                                                        Ybar_rep_tau, 
                                                                                        Ybar_rep_short_epsilon0, 
                                                                                        Ybar_rep_short, 
                                                                                        epsilon_0, 
                                                                                        C_aux,
                                                                                        C_VR)
    
    return nontrivial_pY, cycle_extensions, bar_extensions
end

function run_extension_W_to_VR_at_epsilon0(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    tau::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0, 
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    ### run first half of extension W to VR
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    W_cycle = tau
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex =  build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
 
    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau =  find_epsilon0_Fbar_representation_tau_WtoVR(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    p_Y, BARS_short =  find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim) 
                                            #max_pY = max_pY)

    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps =  find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)

    # find component-wise bar-representations
    Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short =  find_all_bar_representations(C_aux, 
                                                                                         [p_Y[1]], 
                                                                                         cycle_reps, 
                                                                                         C_VR, 
                                                                                         D_VR, 
                                                                                         dim = dim)
    
    # create summary (by parameter)
    nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions =  create_summary_by_parameter(cycle_reps, 
                                                                                        Ybar_rep_tau, 
                                                                                        Ybar_rep_short_epsilon0, 
                                                                                        Ybar_rep_short, 
                                                                                        epsilon_0, 
                                                                                        C_aux,
                                                                                        C_VR)
    
    return nontrivial_pY, cycle_extensions, bar_extensions
end


function run_extension_W_to_VR_at_epsilon0_var(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    tau::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0, 
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    ### run first half of extension W to VR
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    W_cycle = tau
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex =  build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
 
    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau =  find_epsilon0_Fbar_representation_tau_WtoVR(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    p_Y, BARS_short =  find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim) 
                                            #max_pY = max_pY)
    
    epsilon_0 = find_min_epsilon(D_VR, p_Y)
    # re-run the BARS_short
    _, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps =  find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)
    return C_aux, p_Y, BARS_short, cycle_reps
end


function run_baseline_extension_W_to_VR_at_epsilon0(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    tau::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0, 
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1,
    skip_interval = 1)
    
    ### run first half of extension W to VR
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    W_cycle = tau
    barcode_VR = barcode(C_VR, dim = dim)
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex =  analogous_bars.build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
 
    # find epsilon_0, Fbar_representation_tau
    #epsilon_0, Fbar_representation_tau =  analogous_bars.find_epsilon0_Fbar_representation_tau_WtoVR2(W_cycle, C_aux, barcode_VR, Wpsi_default_vertex_simplex2index)
    epsilon_0, Fbar_representation_tau =  analogous_bars.find_epsilon0_Fbar_representation_tau_WtoVR_large_limit(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)
    
     # This happens when we couldn't find Fbar representation of tau
    if epsilon_0 == nothing
        ### create output dictionary
        extension = Dict()
        # input data
        extension["comparison"] = "W to VR"
        extension["dim"] = dim 
        extension["selected_cycle"] = W_cycle 
        # auxiliary filtration & cyclerep
        #extension["C_auxiliary_filtration"] = C_aux
        extension["epsilon_0"] = epsilon_0
            
        # component-wise Ybar representations
        extension["baseline_bar_extension"] = nothing

        return extension
        
    end 
    
    # if epsilon_0 is too large, then no need to check for cycle extensions. Everything will be trivial
    if epsilon_0 > maximum(barcode_VR[:,2])
        
        ### create output dictionary
        extension = Dict()
        # input data
        extension["comparison"] = "W to VR"
        extension["C_VR"] = C_VR
        extension["C_W"] = C_W
        extension["dim"] = dim 
        extension["selected_cycle"] = W_cycle 
        # auxiliary filtration & cyclerep
        extension["C_auxiliary_filtration"] = C_aux
        #extension["aux_filt_cyclerep"] = cycle_reps
        # parameters pY
        #extension["p_Y"] = p_Y
        extension["epsilon_0"] = epsilon_0
            
        # component-wise Ybar representations
        extension["baseline_bar_extension"] = nothing

        return extension
        
    else
        # find p_Y and BARS_short
        #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
        p_Y, BARS_short =  analogous_bars.find_pY_and_BARSshort(C_aux, 
                                                epsilon_0, 
                                                dim = dim) 

        # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
        cycle_reps =  analogous_bars.find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)

        # check the minimum parameter at which one can find the boundary matrix from Eirene -- resolved with Eirene_var
        #epsilon0_new = analogous_bars.return_min_parameter_with_boundary_matrix(D_VR, p_Y; skip_interval = skip_interval)

        # Build C_Y_epsilon and find bars of barcode(Y) present at epsilon_0
        C_Y_epsilon, bars_Y_cyclerep =  analogous_bars.prep_bar_representation(C_VR, D_VR, epsilon_0, dim = dim)

        # find Y-bar representations of Fbar_representation_tau
        Fbar_rep_cycle = cycle_reps["Fbar_rep_tau"]
        try
            Ybar_Fbar_tau =  analogous_bars.find_bar_representation_given_CYepsilon(Fbar_rep_cycle, C_Y_epsilon, bars_Y_cyclerep; dim = dim)
            ### create output dictionary
            extension = Dict()
            # input data
            extension["comparison"] = "W to VR"
            extension["C_VR"] = C_VR
            extension["C_W"] = C_W
            extension["dim"] = dim 
            extension["selected_cycle"] = W_cycle 
            # auxiliary filtration & cyclerep
            extension["C_auxiliary_filtration"] = C_aux
            extension["aux_filt_cyclerep"] = cycle_reps
            # parameters pY
            extension["p_Y"] = p_Y
            extension["epsilon_0"] = epsilon_0
            #extension["min_epsilon_with_boundary"] = epsilon0_new
            # component-wise Ybar representations
            extension["baseline_bar_extension"] = Ybar_Fbar_tau

            return extension
        catch e
            print("Error in finding the boundary matrix")
            return nothing
        end

    end
end


"""
Given a barcode and a vector of bars, find the maximum death parameter
"""
find_max_death_param(barcode, bars) = maximum(barcode[bars,2])

function run_baseline_similarity_analogous(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bars = [],
    dim::Int64 = 1)
    
    ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bars == []
        throw(UndefKeywordError(:W_PQ_class))
    end
    
    analogous_bars_P = Dict()
    analogous_bars_Q = Dict()
    
    ### Compute the dual Witness persistence W(Q,P)
    D_P_Q = W_PQ["distance_matrix"]
    D_Q_P = collect(transpose(D_P_Q))

    # only compute the Witness filtration upto the largest death parameter of the selected bars
    # find maximum parameter
    barcode_W_PQ = barcode(W_PQ["eirene_output"], dim = 1)
    param_max = find_max_death_param(barcode_W_PQ, W_PQ_bars)
    W_QP = compute_Witness_persistence(D_Q_P, maxdim = 1, param_max = param_max)

    #W_QP = compute_Witness_persistence(D_Q_P, maxdim = dim)
    
    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = apply_Dowker_birth(W_PQ, W_QP, dim = dim)
    
    # for each bar in W_PQ_bars, run similarity-centric analogous bars
    for i in W_PQ_bars
        W_PQ_bar = i
        W_QP_bar = P_to_Q[W_PQ_bar]

        # find cycle, psi in W_P and W_Q
        cycle_W_P, psi_W_P =  analogous_bars.find_terminal_class_in_W(W_PQ, bar = W_PQ_bar)
        cycle_W_Q, psi_W_Q =  analogous_bars.find_terminal_class_in_W(W_QP, bar = W_QP_bar)

        # Run baseline extensions at epsilon 0
        extension_P = run_baseline_extension_W_to_VR_at_epsilon0(W = W_PQ, tau = cycle_W_P, psi = psi_W_P, C_VR = VR_P, D_VR = D_P)
        extension_Q = run_baseline_extension_W_to_VR_at_epsilon0(W = W_QP, tau = cycle_W_Q, psi = psi_W_Q, C_VR = VR_Q, D_VR = D_Q)
        
        analogous_bars_P[i] = extension_P
        analogous_bars_Q[i] = extension_Q
    end
    
    return analogous_bars_P, analogous_bars_Q
end

# original name: run_baseline_similarity_analogous_birthtime_fast
# variation on the "run_baseline_similarity_analogous_param", allowing the user to choose the parameter
# at which Witness complexes are built 
function run_baseline_similarity_analogous_param(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bar::Int64 = 0,
    W_param = Inf,
    dim::Int64 = 1)

     ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bar == 0
        throw(UndefKeywordError(:W_PQ_class))
    end



    ############################################################
    # 1. apply the extension method between W(P,Q) and VR(P)
    ############################################################

    # find the birth time of the selected bar in the Witness barcode
    W_PQ_barcode = barcode(W_PQ["eirene_output"], dim = dim)
    W_death_param = find_max_death_param(W_PQ_barcode, [W_PQ_bar])

    # if W_param isn't provided then use the default method (do extensions at the parameter immediately prior to death time)
    if W_param == Inf
        # find class rep "tau" of Witness filtration and the parameter immediately prior to the death
        cycle_W_P, psi_W_P =  analogous_bars.find_terminal_class_in_W(W_PQ, bar = W_PQ_bar)

        # Find extension of cycle_W_P to VR(P)
        extension_P = run_baseline_extension_W_to_VR_at_epsilon0(W = W_PQ, 
                                                                tau = cycle_W_P, 
                                                                psi = psi_W_P, 
                                                                C_VR = VR_P, 
                                                                D_VR = D_P)
    else
        # find class rep "tau" of Witness filtration
        cycle_W_P = analogous_bars.find_classrep_in_W(W_PQ, bar = W_PQ_bar, dim = dim)

        # Find extension of cycle_W_P to VR(P). Use the W_param as "psi" parameter
        extension_P = analogous_bars.run_baseline_extension_W_to_VR_at_epsilon0(W = W_PQ, 
                                                                                    tau = cycle_W_P, 
                                                                                    psi = W_param, 
                                                                                    C_VR = VR_P, 
                                                                                    D_VR = D_P)
    end

    println("extension to P complete.")

    ############################################################
    # 2. Compute the dual Witness complex
    ############################################################
    W_QP, P_to_Q = compute_Dowker_dual(W_PQ, param_max = W_death_param)

    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    W_QP_bar = P_to_Q[W_PQ_bar]

    println("computed dual Witness filtration.")
    
    ############################################################
    # 3. apply the extension method between W(Q,P) and VR(Q)
    ############################################################

    # if W_param isn't provided then use the default method (do extensions at the parameter immediately prior to death time)
    if W_param == Inf
        # find class rep "tau" of Witness filtration and the parameter immediately prior to the death
        cycle_W_Q, psi_W_Q =  analogous_bars.find_terminal_class_in_W(W_QP, bar = W_QP_bar)
        # Find extension of cycle_W_Q to VR(Q)
        extension_Q = run_baseline_extension_W_to_VR_at_epsilon0(W = W_QP, tau = cycle_W_Q, psi = psi_W_Q, C_VR = VR_Q, D_VR = D_Q)

    else
        # find class rep "tau" of Witness filtration
        cycle_W_Q = analogous_bars.find_classrep_in_W(W_QP, bar = W_QP_bar, dim = dim)
        
        # Find extension of cycle_W_Q to VR(Q). Use the W_param as "psi" parameter
        extension_Q = analogous_bars.run_baseline_extension_W_to_VR_at_epsilon0(W = W_QP, 
                                                                                tau = cycle_W_Q, 
                                                                                psi = W_param, 
                                                                                C_VR = VR_Q, 
                                                                                D_VR = D_Q)
    end

    println("extension to Q complete.")
    #extension_to_VR_Q = analogous_bars.run_extension_W_to_VR_bar_birthtime(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
    
    return extension_P, extension_Q
end


"""
Given the Witness filtraiton W_PQ, compute the dual Witness filtration W_QP upto a specified parameter.
Return W_QP and the dictionary encoding the correspondance between bars of W_PQ and W_QP
"""
function compute_Dowker_dual(W_PQ; param_max = Inf)
    D_PQ = W_PQ["distance_matrix"]
    D_QP = collect(transpose(D_PQ))

    W_QP = compute_Witness_persistence(D_QP, maxdim = 1, param_max = param_max)

    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = apply_Dowker_birth(W_PQ, W_QP, dim = 1)
    return W_QP, P_to_Q
end


"""
    run_full_similarity_analogous(; <keyword arguments>)
Similar to function `run_similarity_analogous`. Takes a collection of bars in W_PQ as input.
"""
function run_full_similarity_analogous(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
    W_PQ_bars = [],
    dim::Int64 = 1)
    
    ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if W_PQ == Dict()
        throw(UndefKeywordError(:W_PQ))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bars == []
        throw(UndefKeywordError(:W_PQ_class))
    end
    
    analogous_P = Dict()
    analogous_Q = Dict()
    
    ### Compute the dual Witness persistence W(Q,P)
    D_P_Q = W_PQ["distance_matrix"]
    D_Q_P = collect(transpose(D_P_Q))

    # only compute the Witness filtration upto the largest death parameter of the selected bars
    # find maximum parameter
    barcode_W_PQ = barcode(W_PQ["eirene_output"], dim = 1)
    param_max = find_max_death_param(barcode_W_PQ, W_PQ_bars)
    W_QP = compute_Witness_persistence(D_Q_P, maxdim = 1, param_max = param_max)

    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = apply_Dowker_birth(W_PQ, W_QP, dim = dim)
    
    # for each bar in W_PQ_bars, run similarity-centric analogous bars
    for i in W_PQ_bars
        W_PQ_bar = i
        W_QP_bar = P_to_Q[W_PQ_bar]
    
        ##### apply the extension method between W(P,Q) and VR(P) #####
        extension_P = run_extension_W_to_VR_bar(W = W_PQ, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)

        ##### apply the extension method between W(P,Q) and VR(P) #####
        extension_Q = run_extension_W_to_VR_bar(W = W_QP, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
        
        analogous_P[i] = extension_P
        analogous_Q[i] = extension_Q
    end
    return analogous_P, analogous_Q
end


"""
Check if a given cycle is trivial in the Witness complex at a particular parameter
"""
function check_trivial_cycle_W(W, param, cycle; dim = 1)
    
    _, W_epsilon = prep_bar_representation_W(W, param, dim = dim)
    
    C_Wepsilon = W_epsilon["eirene_output"]
    Wepsilon_simplex2index = W_epsilon["simplex2index"]
    default_vertex_to_W_vertex = W_epsilon["default_vertex_to_W_vertex"]
    
    if default_vertex_to_W_vertex != nothing
        cycle = [[default_vertex_to_W_vertex[item] for item in simplex] for simplex in cycle]
    end

    # express given cycle as a list of indices in C_Wepsilon, using the indexing of C_Wepsilon
    cycle_idx = [Wepsilon_simplex2index[item] for item in cycle] 

    ### check if cycle is trivial
    homologous = check_homologous_cycles([], cycle_idx, C_Wepsilon)
    return homologous
end


"""
Run the feature-centric analogous bars method using the baseline extensions. 
(1) From VR to W: baseline cycle extension
(2) From W to VR: baseline bar extension at ep0 (minimum parameter)

Given barcode(VR(Q)), find its bar extension in barcode(VR(P))
"""
function run_feature_centric_baseline_analogous_bars(;
        VR_Q::Dict{String, Any} = Dict{String, Any}(),
        D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
        VR_Q_bar::Int64 = 0,
        W_QP::Dict{Any, Any} = Dict{Any, Any}(),
        D_QP::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
        W_PQ::Dict{Any, Any} = Dict{Any, Any}(),
        D_PQ::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
        VR_P::Dict{String, Any} = Dict{String, Any}(),
        D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
        dim::Int64 = 1,
        skip_interval::Int64 = 1)
    
    ### 1. Run bar-to-cycle extension from VR(Q) to W(Q,P)
    p_Y, cycle_reps, C_aux, Zpsi_index2simplex = run_extension_VR_to_W_bar_part1(C_VR = VR_Q,
                                                                                D_VR = D_Q,
                                                                                VR_bar = VR_Q_bar,
                                                                                W = W_QP,
                                                                                D_W = D_QP,
                                                                                dim = dim)
    # find minimum parameter and baseline cycle extension in W(Q,P)
    param = p_Y[1]
    cycle_W_QP = cycle_reps["Fbar_rep_tau"]
    
    # check if cycle_W_QP is trivial
    trivial = check_trivial_cycle_W(W_QP, param, cycle_W_QP)
    if trivial == true
        print("Baseline cyclee extension from VR_Q to W_Q is trivial")
        return nothing
    else
    
        ### 2. Use Dowker's Theorem to find the corresponding cycle in W(P,Q)
        cycle_W_PQ = find_Dowker_cycle_correspondence(cycle_W_QP, param, D_QP)
        # find the death time of this cycle ( note for future development: can probably just look at death time of the bar rep of cycle_W_QP)
        parameter = find_cycle_death_in_Witness(cycle_W_PQ, W_PQ)
        psi = maximum(D_PQ[D_PQ.< parameter])

        ### 3. Run the cycle-to-bar extension from W(P,Q) to VR(P). Report only the baseline bar extension
        extension = run_baseline_extension_W_to_VR_at_epsilon0(W = W_PQ,
                                                                tau = cycle_W_PQ,
                                                                psi = psi,
                                                                C_VR = VR_P,
                                                                D_VR = D_P,
                                                                dim = dim,
                                                                skip_interval = skip_interval)

        return extension
    end

end

# ------------------------------------------------------------------------------
# Lazy extension
# ------------------------------------------------------------------------------
function permute_symmetric_array(D)
    # Permute a symmetric (distance) array, whose diagonals are 0. 
    # "D": distance array
    
    # Columns of relevant entries
    #row 1: 2, ... , n
    #row 2: 3, ... , n  
    #row 3: 4, ... , n
    #    ...
    # row n-1: n

    # which have the following indices:
    # 2, ... , n
    # n+3, ... , 2n
    # 2n + 4, ... , 3n, 
    # ...
    # (n-1)n

    n = size(D, 1)
    
    # extract above-diagonal elements in order
    elements = []
    for i=1:n
        row = D[i,i+1:end]
        append!(elements, row)
    end

    # Indices of above-diagonal elements
    indices = []
    for i =1:n-1
        row_indices = collect(n*(i-1)+(i+1):i*n)
        append!(indices, row_indices)
    end

    # permute indices 
    perm_indices = shuffle(indices)

    # array with permuted indices
    D_perm = zeros(n,n)
    setindex!(D_perm, elements, perm_indices)

    # turn D_perm into a symmetric matrix 
    D_perm = D_perm + transpose(D_perm);
    
    return D_perm
end



function get_barlength_distribution(D; alpha = 0.05, n = 10)
    # find empirical distribution of bar lengths 
    # (permute matrix D and compute persistençce) x 10   
    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1]  
        append!(length_sample, bar_lengths)
    end

    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    length_param = quantile(length_sample, 1-alpha)

    return length_param, length_sample

    # To plot histogram of sampled bar_lengths:
    #histogram(length_sample, label="", title="Histogram of bar lengths of shuffled distance", color = :grey)
    #vline!([length_param], label = "alpha = 0.05", linewidth = 5)
end


function get_relative_barlength_distribution(D; alpha = 0.05, n = 10)
    # find empirical distribution of bar lengths relative to the median bar length 
    # (permute matrix D and compute persistençce) x 10   
    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = analogous_bars.permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1] 
        bar_median = median(bar_lengths)
        append!(length_sample, bar_lengths ./ bar_median)
    end

    # find barlength parameter l such that P(obeserving bar length >= l) <= alpha
    length_param = quantile(length_sample, 1-alpha)

    return length_param, length_sample

    # To plot histogram of sampled bar_lengths:
    #histogram(length_sample, label="", title="Histogram of bar lengths of shuffled distance", color = :grey)
    #vline!([length_param], label = "alpha = 0.05", linewidth = 5)

end


function get_empirical_relative_barlengths(D; n = 10)
    # find empirical distribution of bar lengths relative to the median bar length 
    # (permute matrix D and compute persistençce) x 10   
    # D: Distance matrix
    # !!! SPEEDUP VIA MULTIPROCESSING
    length_sample = []
    for i =1:n
        D_perm = analogous_bars.permute_symmetric_array(D)
        C_perm = eirene(D_perm)
        barc = barcode(C_perm, dim=1)
        bar_lengths = barc[:,2]-barc[:,1] 
        bar_median = median(bar_lengths)
        append!(length_sample, bar_lengths ./ bar_median)
    end

    return length_sample

end

"""select_persistent_intervals_IQR()
Use IQR to identify "outlier points" on a persistence diagram with large lifetime 
"""
function select_persistent_intervals_IQR(data_barcode; k = 3)
    bar_lengths = data_barcode[:,2] - data_barcode[:,1]
    _, Q1, Q2, Q3, _ = nquantile(bar_lengths, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above cutoff
    selected_idx = [idx for (idx, v) in enumerate(bar_lengths) if v > cutoff]
    
    return selected_idx, cutoff
end


function select_persistent_intervals(data_barcode, D; dim = 1, n = 10, k = 3, use_min_length = false)
    
    # get relative lengths from given matrix
    interval_length = data_barcode[:,2] - data_barcode[:,1]
    if use_min_length == true
        min_length = minimum(interval_length)
        rel_length = interval_length ./ min_length
    else
        median_length = median(interval_length)
        rel_length = interval_length ./ median_length
    end
    
    # get empirical distribution of relative lengths from shuffled distances
    empirical_rel_lengths = get_empirical_relative_barlengths(D, n = n)
   
    # combine true and empirical relative lengths
    combined = vcat(rel_length, empirical_rel_lengths)
    
    # square-root transform
    sqrt_combined = [sqrt(x) for x in combined]
    
    # compute the points above the IQR range with given "k"
    _, Q1, Q2, Q3, _ = nquantile(sqrt_combined, 4)
    cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above the cutoff 
    selected_idx = [idx for (idx, v) in enumerate(sqrt_combined) if v > cutoff]
    
    return selected_idx, empirical_rel_lengths, combined, cutoff
    
end


function select_persistent_intervals_sub(data_barcode, D; dim = 1, n = 10, use_min_length = false)
    
    # get relative lengths from given matrix
    interval_length = data_barcode[:,2] - data_barcode[:,1]
    if use_min_length == true
        min_length = minimum(interval_length)
        rel_length = interval_length ./ min_length
    else
        median_length = median(interval_length)
        rel_length = interval_length ./ median_length
    end
    
    # get empirical distribution of relative lengths from shuffled distances
    empirical_rel_lengths = get_empirical_relative_barlengths(D, n = n)
   
    # combine true and empirical relative lengths
    combined = vcat(rel_length, empirical_rel_lengths)
    
    # square-root transform
    sqrt_combined = [sqrt(x) for x in combined]
    
    # compute the points above the IQR range with given "k"
    #_, Q1, Q2, Q3, _ = nquantile(sqrt_combined, 4)
    #cutoff = Q3 + k * (Q3 - Q1)
    
    # find index of points above the cutoff 
    #selected_idx = [idx for (idx, v) in enumerate(sqrt_combined) if v > cutoff]
    
    return empirical_rel_lengths, combined#, cutoff
    
end


"""
    select_intervals(D; <keyword arguments>)
Select intervals of a barcode with long enough length. Performs the following. \n
1. Get empirical distribution of interval lengths by permuting the distance matrix `D`. \n
2. Select `min_length` at which ``P(x \\geq min\\_length) \\leq \\alpha`` for some value of ``\\alpha`` (defaults to 0.05). \n
3. Select intervals of barcode(Y) whose length is greater than or equal to `min_length` \n

### Arguments
- `D::Array` : Distance matrix
- `dim::Int64=1`: Dimension
- `n::Int64=10`: number of times to shuffle the distance matrix
- `relative::bool = true`: whether to consider relative bar lengths or absolute bar lengths
- `use_min_length::bool = false`: When `relative = true` and `use_min_length = true`, consider bar lengths relative to median. If `relative = true` and `use_min_length = false`, consider bar lengths relative to the minimum bar length. Set this parameter to true when your barcode has few bars. 

### Outputs
- `selected::Array` : List of indices corresponding to selected intervals of the barcode. 
"""
function select_intervals(D; dim = 1, n = 10, alpha = 0.001, relative = true, use_min_length = false)
    # get barcode
    C = eirene(D, maxdim = dim)
    barc = barcode(C, dim = dim)
    interval_length = barc[:,2] - barc[:,1]
    
    # empirical distribution of relative length
    if relative == true
        min_length, sample_distribution = get_relative_barlength_distribution(D, alpha = alpha, n = n) 
        

        if use_min_length == true
            median_length = minimum(interval_length)
        else
            median_length = median(interval_length)
        end
        
        rel_length = interval_length ./ median_length
        selected = [i for (i, v) in enumerate(rel_length) if v >= min_length]

        return selected, min_length, sample_distribution
    # get empirical distribution of absolute length
    else
        min_length, sample_distribution = analogous_bars.get_barlength_distribution(D, alpha = alpha, n = n) 
        selected = [i for (i,v) in enumerate(interval_length) if v >= min_length]
        return selected, min_length, sample_distribution
    end
    
end



"""
    prep_lazy_component_extension(C_Y, D_Y, epsilon; <keyword arguments>)
Prepare for component-wise extension at parameter "epsilon" by
    (1) building `C_Y_epsilon`, which is the persistence `C_Y` with `maxrad = epsilon`.
    (2) building `C_Y_epsilon_filled`, which is 'C_Y_epsilon` with specific cycles filled in. 
        For each interval of 'barcode(C_Y, dim = dim)' that's not in `selected_intervals`, we add a (dim + 1)-dimensional cell
        whose boundary is the cycle representative of the interval.     
    (3) finding the cycle representatives of intervals, expressed using the indexing of C_Y_epsilon (and hence C_Y_epsilon_filled)
We prepare `C_Y_epsilon_filled` because we use the boundary matrix and code from Eirene to check linear systems.

### Arguments
- `C_Y::Dict`: Output of Eirene on D_Y
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `epsilon::Float64`: Selected parameter
- `dim::Int64=1`: Dimension. Defaults to 1
- `selected_intervals::Array = []`: list of integers corresponding to the selected intervals in barcode(C_Y, dim = dim)

### Outputs
- `C_Y_epsilon::Dict`: output of Eirene on D_Y with maxrad = epsilon 
- `C_Y_epsilon_filled::Dict`: output of Eirene after adding specific cells to `C_Y_epsilon`
- `candidates_rep::Dict`: 
    * keys: Selected intervals of barcode(C_Y, dim = dim) that exist at parameter epsilon
    * vals: cycle representative of the interval expressed as a list of indices in C_Y_epsilon_filled
"""
function prep_lazy_component_extension(
    C_Y::Dict, 
    D_Y::Array,
    epsilon::Float64; 
    dim::Int64=1,
    selected_intervals = [])

    ##### 1. Build 'C_Y_epsilon_filled' #####
    # barcode of C_Y
    barc = barcode(C_Y, dim = dim)
    
    # build the complex "Y_epsilon" by running Eirene on D_Y up to epsilon 
    C_Y_epsilon = eirene(D_Y, maxrad = epsilon, record = "all", maxdim = dim)
    
    # get rv, cp, fv of C_Y_epsilon
    rv, cp, fv = Eirene_var.eirened2complex(C_Y_epsilon)
    
    # find unselected intervals of barcode(C_Y) that exist at parameter 'epsilon'
    unselected = [i for i = 1:size(barc,1) if (i ∉ selected_intervals) & (barc[i,1] <= epsilon) & (epsilon < barc[i,2])]
    
    # add (dim + 1)-dimensional cell by appending to rv[dim + 1], cp[dim + 1], and fv[dim + 1]
    # note: adding cells to higher dimensional cells will require
    # me to figure out some kind of indexing issue
    # COME BACK TO THIS
    for i in unselected
        rep = classrep(C_Y, dim = dim, class = i, format = "vertex x simplex") 
        rep_v = [rep[:,j] for j =1:size(rep,2)]
        rep_index = analogous_bars.chain_to_index(rep_v, C_Y_epsilon)

        # add cell having cycle_idx as boundary
        append!(rv[dim + 2], rep_index)
        push!(cp[dim + 2], last(cp[dim + 2]) + size(rep_index)[1])
        push!(fv[dim + 2], epsilon)
    end
        
    # run Eirene with filled in cycles
    # !!!!! NOTE !!!!! The following line can take a long time
    C_Y_epsilon_filled = eirene(rv = rv, cp = cp, fv = fv, record = "all")

    ##### 2. Find cycle representatives of selected intervals in 'C_Y_epsilon_filled' #####
    # select intervals that exist at given parameter
    selected_epsilon = [i for i in selected_intervals if (barc[i,1] <= epsilon) & (epsilon < barc[i,2])]
    # note: indexing is according to C_Y_epsilon, which is the same as C_Y_epsilon_filled
    candidates_rep = Dict()
    for i in selected_epsilon
        rep = classrep(C_Y, dim = dim, class = i, format = "vertex x simplex") 
        rep_v = [rep[:,j] for j =1:size(rep,2)]
        candidates_rep[i] = analogous_bars.chain_to_index(rep_v, C_Y_epsilon)
    end
    
    return C_Y_epsilon, C_Y_epsilon_filled, candidates_rep
end


function find_lazy_component_extension(
    cycle_reps, 
    C_Y_epsilon::Dict, 
    candidates_Y_epsilon::Dict; 
    dim::Int64 = 1,
    C_Y_epsilon_filled::Dict = Dict())

    # express intervals_Xdelta as a chain (list of indices) in C_Y_epsilon, using the indexing of C_Y_epsilon
    intervals_Xdelta_epsilon = chain_to_index(cycle_reps, C_Y_epsilon)

    # Find linear combinations of candidate intervals in barcode(Y)
    candidates_Y = collect(keys(candidates_Y_epsilon))
    limit = size(candidates_Y,1)
    #comb_Y = find_linear_combinations(candidates_Y)

    # For each linear combination, check if the cycle rep of the combination is homologous 
    # to intervals_Xdelta_epsilon in Y_epsilon
    extension = nothing
    
    i = 1 
    while (extension == nothing) & (i <= limit) 
        # find combinations of "i" number of items 
        combination = collect(combinations(candidates_Y,i))
    
        for comb in combination
            comb_chain = vcat([candidates_Y_epsilon[item] for item in comb]...)
            comb_chain = select_odd_count(comb_chain)
            # LAZY EXTENSION
            homologous = check_homologous_cycles(comb_chain, intervals_Xdelta_epsilon, C_Y_epsilon_filled)
            if homologous == true
                extension = comb
                return extension
            end
        end
        i += 1
    end
    return extension
end

"""
    find_Ybar_representations_at_parameter(C_aux, param, cyclereps, C_Y, D_Y; <keyword arguments>)
Find component-wise Y-bar representations at given parameter for a fixed interval decomposition of barcode(C_Y). 

### Arguments
- `C_aux::Dict`: dictionary output of Eirene. Represents filtration on X_delta
- `param::Float64`: parameter output of function "find_epsilon0_Fbar_representation_tau"
- `cyclereps::Dict`: Cycle representatives of bars in barcode(C_aux, dim = dim)
- `C_Y::Dict`: dictionary output of Eirene
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `dim::Int64=1`: Dimension. Defaults to 1

### Outputs
- `Ybar_Far_tau`: Y-bar representations of Fbar_tau at given parameter
- `Ybar_short`: Y-bar representations of short bars at given parameter
"""
function find_Ybar_representations_at_parameter(
    C_aux::Dict,
    param::Float64,
    cyclereps::Dict,
    C_Y::Dict,
    D_Y::Array;
    dim::Int64=1)
    
    # Build C_Y_epsilon and find bars of barcode(Y) present at epsilon_0
    C_Y_epsilon, bars_Y_cyclerep = prep_bar_representation(C_Y, D_Y, param, dim = dim)
    
    # find Y-bar representations of Fbar_representation_tau
    Fbar_rep_cycle = cyclereps["Fbar_rep_tau"]
    Ybar_Fbar_tau = find_bar_representation_given_CYepsilon(Fbar_rep_cycle, C_Y_epsilon, bars_Y_cyclerep)
    
    # find bar representations of short bars that exist at epsilon_0
    intervals_short = find_short_intervals_at_param(C_aux, param, dim = dim)
    bar_rep_short = Dict()
    for i in intervals_short
        result = find_bar_representation_given_CYepsilon(cyclereps[i], C_Y_epsilon, bars_Y_cyclerep)
        if result != nothing
            bar_rep_short[i] = result
        end
    end
    
    return Ybar_Fbar_tau, bar_rep_short
    
end


"""
    find_lazy_Ybar_representations_at_parameter(C_aux, param, cyclereps, C_Y, D_Y; <keyword arguments>)
Find component-wise LAZY Y-bar representations at given parameter for a fixed interval decomposition of barcode(C_Y). 

### Arguments
- `C_aux::Dict`: dictionary output of Eirene. Represents filtration on X_delta
- `param::Float64`: parameter output of function "find_epsilon0_Fbar_representation_tau"
- `cyclereps::Dict`: Cycle representatives of bars in barcode(C_aux, dim = dim)
- `C_Y::Dict`: dictionary output of Eirene
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `dim::Int64=1`: Dimension. Defaults to 1
- `selected_Y_bars::Array = []`: list of integers corresponding to the selected bars in barcode(C_Y, dim = dim)

### Outputs
- `Ybar_Far_tau`: Lazy Y-bar representations of Fbar_tau at given parameter
- `Ybar_short`: Lazy Y-bar representations of short bars at given parameter
"""
function find_lazy_Ybar_representations_at_parameter(C_aux::Dict,
                                                    param::Float64,
                                                    cyclereps::Dict,
                                                    C_Y::Dict,
                                                    D_Y::Array;
                                                    dim::Int64=1,
                                                    selected_Y_bars = [])

    # prepare for lazy component extension
    C_Y_epsilon, C_Y_epsilon_filled, candidates_rep = prep_lazy_component_extension(C_Y, D_Y, param, dim = dim, selected_intervals = selected_Y_bars)

    # find the lazy bar representation of "Fbar_rep_tau"
    Ybar_Fbar_tau = find_lazy_component_extension(cyclereps["Fbar_rep_tau"], C_Y_epsilon, candidates_rep, C_Y_epsilon_filled = C_Y_epsilon_filled)
    
    # find the lazy bar representations for short intervals that exist at parameter
    intervals_short = find_short_intervals_at_param(C_aux, param, dim = dim)
    Ybar_short = Dict()
    for i in intervals_short
        result = find_lazy_component_extension(cyclereps[i], C_Y_epsilon, candidates_rep, C_Y_epsilon_filled = C_Y_epsilon_filled)
        if result != nothing
            Ybar_short[i] = result
        end
    end
    
    return Ybar_Fbar_tau, Ybar_short
    
end

"""
    cycle_extension_to_bar_extension(cycle_reps, C_aux, C_Y, D_Y, Y_param; selected_Y_bars = [], dim = 1)
Given component-wise cycle extensions, find the (lazy) bar extensions at a fixed parameter and interval decomposition of the target barcode.
If `selected_Y_bars` isn'y empty, then find the LAZY bar extension that consists only of the selected bars.

### Arguments
- `cycle_reps::Dict`: Component-wise cycle extensions. Coincides with the cycle representatives of bars in the auxiliary filtration `C_aux` 
- `C_aux::Dict`: dictionary output of Eirene on the auxiliary filtration
- `C_Y::Dict`: dictionary output of Eirene for target filtration
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `Y_param::Float`: Selected parameter in the target barcode. 
- `dim::Int64=1`: Dimension. Defaults to 1
- `selected_Y_bars::Array = []`: list of integers corresponding to the selected bars in barcode(C_Y, dim = dim)

### Outputs
- `Ybar_Fbar_tau`: Y-bar representation of "Fbar_tau"
- `bar_rep_short`: dictionary representing the bar extensions of short bars that are non-trivial at `Y_param`
"""
function cycle_extension_to_bar_extension(cycle_reps::Dict, 
                                          C_aux::Dict, 
                                          C_Y::Dict, 
                                          D_Y::Array, 
                                          Y_param::Float64; 
                                          selected_Y_bars = [], 
                                          dim = 1)
    
    # component-wise bar extension
    if size(selected_Y_bars,1) == 0
        Ybar_Fbar_tau, bar_rep_short = find_Ybar_representations_at_parameter(C_aux, 
                                                                              Y_param, 
                                                                              cycle_reps, 
                                                                              C_Y, 
                                                                              D_Y, 
                                                                              dim = dim)
        return Ybar_Fbar_tau, bar_rep_short
    
    # component-wise LAZY bar extension 
    else
        Ybar_Fbar_tau_lazy, bar_rep_short_lazy = find_lazy_Ybar_representations_at_parameter(C_aux, 
                                                                                            Y_param, 
                                                                                            cycle_reps, 
                                                                                            C_Y, 
                                                                                            D_Y; 
                                                                                            dim = dim, 
                                                                                            selected_Y_bars = selected_Y_bars)
    
        return Ybar_Fbar_tau_lazy, bar_rep_short_lazy
    end
    
end

"""
    run_cycle_extension_VR_to_VR_bar(; <keyword arguments>)
Runs the bar-to-bar extension method to compare two Vietoris-Rips filtrations. Given a selected bar `Z_bar` in `barcode(C_Z, dim = dim)`, finds its cycle extension and bar extension in `C_Y` and `barcode(C_Y, dim = dim)`.

Using paper notation: given a selected bar in ``\\text{BC}_k(Z^{\\bullet})``, finds the cycle extension in ``Y^{\\bullet}`` and bar extension in ``\\text{BC}_k(Y^{\\bullet})``.

### Arguments
- `C_Z::Dictionary`: Eirene output of filtration `Z`
- `D_Z::Array`: Distance matrix used to compute `C_Z`
- `C_Y::Dictionary`: Eirene output of filtration `Y`
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `Z_bar::Integer`: Index of selected bar from `barcode(C_Z, dim = dim)`
- `dim::Integer=1`: Dimension. Defaults to 1

### Returns
- `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_VR_to_VR`
"""
function run_cycle_extension_VR_to_VR_bar(;
    C_Z::Dict{String, Any} = Dict{String, Any}(),
    D_Z::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    C_Y::Dict{String, Any} = Dict{String, Any}(),
    D_Y::Array{Float64,2} = Array{Float64}(undef, 0, 0),
    Z_bar::Int64 = 0,
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "delta"
    tau, psi = find_terminal_class_in_VR(C_Z, D_Z, bar = Z_bar, dim = dim)
    
    # run extension method
    extension = run_cycle_extension_VR_to_VR(C_Z = C_Z, Z_cycle = tau, psi = psi, C_Y = C_Y, D_Y = D_Y, dim = dim)

    return extension
end
"""
    run_cycle_extension_VR_to_VR(; <keyword arguments>)
Runs the component-wise cycle-to-bar persistent extension method to compare two Vietoris-Rips filtrations. Given a cycle `Z_cycle` and a parameter `psi`, finds its cycle and bar extensions in `C_Y` and `barcode(C_Y, dim = dim)`.

Using paper notation: given a cycle and a parameter ``[\\tau] \\in H_k(Z^{\\psi})``, finds its representations in ``Y^{\\bullet}`` and ``\\text{BC}_k(Y^{\\bullet})``.

### Arguments
- `C_Z::Dictionary`: Eirene output of filtration `Z`
- `Z_cycle`: Selected cycle of interest in complex `Z`. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
- `psi`: parameter
- `C_Y::Dictionary`: Eirene output of filtration `Y`
- `D_Y::Array`: Distance matrix used to compute `C_Y`
- `dim::Integer=1`: Dimension. Defaults to 1

### Outputs
Function outputs a dictionary with the following keys. 
**For a more detailed explanation, see the document "code_details.pdf".**

- `comparison`: Indicates the types of filtrations under comparison. Value will be "VR to VR".
- `C_Z`: copy of input `C_Z`
- `C_Y`: copy of input `C_Y`
- `dim`: dimension of interest. copy of input `dim`
- `selected_cycle`: copy of input `Z_cycle`
- `C_auxiliary_filtration`: Eirene output of filtration ``Z^{\\psi} \\cap Y^{\\bullet}``
- `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
- `p_Y`: collection of parameter values for ``p_Y``
- `epsilon_0`: minimum value in `p_Y`
- `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
- `nontrivial_pY_dict`: A dictionary of parameter index and values.
- `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
- `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
- `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
"""
function run_cycle_extension_VR_to_VR(;
    C_Z::Dict{String, Any} = Dict{String, Any}(),
    Z_cycle::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0, 
    C_Y::Dict{String, Any} = Dict{String, Any}(),
    D_Y::Array{Float64,2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)

    ##### check input #####
    if C_Z == Dict()
        throw(UndefKeywordError(:C_Z))
    end
    if C_Y == Dict()
        throw(UndefKeywordError(:C_Y))
    end
    if D_Y == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Y))
    end
    if Z_cycle == 0
        throw(UndefKeywordError(:Z_cycle))
    end
    if psi == -1.0
        throw(UndefKeywordError(:psi))
    end

    # build auxiliary filtration 
    C_aux, Zpsi_index2simplex, Zpsi_simplex2index = build_auxiliary_filtration(C_Z, psi, D_Y, format = "VR to VR", dim = dim)
    
    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau(Z_cycle, C_aux, Zpsi_simplex2index, dim = dim)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_Y, dim = dim)[:,2])
    p_Y, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using vertices
    cycle_reps = find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex)

    # find component-wise bar-representations
    #Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short = find_all_bar_representations(C_aux, p_Y, cycle_reps, C_Y, D_Y, dim = dim)
    
    ### create output dictionary
    extension = Dict()
    # input data
    extension["comparison"] = "VR to VR"
    extension["C_Z"] = C_Z
    extension["C_Y"] = C_Y
    extension["dim"] = dim 
    extension["selected_cycle"] = Z_cycle 
    # auxiliary filtration & cyclerep
    extension["C_auxiliary_filtration"] = C_aux
    extension["aux_filt_cyclerep"] = cycle_reps
    # parameters pY
    extension["p_Y"] = p_Y
    extension["epsilon_0"] = epsilon_0
    # component-wise Ybar representations
    #extension["Ybar_rep_tau"] = Ybar_rep_tau
    #extension["Ybar_rep_short_epsilon0"] = Ybar_rep_short_epsilon0
    #extension["Ybar_rep_short"] = Ybar_rep_short
    # exploring results by parameter
    #extension["nontrivial_pY"] = nontrivial_pY
    #extension["nontrivial_pY_dict"] = nontrivial_pY_dict
    #extension["cycle_extensions"] = cycle_extensions
    #extension["bar_extensions"] = bar_extensions
    
    return extension
end




"""
    run_cycle_extension_W_to_VR_bar(<keyword arguments>)   
Runs the bar-to-bars extension method to compare a Witness barcode and a VR barcode. Given a bar `W_bar` of a Witness filtration, finds its cycle and bar extensions in the VR barcode `barcode(C_VR, dim = dim)`.

### Arguments
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `W_bar::Int64`: Selected bar in Witness barcode
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Returns
- `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_W_to_VR` 
"""
function run_cycle_extension_W_to_VR_bar(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_bar::Int64 = 0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "delta"
    tau, psi = find_terminal_class_in_W(W, bar = W_bar, dim = dim)
    
    # run extension method from W to VR
    extension = run_cycle_extension_W_to_VR(W = W, 
                                      W_cycle = tau, 
                                      psi = psi, 
                                      C_VR = C_VR, 
                                      D_VR = D_VR, 
                                      dim = dim)
    
    # add key to 'extension' dic
    extension["selected_bar"] = W_bar
    return extension
end

"""
    run_cycle_extension_W_to_VR(; <keyword arguments>)
Runs the cycle-to-bar extension method to compare a Witness barcode to a Vietoris-Rips barcode. Given a cycle `W_cycle` in the Witness filtration, finds its cycle and bar extensions in `C_VR` and `barcode(C_VR, dim = dim)`. Note: The point cloud of the VR filtration and the landmark of the Witness filtration must be the same.

### Arguments
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `W_cycle`: cycle of interest. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
- `psi`: parameter
- `W_class::Int64`: Interval of interest in Witness barcode
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Outputs
Function outputs a dictionary with the following keys. 
**For a more detailed explanation, see the document "code_details.pdf".**

- `comparison`: Indicates the types of filtrations under comparison. Value will be "W to VR".
- `C_W`: copy of input `C_W`
- `C_VR`: copy of input `C_VR`
- `dim`: dimension of interest. copy of input `dim`
- `selected_cycle`: copy of input `W_cycle`
- `C_auxiliary_filtration`: Eirene output of filtration ``W^{\\psi} \\cap VR^{\\bullet}``
- `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
- `p_Y`: collection of parameter values for ``p_Y``
- `epsilon_0`: minimum value in `p_Y`
- `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
- `nontrivial_pY_dict`: A dictionary of parameter index and values.
- `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
- `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
- `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
- `cycle_extensions`: A dictionary summarizing cycle extensions at various parameters.
- `bar_extensions`: A dictionary summarizing bar extensions at various parameters. 
"""
function run_cycle_extension_W_to_VR(;
    W::Dict{Any, Any} = Dict{Any, Any}(),
    W_cycle::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # unpack variables
    C_W = W["eirene_output"]
    D_W = W["distance_matrix"]
    W_index2simplex = W["index2simplex"]
    W_fv = W["fv"]
    
    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex = build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
    
    # find epsilon_0, Fbar_representation_tau
    
    epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau_WtoVR(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)
    
    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    p_Y, BARS_short = find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim) 
    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps = find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)
    
       
    ### create output dictionary
    extension = Dict()
    # input data
    extension["comparison"] = "W to VR"
    extension["C_W"] = C_W
    extension["C_VR"] = C_VR
    extension["dim"] = dim 
    extension["selected_cycle"] = W_cycle 
    # auxiliary filtration & cyclerep
    extension["C_auxiliary_filtration"] = C_aux
    extension["aux_filt_cyclerep"] = cycle_reps
    # parameters pY
    extension["p_Y"] = p_Y
    extension["epsilon_0"] = epsilon_0

    return extension
end


"""
    run_cycle_extension_VR_to_W_bar(<keyword arguments>)
Runs the bar-to-bars extension method to compare a Vietoris-Rips barcode and a Witness barcode. Given a bar `VR_bar` in the VR barcode, finds the cycle and bar extensions in `W` and `barcode(W, dim = dim)`. 

### Arguments
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `VR_bar::Int64`: Selected bar in the VR barcode
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `D_W::Array{Float64, 2}`: Distance matrix used to compute `C_W`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Returns
- `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_VR_to_W` 
"""
function run_cycle_extension_VR_to_W_bar(;
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_bar::Int64 = 0,
    W::Dict{Any, Any} = Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "psi"
    tau, psi = find_terminal_class_in_VR(C_VR, D_VR, bar = VR_bar, dim = dim)
    
    # run extension method from W to VR
    extension = run_cycle_extension_VR_to_W(C_VR = C_VR, D_VR = D_VR, VR_cycle = tau, psi = psi, W = W, D_W = D_W, dim = dim)
    return extension
end

"""
    run_cycle_extension_VR_to_W(; <keyword arguments>)
Runs the cycle-to-bars extension method to compare a Vietoris-Rips barcode and a Witness barcode. Given a cycle `VR_cycle` in the VR barcode, finds the cycle and bar extensions in `W` and `barcode(W, dim = dim)`. 
Note: The 0-simplices of the Vietoris-Rips filtration and the Witness filtration must be the same.

### Arguments
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `VR_cycle`: cycle of interest in the VR filtration. Format: ``[[v^1_1, v^1_2], [v^2_1, v^2_2], ...,]``
- `psi`: parameter
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `D_W::Array{Float64, 2}`: Distance matrix used to compute the Witness filtration
- `dim::Int64 = 1`: Dimension. Defaults to 1    

### Outputs
Function outputs a dictionary with the following keys. 
**For a more detailed explanation, see the document "code_details.pdf".**

- `comparison`: Indicates the types of filtrations under comparison. Value will be "W to VR".
- `C_VR`: copy of input `C_VR`
- `C_W`: copy of input `C_W`
- `dim`: dimension of interest. copy of input `dim`
- `selected_cycle`: copy of input `VR_cycle`
- `C_auxiliary_filtration`: Eirene output of filtration ``VR^{\\psi} \\cap W^{\\bullet}``
- `aux_filt_cycle_rep`: cycle representatives of bars in `C_auxiliary_filtration`
- `p_Y`: collection of parameter values for ``p_Y``
- `epsilon_0`: minimum value in `p_Y`
- `nontrivial_pY`: A subcollection of parameters in `p_Y`. To find all cycle extensions or bar extensions, it suffices to consider only the parameters in `nontrivial_pY` (instead of `p_Y`, which is quite large)
- `nontrivial_pY_dict`: A dictionary of parameter index and values.
- `Ybar_rep_tau`: ``\\mathcal{D}``-bar representation of the ``\\mathcal{F}``-bar representations of the `selected_cycle`. 
- `Ybar_rep_short_epsilon0`: A dictionary of the short bars of `C_auxiliary_filtration` that are alive at parameter `epsilon_0` and their ``\\mathcal{D}``-bar representations.
- `Ybar_rep_short`: A dictionary of all other short bars of `C_auxiliary_filtration` and their ``\\mathcal{D}``-bar representations.
- `cycle_extensions`: A dictionary summarizing cycle extensions at various parameters.
- `bar_extensions`: A dictionary summarizing bar extensions at various parameters. 
"""
function run_cycle_extension_VR_to_W(;
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_cycle::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0, 
    W::Dict{Any, Any} = Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
      
    ##### check input #####
    if W == Dict()
        throw(UndefKeywordError(:W))
    end
    if C_VR == Dict()
        throw(UndefKeywordError(:C_VR))
    end
    if D_VR == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_VR))
    end
    if D_W == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_W))
    end
    if psi == -1.0
        throw(UndefKeywordError(:psi))
    end
    
    
    # unpack variables
    C_W = W["eirene_output"]
    
    # build the auxiliary filtration
    C_aux, Zpsi_index2simplex, Zpsi_simplex2index = build_auxiliary_filtration(C_VR, psi, D_W, format = "VR to W", dim = dim)

    # find epsilon_0, Fbar_representation_tau
    epsilon_0, Fbar_representation_tau = find_epsilon0_Fbar_representation_tau(VR_cycle, C_aux, Zpsi_simplex2index, dim = dim)

    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_W, dim = dim)[:,2])
    p_Y, BARS_short = find_pY_and_BARSshort(C_aux, epsilon_0, dim = dim)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using vertices
    cycle_reps = find_cyclereps_auxiliary(Fbar_representation_tau, BARS_short, C_aux, Zpsi_index2simplex)
 
    ### create output dictionary
    extension = Dict()
    # input data
    extension["comparison"] = "VR to W"
    extension["C_VR"] = C_VR
    extension["C_W"] = C_W
    extension["dim"] = dim 
    extension["selected_cycle"] = VR_cycle 
    # auxiliary filtration & cyclerep
    extension["C_auxiliary_filtration"] = C_aux
    extension["aux_filt_cyclerep"] = cycle_reps
    # parameters pY
    extension["p_Y"] = p_Y
    extension["epsilon_0"] = epsilon_0
    
    return extension
end



#------------------------------------------------------------------------------------------------------------------
# Code for running (baseline) persistent extension from saved Witness persistence info
#------------------------------------------------------------------------------------------------------------------

function apply_Dowker_from_saved_Witness(
    C_W_PQ,
    C_W_QP;
    dim = 1)

    P_to_Q = Dict()
    barcode_W_PQ = barcode(C_W_PQ, dim = dim)
    barcode_W_QP = barcode(C_W_QP, dim = dim)
    n = size(barcode_W_PQ)[1]
    for i=1:n
        birth_rows = findall(x->x==1, (barcode_W_QP[:, 1] .== barcode_W_PQ[i,1]))
        if size(birth_rows)[1] > 1
            print("ERROR: multiple bars with same birth time")
            break
        else
            P_to_Q[i] = birth_rows[1]
        end
    end
    return P_to_Q
end

"""
    run_similarity_analogous_from_saved_Witness(; <keyword arguments>)
runs the `run_full_similarity_analogous`, but using pre-computed Witness persistence. Return both baseline and offset bar extensions at epsilon0
"""
function run_epsilon0_similarity_analogous_from_saved_Witness(;
    VR_P::Dict{String, Any} = Dict{String, Any}(),
    D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    VR_Q::Dict{String, Any} = Dict{String, Any}(),
    D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    C_W_PQ::Dict{String, Any} = Dict{Any, Any}(),
    D_PQ::Array{Float64, 2} = Array{Float64}(under, 0, 0), 
    W_PQ_index2simplex = Dict{Any, Any}(), 
    W_PQ_vertex_to_default_vertex = nothing, 
    W_PQ_bars::Array{Int64,1} = [],
    C_W_QP::Dict{String, Any} = Dict{Any, Any}(),
    D_QP::Array{Float64, 2} = Array{Float64}(under, 0, 0), 
    W_QP_index2simplex= Dict{Any, Any}(),
    W_QP_vertex_to_default_vertex = nothing, 
    dim::Int64 = 1)
    
    ##### check input #####
    if VR_P == Dict()
        throw(UndefKeywordError(:VR_P))
    end
    if VR_Q == Dict()
        throw(UndefKeywordError(:VR_Q))
    end
    if D_P == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_P))
    end
    if D_Q == Array{Float64}(undef, 0, 0)
        throw(UndefKeywordError(:D_Q))
    end
    if W_PQ_bars == []
        throw(UndefKeywordError(:W_PQ_class))
    end
    
    analogous_P = Dict()
    analogous_Q = Dict()
    
    # find the bar in W(Q,P) that corresponds to W_PQ_bar
    P_to_Q = apply_Dowker_from_saved_Witness(C_W_PQ, C_W_QP, dim = dim)
    
    for i in W_PQ_bars
        W_PQ_bar = i
        W_QP_bar = P_to_Q[W_PQ_bar]
    
        ##### apply the extension method between W(P,Q) and VR(P) #####
        extension_P = run_extension_W_to_VR_bar_epsilon0_from_saved_Witness(C_W = C_W_PQ, D_W = D_PQ, W_index2simplex = W_PQ_index2simplex, W_vertex_to_default_vertex = W_PQ_vertex_to_default_vertex, W_bar = W_PQ_bar, C_VR = VR_P, D_VR = D_P, dim = dim)

        ##### apply the extension method between W(P,Q) and VR(P) #####
        extension_Q = run_extension_W_to_VR_bar_epsilon0_from_saved_Witness(C_W = C_W_QP, D_W = D_QP, W_index2simplex = W_QP_index2simplex, W_vertex_to_default_vertex = W_QP_vertex_to_default_vertex, W_bar = W_QP_bar, C_VR = VR_Q, D_VR = D_Q, dim = dim)
        
        analogous_P[i] = extension_P
        analogous_Q[i] = extension_Q
        
    end

    return analogous_P, analogous_Q
end

function find_terminal_class_in_W_from_saved_Witness(
    ;
    C_W::Dict{String, Any}= Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(under, 0, 0), 
    W_index2simplex = Dict{Any, Any}(), 
    W_vertex_to_default_vertex = nothing, 
    bar::Int64 = 0,
    dim = 1)

    
    # find class rep of bar
    rep = classrep(C_W, class = bar, dim = dim)
    
    # express as a collection of simplices
    tau = [W_index2simplex[(item, dim)] for item in rep]
        
    if W_vertex_to_default_vertex != nothing
        tau = [[W_vertex_to_default_vertex[item] for item in simplex] for simplex in tau]
    end
    
    # find parameter delta(tau)-1
    d = barcode(C_W, dim = dim)[bar, 2]
    psi = maximum(D_W[D_W.< d])
    
    return tau, psi
end

"""
    run_extension_W_to_VR_bar_epsilon0_from_saved_Witness(<keyword arguments>)   
Runs the bar-to-bars extension method to compare a Witness barcode and a VR barcode. Given a bar `W_bar` of a Witness filtration, finds its cycle and bar extensions in the VR barcode `barcode(C_VR, dim = dim)`.

### Arguments
- `W::Dict{Any, Any}`: Dictionary output of function `compute_Witness_persistence`
- `W_bar::Int64`: Selected bar in Witness barcode
- `C_VR::Dict{String, Any}`: Dictionary output of Eirene on Vietoris-Rips filtration
- `D_VR::Array{Float64, 2}`: Distance matrix used to compute `C_VR`
- `dim::Int64 = 1`: Dimension. Defaults to 1

### Returns
- `extension`: Dictionary. **For details, see the document "code_details.pdf"** or the comments on function `run_extension_W_to_VR` 
"""
function run_extension_W_to_VR_bar_epsilon0_from_saved_Witness(;
    C_W::Dict{String, Any} = Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(under, 0, 0), 
    W_index2simplex = Dict{Any, Any}(),
    W_vertex_to_default_vertex = nothing,
    W_bar::Int64 = 0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    
    # find terminal class rep "tau" and parameter "delta"
    tau, psi = find_terminal_class_in_W_from_saved_Witness(C_W = C_W, D_W = D_W, W_index2simplex = W_index2simplex, W_vertex_to_default_vertex = W_vertex_to_default_vertex, bar = W_bar, dim = dim)
    
    # run extension method from W to VR
    extension = run_extension_W_to_VR_epsilon0_from_saved_Witness(C_W = C_W, 
                                      D_W = D_W,
                                      W_index2simplex = W_index2simplex,
                                      W_cycle = tau, 
                                      psi = psi, 
                                      C_VR = C_VR, 
                                      D_VR = D_VR, 
                                      dim = dim)
    
    # add key to 'extension' dic
    extension["selected_bar"] = W_bar
    return extension
end


"""
    run_extension_W_to_VR(; <keyword arguments>)
Similar to `run_extension_W_to_VR`. Uses pre-computed Witness complexes as input
"""
function run_extension_W_to_VR_epsilon0_from_saved_Witness(;
    C_W::Dict{String, Any} = Dict{Any, Any}(),
    D_W::Array{Float64, 2} = Array{Float64}(under, 0, 0), 
    W_index2simplex = Dict{Any, Any}(),
    W_cycle::Array{Array{Int64,1},1} = [[0,0]],
    psi::Float64 = -1.0,
    C_VR::Dict{String, Any} = Dict{String, Any}(),
    D_VR::Array{Float64, 2} = Array{Float64}(undef, 0, 0),
    dim::Int64 = 1)
    

    # build the auxiliary filtration
    C_aux, Wpsi_index2simplex, Wpsi_simplex2index, Wpsi_index2simplex_default_vertex, Wpsi_default_vertex_simplex2index,Wpsi_vertex_to_default_vertex, default_vertex_to_Wpsi_vertex = analogous_bars.build_auxiliary_filtration_W_to_VR(D_W, psi, D_VR, dim = dim)
    
    # find epsilon_0, Fbar_representation_tau
    
    epsilon_0, Fbar_representation_tau = analogous_bars.find_epsilon0_Fbar_representation_tau_WtoVR(W_cycle, C_aux, Wpsi_default_vertex_simplex2index)
    
    # find p_Y and BARS_short
    #max_pY = maximum(barcode(C_VR, dim = dim)[:,2])
    p_Y, BARS_short = analogous_bars.find_pY_and_BARSshort(C_aux, 
                                            epsilon_0, 
                                            dim = dim) 
#                                            max_pY = max_pY)
    
    # find cyclereps of Fbar_representation_tau and BARS_short using DEFAULT VERTEX FORMAT
    cycle_reps = analogous_bars.find_cyclereps_auxiliary_WtoVR(Fbar_representation_tau, BARS_short, C_aux, Wpsi_index2simplex_default_vertex)
    
    # check the minimum parameter at which one can find the boundary matrix from Eirene -- resolved with Eirene_var
    #min_epsilon_with_boundary = analogous_bars.return_min_parameter_with_boundary_matrix(D_VR, p_Y)
    #pY_new = [i for i in p_Y if i >= min_epsilon_with_boundary]
    #pY_new = [min_epsilon_with_boundary]
    
     # find component-wise bar-representations
    Ybar_rep_tau, Ybar_rep_short_epsilon0, Ybar_rep_short = analogous_bars.find_all_bar_representations(C_aux, 
                                                                                         p_Y, 
                                                                                         cycle_reps, 
                                                                                         C_VR, 
                                                                                         D_VR, 
                                                                                         dim = dim)
    
    # create summary (by parameter)
    nontrivial_pY, nontrivial_pY_dict, cycle_extensions, bar_extensions = analogous_bars.create_summary_by_parameter(cycle_reps, 
                                                                                        Ybar_rep_tau, 
                                                                                        Ybar_rep_short_epsilon0, 
                                                                                        Ybar_rep_short, 
                                                                                        min_epsilon_with_boundary, 
                                                                                        C_aux,
                                                                                        C_VR)
       
    ### create output dictionary
    extension = Dict()
    # input data
    extension["comparison"] = "W to VR"
    extension["dim"] = dim 
    extension["selected_cycle"] = W_cycle 
    # auxiliary filtration & cyclerep
    extension["C_auxiliary_filtration"] = C_aux
    extension["aux_filt_cyclerep"] = cycle_reps
    # parameters pY
    extension["p_Y"] = p_Y
    extension["epsilon_0"] = epsilon_0
    extension["min_epsilon_with_boundary"] = min_epsilon_with_boundary
    # component-wise Ybar representations
    extension["baseline_bar_extensions"] = Ybar_rep_tau
    extension["offset_bar_extensions_epsilon0"] = Ybar_rep_short_epsilon0
    extension["offset_bar_extensions"] = Ybar_rep_short
    # exploring results by parameter
    extension["nontrivial_pY"] = nontrivial_pY
    extension["nontrivial_pY_dict"] = nontrivial_pY_dict
    extension["cycle_extensions"] = cycle_extensions
    extension["bar_extensions"] = bar_extensions

    return extension
end

dropcol(M::AbstractMatrix, j) = M[:, deleteat!(collect(axes(M, 2)), j)]

function vector_to_symmetric_matrix(vector, n)
    D = zeros(n,n)
    lower_index = [j+(i-1)*n  for i=1:n for j=i+1:n]
    D[lower_index] = vector
    D = D + transpose(D)
    return D
    
end

function find_single_connected_cycle(cr)
     # cr: classrep returned by eirene. For dim 1, array of shape (2, n), where n is the number of 1-simplices involved
    new_cr = cr[:,1]
    cr = cr[:,2:end]
    v = new_cr[2]
    
    while isempty(cr) == false
        # find the column of cr that contains "v"
        idx = findall(x -> x == v, cr)
        
        if idx != []
            row = idx[1][1]
            col = idx[1][2]

            # find the new vertex (same column, different row)
            if row == 1
                v_row = 2
            elseif row == 2
                v_row = 1
            end

            # add to new_cr
            v = cr[v_row, col]
            append!(new_cr, v)

            # remove column "col" from cr
            cr = dropcol(cr, col)
        else
            return new_cr, cr
        end
        
    end
    return new_cr, cr
end
    
    
function convert_classrep(cr)
    # cr: classrep returned by eirene. For dim 1, array of shape (2, n), where n is the number of 1-simplices involved
    new_cr = []
    
    while isempty(cr) == false
        single_loop, cr = find_single_connected_cycle(cr)
        push!(new_cr, single_loop)
        
    end
    
    return new_cr, cr
end 


function save_classreps(C, classes, filename_prefix)

    for c in classes
        eirene_cr = classrep(C, dim = 1, class = c)
        new_cr, _ = convert_classrep(eirene_cr)
        
        for i = 1:size(new_cr,1)
            filename = filename_prefix * "_" * string(c) * "_component_" * string(i) * ".h5"
            f = h5open(filename, "w")
            f["data"] = new_cr[i]
            close(f)
        end
    end
    
    
end


function select_significant_subcollection(dictionary, selected)
    new_dict = Dict()
    for (key, item) in dictionary
        new_item = [i for i in item if i in selected]
        if new_item != []
            new_dict[key] = new_item
        end
    end
    return new_dict
end


function analogous_cycles_significant(analogous_P, analogous_Q; significant_P = [], significant_Q = [])
    analogous_P_significant = Dict()
    analogous_Q_significant = Dict()
    
    for k in keys(analogous_P)
        if analogous_P[k]["baseline_bar_extension"] == nothing
            analogous_P_significant[k] = nothing
        else
            if significant_P == []
                analogous_P_significant[k] = analogous_P[k]["baseline_bar_extension"]
            else
                analogous_P_significant[k] = [i for i in significant_P if i in analogous_P[k]["baseline_bar_extension"]]
            end 
        end

        if analogous_Q[k]["baseline_bar_extension"] == nothing
            analogous_Q_significant[k] = nothing
        else
            if significant_Q == []
                analogous_Q_significant[k] = analogous_Q[k]["baseline_bar_extension"]
            else
                analogous_Q_significant[k] = [i for i in significant_Q if i in analogous_Q[k]["baseline_bar_extension"]]
            end 
        end
    end

    return analogous_P_significant, analogous_Q_significant
end

##########################################################################################
# functions for plotting Dowker complexes
##########################################################################################

function get_1simplices(D_param)
    n_cols = size(D_param, 2)
    one_simplices = []
    for i = 1:n_cols
        rows = findall(x -> x == 1, D_param[:,i])
        append!(one_simplices, combinations(rows, 2))
    end
    
    return one_simplices
end

function get_2simplices(D_param)
    n_cols = size(D_param,2)
    two_simplices = []
    for i = 1:n_cols
        ones = findall(D_param[:,i])
        append!(two_simplices, collect(combinations(ones,3)))
    end
    
    return unique(two_simplices)
end

"""
    plot_Dowker_complex
Given a dissimilarity matrix D, compute the Dowker complex at parameter `param`.
Uses the rows as potential vertex set. Must provide `PC`, the locations of vertex set corresponding to the rows.

### Inputs
- `D`: Dissimilarity matrix
- `param`: Parameter. Builds the Dowker complex at this parameter.
- `PC`: An array of size (n,2), where n is the number of rows of D. 
        Provides (x,y)-coordinates of vertices corresponding to the  rows of D. 

### Outputs
- a plot
"""
function plot_Dowker_complex(D, param, PC; 
                             show_2simplex = false, 
                             show_unborn_vertices = false,
                             sample_proportion = 1, # if 1, plot all simplices. If less than 1, say 0.5, then plot only 50% of the 2- and 1-simplices
                             c = "#29A0B1",
                            kwargs...)
    n_rows, n_cols = size(D)
    D_param = D .< param
    
    p = plot()
    
    # plot 2-simplices
    if show_2simplex == true
        two_simplices = get_2simplices(D_param)
        # sample "sample_proportion"% of simplices to actually plot
        two_simplices = randsubseq(two_simplices, sample_proportion)
        for simplex in two_simplices
            plot!(p, Plots.Shape([(PC[i,1], PC[i,2]) for i in simplex]), 
                 label="", legend = :false, c = c, alpha = 0.1
                 )
        end
    end
    
    # plot 1-simplices
    one_simplices = get_1simplices(D_param)
    # sample "sample_proportion"% of simplices to actually plot
    one_simplices = randsubseq(one_simplices, sample_proportion)
    for simplex in one_simplices
        plot!(p, [PC[simplex[1],1], PC[simplex[2],1]], [PC[simplex[1],2], PC[simplex[2],2]], label = "", c = :grey, linewidth = 0.5) 
    end
    
    # plot 0-simplices
    idx0 = findall(x -> x != 0, vec(sum(D_param, dims = 2)))
    scatter!(p, PC[idx0,1], PC[idx0, 2]; label = "", frame = :box, ticks = [], c = c, aspect_ratio = :equal, kwargs...)
    
    # plot unborn vertices 
    if show_unborn_vertices == true
        idx_unborn = findall(x -> x == 0, vec(sum(D_param, dims = 2)))
        scatter!(p, PC[idx_unborn,1], PC[idx_unborn, 2], label = "",
                 markershape = :xcross,
                 markerstrokewidth = 3,
                 c = c) 
    end
    return p
end



end
