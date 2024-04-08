"""
Computes the matrix of analogous stats using a null model. Uses multiprocessing

@author Iris Yoon
irishryoon@gmail.com
"""

module analogous_stats
include("Eirene_var.jl")
include("analogous_bars.jl")
using .Eirene_var
using .analogous_bars
using JLD2
using FileIO
using Random
using Statistics
using StatsBase
using Plots

export  angle_distance,
        sample_torus,
        compute_distance_three_torus,
        compute_cross_system_dissimilarity_tori,
        compute_analogous_bars_null_model,
        parse_null_model_stats
        


function test(i, output_dir)
    AP = Dict()
    AP[1] = 1
    save(output_dir * string(i) * ".jld2", "AP", AP)
    print("saving complete")
    return nothing
end

function angle_distance(phi, theta)
    s = min(phi, theta)
    l = max(phi, theta)
    distance = min(l-s, s + 2 * π - l)
    return distance
end

"""
    sample_torus(n)
Samples `n` points on a torus.
"""
function sample_torus(n; dim = 2, ambient_dim = 2)
    
    if dim == 1
        x = rand(n) * 2π 
        T = zeros((n, ambient_dim))
        T[:, 1] = x
        return T
    elseif dim == 2
        x = rand(n) * 2π
        y = rand(n) * 2π
        
        T = zeros((n, ambient_dim))
        T[:, 1] = x
        T[:, 2] = y
        return T
    elseif dim == 3
        x = rand(n) * 2π
        y = rand(n) * 2π
        z = rand(n) * 2π
        
        T = zeros((n, ambient_dim))
        T[:,1] = x
        T[:,2] = y
        T[:,3] = z 
        return T
    end
end


"""
    compute_distance_three_torus()
Given two points on a three torus X and Y, where each is a list of angles in [0, 2pi], compute the distance between X and Y
"""
function compute_distance_three_torus(X, Y)
    
    angle_distances = [angle_distance(X[i], Y[i]) for i = 1:3]
    distance = sqrt(sum([i^2 for i in angle_distances]))

    return distance
end

"""
    compute_distance_torus()
Given two points on a torus X and Y, where each is a list of angles in [0, 2pi], compute the distance between X and Y
"""
function compute_distance_torus(X, Y)
    
    n = size(X, 2)
    angle_distances = [angle_distance(X[i], Y[i]) for i = 1:n]
    distance = sqrt(sum([i^2 for i in angle_distances]))

    return distance
end

"""
    compute_cross_system_dissimilarity_tori(X,Y)
Computes the cross-system dissimilarity matrix between two tori X and Y. 

### Inputs
- `X`: Array of size `(n_X,m)`, where `n_X` is the number of points and `m` is the dimension of torus
- `Y`: Array of size `(n_Y,m)`, where `n_Y` is the number of points and `m` is the dimension of torus

### Outputs
- `D`: Array of size `(n_X, n_Y)`
"""
function compute_cross_system_dissimilarity_tori(X, Y)

    n_X = size(X,1)
    n_Y = size(Y,1)
    println(n_X)
    println(n_Y)

    D = zeros((n_X, n_Y))
    for i = 1:n_X
        for j = 1:n_Y
            X_i = X[i,:]
            Y_j = Y[j,:]
            d = compute_distance_torus(X_i, Y_j)
            D[i,j] = d
        end
    end
    return D
end

"""
    compute_analogous_bars_null_model(; <keyword arguments>)
Computes analogous bars using the tori null model for cross-system dissimilarity. 

### Inputs
- `VR_P::Dictionary`: Eirene output of Vietoris-Rips filtration on system `P`
- `D_P::Array`: Dissimilarity matrix used to compute `VR_P`
- `VR_Q::Dictionary`: Eirene output of Vietoris-Rips filtration on system `Q`
- `D_Q::Array`: Dissimialrity matrix used to compute `VR_Q`
- `P_null_dim::Integer`: Dimension of torus for system `P`
- `Q_null_dim::Integer`: Dimension of torus for system `Q`
- `output_fillename::String`: Filename for saving output. All file names will be appended with `jld.2`
- `save_output::Bool`: Whether to save the output or not

### Outputs
- `analogous_P::Dictionary`: Analogous bars in system `P`
- `analogous_Q::Dictionary`: Analogous bars in system `Q`
- `barcode_W_PT_QT`: barcode of the null model tori
- `W_idx`: Selected bars in `barcode_W_PT_QT`
- `W_cutoff`: Significance cutoff of `barcode_W_PT_QT`
"""
function compute_analogous_bars_null_model(;
        VR_P::Dict{String, Any} = Dict{String, Any}(), 
        D_P::Array{Float64, 2} = Array{Float64}(undef, 0, 0), 
        VR_Q::Dict{String, Any} = Dict{String, Any}(), 
        D_Q::Array{Float64, 2} = Array{Float64}(undef, 0, 0), 
        P_null_dim = 3,
        Q_null_dim = 2,
        output_filename = "", 
        save_output = true)
    
    # get number of points in system P, Q
    n_P = size(D_P, 1)
    n_Q = size(D_Q, 1)

    # Sample points from the "null model" tori
    ambient_dim = maximum([P_null_dim, Q_null_dim])
    P_T = sample_torus(n_P, dim = P_null_dim, ambient_dim = ambient_dim)
    Q_T = sample_torus(n_Q, dim = Q_null_dim, ambient_dim = ambient_dim)
    
    # compute cross-system dissimilarity between the "null model" tori
    D_PT_QT = compute_cross_system_dissimilarity_tori(P_T, Q_T)
    D_QT_PT = Array(transpose(D_PT_QT))
    
    # Compute Witness persistence diagrams of "null model" tori
    W_PT_QT = compute_Witness_persistence(D_PT_QT, maxdim = 1)
    W_QT_PT = compute_Witness_persistence(D_QT_PT, maxdim = 1)
    barcode_W_PT_QT = barcode(W_PT_QT["eirene_output"], dim = 1)
    barcode_W_QT_PT = barcode(W_QT_PT["eirene_output"], dim = 1)
    
    # select significant points in Witness PD
    n_bars = size(barcode_W_PT_QT, 1)
    if n_bars < 5
        W_cutoff = 0
        W_idx = collect(1:n_bars)
    else
        # select significant points in Witness PD
        W_idx, W_cutoff = select_persistent_intervals_IQR(barcode_W_PT_QT)
    end
    
    # select significant points in Witness PD
    #W_idx, W_cutoff = select_persistent_intervals_IQR(barcode_W_PT_QT)

    # Dowker duality: Find the correspondence between bars
    PT_to_QT = ext_var.apply_Dowker(W_PT_QT, W_QT_PT, dim = 1)
    
    
    # find similarity-centric analogous bars for each significant point in the Witness PD
    analogous_P = Dict()
    analogous_Q = Dict()
    for i in W_idx
        W_PQ_bar = i
        W_QP_bar = PT_to_QT[W_PQ_bar]

        # find cycle, psi in W_P and W_Q
        cycle_W_P, psi_W_P =  ext_var.find_terminal_class_in_W(W_PT_QT, bar = W_PQ_bar)
        cycle_W_Q, psi_W_Q =  ext_var.find_terminal_class_in_W(W_QT_PT, bar = W_QP_bar)

        # Run baseline extensions at epsilon 0
        extension_P = ext_var.run_baseline_extension_W_to_VR_at_epsilon0(W = W_PT_QT, tau = cycle_W_P, psi = psi_W_P, C_VR = VR_P, D_VR = D_P)
        extension_Q = ext_var.run_baseline_extension_W_to_VR_at_epsilon0(W = W_QT_PT, tau = cycle_W_Q, psi = psi_W_Q, C_VR = VR_Q, D_VR = D_Q)

        # update analogous pairs
        if extension_P != nothing
            if extension_P["epsilon_0"] == nothing
                analogous_P[i] = "no Ftau"
            else
                analogous_P[i] =  extension_P["baseline_bar_extension"]
            end
        else
            analogous_P[i] = "boundary matrix error"
        end

        if extension_Q != nothing
            if extension_Q["epsilon_0"] == nothing
                analogous_Q[i] = "no Ftau"
            else
                analogous_Q[i] =  extension_Q["baseline_bar_extension"]
            end
        else
            analogous_Q[i] = "boundary matrix error"
        end
    end
    
    if save_output == true 
        save(output_filename * ".jld2", 
            "analogous_P", analogous_P, 
            "analogous_Q", analogous_Q, 
            "barcode_W_tori", barcode_W_PT_QT,
            "W_selected", W_idx,
            "cutoff", W_cutoff
            )       
    end
    
    return analogous_P, analogous_Q, barcode_W_PT_QT, W_idx, W_cutoff
end


function parse_null_model_stats(output_dir, barcode_VR_P, barcode_VR_Q, selected_P, selected_Q; n_shuffles = 100)

    # go through the output files and count how many times an analogous pair appears
    selected_bar_count = 0
    n_P = size(barcode_VR_P,1)
    n_Q = size(barcode_VR_Q,1)
    stats_matrix = zeros((n_P, n_Q))
    
    no_Ftau = 0
    
    # count the files that actually exist
    valid_files = 0

    # count the total number of bars from which we ran similarity_centric_analogous_bars
    for i = 1:n_shuffles
        
        # count the files that actually exist
        filename = output_dir * string(i) * ".jld2"
        if isfile(filename) == true
            valid_files += 1

            # open julia files
            analogous_P = load(output_dir * string(i) * ".jld2")["analogous_P"]
            analogous_Q = load(output_dir * string(i) * ".jld2")["analogous_Q"]
            selected = load(output_dir * string(i) * ".jld2")["W_selected"]

            for j in selected
                #if analogous_P[j] ==  "no Ftau"
                #    P_no_Ftau += 1
                #end

                #if analogous_Q[j] == "no Ftau"
                #    Q_no_Ftau += 1
                #end

                if (analogous_P[j] ==  "no Ftau") || (analogous_Q[j] == "no Ftau")
                   no_Ftau += 1     
                end

                if (analogous_P[j] != nothing) & (analogous_Q[j] != nothing) & (analogous_P[j] != "boundary matrix error") & (analogous_Q[j] != "boundary matrix error") & (analogous_P[j] !=  "no Ftau") & (analogous_Q[j] !=  "no Ftau")        

                    for p in analogous_P[j]
                        for q in analogous_Q[j]
                            stats_matrix[p,q] += 1
                        end
                    end
                end
            end
            selected_bar_count += size(selected,1) 
        end
    end
    return stats_matrix[selected_P, selected_Q], stats_matrix, selected_bar_count, no_Ftau, valid_files
end

end