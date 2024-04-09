module eirene_helper

include("Eirene_var.jl")

using HDF5
using .Eirene_var
export find_single_connected_cycle,
    convert_classrep,
    save_classreps,
    vector_to_symmetric_matrix

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

end