export line_load!, use_elements!, fix_dof!, symmetry_axis!

function line_load!(prob::TopOptProblem, ind_y::Int)
    nx = prob.nx
    ny = prob.ny
    dim = prob.dofs
    ind = 2 * (ind_y):2 * (ny + 1):dim
    prob.F[ind] .= -1
    nothing
end

function use_elements!(prob::TopOptProblem, x::Int, y::Int)
    prob.passive[x, y] = 2
    nothing
end

function use_elements!(prob::TopOptProblem, x::Array{Int}, y::Array{Int})
    prob.passive[x, y] = 2
    nothing
end

function use_elements!(prob::TopOptProblem, concept::Symbol; y)
    nx = prob.nx
    ny = prob.ny
    dim = prob.dofs

    if concept == :hline
        line_elems = y + 1:ny:nx * ny
        for elem in line_elems
            ix = elem ÷ ny + 1
            iy = elem - ny * (elem ÷ ny)
            prob.passive[iy, ix] = 2
        end
    end
    nothing
end

"""
    fix_dof!(prob::TopOptProblem, dof::Array{Int})

Define which DOFs should be constrained.

# Arguments
- `prob::TopOptProblem`: the TopOpt problem.
- `dof::Array{Int}`: Array with the indices of DOFs that should be constrained.

"""
function fix_dof!(prob::TopOptProblem, dof::Array{Int})
    union!(prob.fixed_dofs, dof)
    nothing
end

function fix_dof!(prob::TopOptProblem, border::Symbol)
    nx = prob.nx
    ny = prob.ny
    dim = prob.dofs

    local fixed::Array{Int}

    if border == :top
        fixed = [1:(2 * (ny + 1)):dim
                 2:(2 * (ny + 1)):dim]
    elseif border == :bottom
        fixed = [(2 * (ny + 1)):(2 * (ny + 1)):dim
                 (2 * (ny + 1) - 1):(2 * (ny + 1)):dim]
    elseif border == :left
        fixed = Array(1:1:(2 * (ny + 1)))
    elseif border == :right
        fixed = Array(dim - ny - 1:1:dim)
    end
    union!(prob.fixed_dofs, fixed)
    nothing
end

"""
    symmetry_axis!(prob::TopOptProblem, border::Symbol)

Apply symmetry conditions along one of the four borders of the problem space.

# Arguments
- `prob::TopOptProblem`: TODO
- `border::Symbol`: one of the following symbols:
                    `:top`, `:bottom`, `:left`, `:right`

"""
function symmetry_axis!(prob::TopOptProblem, border::Symbol)
    nx = prob.nx
    ny = prob.ny
    dim = prob.dofs

    local sym::Array{Int}

    if border == :top
        sym = [1:2 * (ny + 1):dim]
    elseif border == :bottom
        sym = [ny + 1:2 * (ny + 1):dim]
    elseif border == :left
        sym = Array(1:2:(2 * (ny + 1)))
    elseif border == :rigth
        sym = Array(dim - 2 * (ny + 1):2:dim)
    end
    union!(prob.fixed_dofs, sym)
    nothing
end

