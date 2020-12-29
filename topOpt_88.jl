module TopOpt88

include("FastConv.jl/src/FastConv.jl")

using LinearAlgebra, SparseArrays
using Plots, PlotUtils, Statistics
import .FastConv.fastconv

BLAS.set_num_threads(1)

abstract type Filter end
abstract type TopOptProblem end
abstract type TopOptResult end

struct FilterBase <: Filter; end
struct FilterConv <: Filter; end

export TopOptProblemBase
export topopt
export line_load!, fix_dof!, symmetry_axis!
export use_elements!

# Define operator for Kronecker product
⊗(x, y) = kron(x, y)

"""
    fastconv_2(E::Array{T,N}, k::Array{T,N}) where {T,N}

Convolution that returns an array of the same size as the input.

# Arguments
- `E::Array{T`: TODO
- `N}`: TODO
- `k::Array{T`: TODO
- `N}`: TODO

"""
function fastconv_2(E::Array{T,N}, k::Array{T,N}) where {T,N}
    i, j = size(E)
    ki, kj = size(k)

    ret = fastconv(E, k)[ki÷2+1:i+ki÷2, kj÷2+1:j+kj÷2]

    return ret
end

mutable struct FEProblem
    F::Array{Float64}
    U::Array{Float64}
    Kₑ::Array{Float64}
    iK::Array{Integer}
    jK::Array{Integer}
    freedofs::Array{Integer}
end

mutable struct TopOptProblemBase <: TopOptProblem
    nx::Integer
    ny::Integer
    dofs::Integer
    F::Array{Float64}
    passive::Array{Integer,2}
    fixed_dofs::Array{Integer}

    function TopOptProblemBase(nx, ny)
        dofs::Int = 2 * (nx + 1) * (ny + 1)
        F::Array{Float64} = zeros(dofs)
        passive::Array{Integer,2} = zeros(ny, nx)
        fixed_dofs::Array{Integer} = Array([])
        new(nx, ny, dofs, F, passive, fixed_dofs)
    end
end

mutable struct TopOptProblemConv <: TopOptProblem
    nx::Integer
    ny::Integer
    dofs::Integer
    F::Array{Float64}
    passive::Array{Integer,2}
    fixed_dofs::Array{Integer}

    function TopOptProblemConv(nx, ny)
        dofs::Int = 2 * (nx + 1) * (ny + 1)
        F::Array{Float64} = zeros(dofs)
        passive::Array{Integer,2} = zeros(ny, nx)
        fixed_dofs::Array{Integer} = Array([])
        new(nx, ny, dofs, F, passive, fixed_dofs)
    end
end

mutable struct TopOptResultBase{T} <: TopOptResult
    iter::Integer
    obj::Float64
    vol::Float64
    x::T
    x̃::T

    function TopOptResultBase(x)
        iter = 0
        obj = maxintfloat(Float64)
        vol = 1.0
        x̃ = copy(x)
        new{typeof(x)}(iter, obj, vol, x, x̃)
    end
end

mutable struct TopOptResultConv{T} <: TopOptResult
    iter::Integer
    obj::Float64
    vol::Float64
    x::T

    function TopOptResultConv(x)
        iter = 0
        obj = maxintfloat(Float64)
        vol = 1.0
        new{typeof(x)}(iter, obj, vol, x)
    end
end

TopOptResult(x, ::TopOptProblemBase) = TopOptResultBase(x)
TopOptResult(x, ::TopOptProblemConv) = TopOptResultConv(x)

function topopt(prob::TopOptProblem, volfrac, p, rmin; Δ=0.01, filter=1,
                filter_type=1)
    # Material properties
    E₀ = 1.0
    E_min = 1e-9
    ν = 0.3

    nx = prob.nx
    ny = prob.ny
    # Get element stiffness matrix
    Kₑ = element_K(E₀, ν)
    #
    nodenrs = reshape(1:(1 + nx) * (1 + ny), 1 + ny, 1 + nx);
    edofVec = reshape(2 .* nodenrs[1:end - 1,1:end - 1] .+ 1, nx * ny, 1);

    # Array with indices relative to the first dof
    rel_ind = [0 1 (2 * ny .+ [2 3 0 1]) -2 -1]
    edofMat = repeat(edofVec, 1, 8) .+ repeat(rel_ind, nx * ny);

    iK::Array{Int} = reshape((edofMat ⊗ ones(8, 1))', 64 * nx * ny);
    jK::Array{Int} = reshape((edofMat ⊗ ones(1, 8))', 64 * nx * ny);

    # Define loads and supports (half MBB-beam)
    dim = prob.dofs

    F = prob.F
    U = zeros(dim)

    # Passive elements
    passive = prob.passive

    alldofs = Array(1:dim)
    freedofs = setdiff(alldofs, prob.fixed_dofs)

    # Define FE-problem
    fe_prob = FEProblem(prob.F, U, Kₑ, iK, jK, freedofs)

    # Prepare filter
    if filter_type == 1
        H, Hs = set_filter(nx, ny, rmin, FilterBase())
    elseif filter_type == 2
        h, Hs = set_filter(nx, ny, rmin, FilterConv())
    end

    # Initialize iteration
    x = volfrac .* ones(ny, nx)

    # Intatiate result object
    res = TopOptResult(x, prob)

    β = 1
    if filter in [1 2]
        x̄ = copy(x)
        x̃ₑ = copy(x)
    elseif filter == 3
        x̄ = copy(x)
        x̃ₑ = @. 1 - exp(-β * x̄) + x̄ * exp(-β)
    end

    ∂c = similar(x)
    ∂V = zeros(ny, nx)

    loop = 0
    loopbeta = 0
    change = 1.0
    # Start iteration
    while change > Δ
        loop += 1
        loopbeta += 1
        # FE-analysis
        # solve_fe!(fe_prob, prob, res, p, E₀, E_min)
        sK = reshape(Kₑ[:] * (E_min .+ x̃ₑ[:]'.^p .* (E₀ - E_min)), 64 * nx * ny)
        K = sparse(iK, jK, sK)
        U[freedofs] = Symmetric(K[freedofs, freedofs]) \ F[freedofs]

        # Objective function and sensitivity analysis
        cₑ = zeros(ny, nx)
        for ely in 1:ny, elx in 1:nx
            nₑ = ely + ny * (elx - 1)
            cₑ[ely, elx] = sum(U[view(edofMat, nₑ, :)]' * Kₑ .* U[view(edofMat, nₑ, :)]')
        end

        c = sum((E_min .+ x̃ₑ.^p .* (E₀ - E_min)) .* cₑ)
        ∂c[:, :] = -p .* (E₀ - E_min) .* x̃ₑ.^(p - 1) .* cₑ
        ∂V[:, :] = ones(ny, nx);
        # Filtering/modification of sensitivities
        if filter_type == 1
            if filter == 1
                ∂c[:] = H * (x[:] .* ∂c[:]) ./ Hs ./ max.(1e-3, x)
            elseif filter == 2
                ∂c[:] = H * (∂c[:] ./ Hs)
                ∂V[:] = H * (∂V[:] ./ Hs)
            elseif filter == 3
                ∂x = @. β * exp(-β * x̄) + exp(-β)
                ∂c[:] = H * (∂c[:] .* ∂x[:] ./ Hs)
                ∂V[:] = H * (∂V[:] .* ∂x[:] ./ Hs)
            end
            # Optimality criteria update of design variables and physical densities
            x_new, x̃ₑ, x̄ = optimal_crit(nx, ny, x, x̃ₑ, x̄, volfrac, β, ∂c, ∂V, H, Hs,
                                     filter, passive, FilterBase())
        elseif filter_type == 2
            if filter == 1
                ∂c[:] = fastconv_2(∂c .* x̃ₑ, h) ./ Hs ./ max.(1e-3, x̃ₑ)
            elseif filter == 2
                ∂c[:] = fastconv_2(∂c ./ Hs, h)
                ∂V[:] = fastconv_2(∂V ./ Hs, h)
            elseif filter == 3
                ∂x = @. β * exp(-β * x̄) + exp(-β)
                ∂c[:] = fastconv_2(∂c .* ∂x ./ Hs, h)
                ∂V[:] = fastconv_2(∂V .* ∂x ./ Hs, h)
            end
            # Optimality criteria update of design variables and physical densities
            x_new, x̃ₑ, x̄ = optimal_crit(nx, ny, x, x̃ₑ, x̄, volfrac, β, ∂c, ∂V, h, Hs,
                                     filter, passive, FilterConv())
        end
        change = maximum(abs.(x_new - x))
        x = x_new

        if filter == 3 && β < 512 && ((loopbeta ≥ 50) || change ≤ 0.01)
            β = 2 * β
            loopbeta = 0
            change = 1
            println("Parameter β increased to $β")
        end
        # Print results
        vol_x = mean(x̃ₑ)
        println("It. $loop  Obj.: $c  Vol.: $vol_x  ch.: $change")
        # Plot the result
        cmap = cgrad(:greys, rev=true)
        plot_1 = heatmap(x̃ₑ, c=cmap, aspect_ratio=:equal, yflip=true,
                        grid=false, axis=false, colorbar=false)
        display(plot_1)

    end
    # Mirror array to show the full structure
    x_full = [x̃ₑ[:, end:-1:1] x̃ₑ]
    cmap = cgrad(:greys, rev=true)
    plot_1 = heatmap(x_full, c=cmap, aspect_ratio=:equal, yflip=true,
                    grid=false, axis=false, colorbar=false)
    savefig(plot_1, "plot_88.pdf")
    display(plot_1)

end

function set_filter(nx::Int, ny::Int, rmin, ::FilterBase)
    # Prepare filter
    #
    dim_1::Int = nx * ny * (2 * (ceil(rmin) - 1) + 1)^2
    iH::Array{Int} = ones(dim_1)
    jH::Array{Int} = ones(size(iH))
    sH = zeros(size(iH))
    k = 0
    for i1 in 1:nx, j1 in 1:ny
        e1 = (i1 - 1) * ny + j1
        r_1 = max(i1 - (ceil(rmin) - 1), 1)
        r_2 = min(i1 + (ceil(rmin) - 1), nx)
        s_1 = max(j1 - (ceil(rmin) - 1), 1)
        s_2 = min(j1 + (ceil(rmin) - 1), ny)
        for i2 in r_1:r_2, j2 in s_1:s_2
            e2 = (i2 - 1) * ny + j2
            k += 1
            iH[k] = e1
            jH[k] = e2
            sH[k] = max(0, rmin - sqrt((i1 - i2)^2 + (j1 - j2)^2))
        end
    end
    H = sparse(iH, jH, sH);
    Hs = sum(H, dims=2)
    return H, Hs
end

function set_filter(nx::Int, ny::Int, rmin, ::FilterConv)
    kern_size = length(-ceil(rmin) + 1:ceil(rmin) - 1)
    dy = -ceil(rmin) + 1:ceil(rmin) - 1
    dx = -ceil(rmin) + 1:ceil(rmin) - 1

    dy = dy' .* ones(kern_size)
    dx = ones(kern_size)' .* dx

    h = max.(0, rmin .- sqrt.(dx.^2 + dy.^2))
    Hs = fastconv_2(ones(ny, nx), h)

    return h, Hs
end

function optimal_crit(nx::Int, ny::Int, x, x̃ₑ, x̄, volfrac, β, ∂c, ∂V, H, Hs,
                      filter, passive, ::FilterBase)
    λ₁ = 0.0;
    λ₂ = 1e9;
    move = 0.2;
    x_new = similar(x)
    while (λ₂ - λ₁) / (λ₁ + λ₂) > 1e-3
        λₘ = 0.5 * (λ₂ + λ₁);
        # x_new .= max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-∂c ./ ∂V / λₘ)))))
        xₑBₑ = @. x * sqrt(-∂c / ∂V / λₘ)
        x_new .= criteria.(x, move, xₑBₑ)
        if filter == 1
            x̃ₑ = x_new
        elseif filter == 2
            x̃ₑ[:] = (H * view(x_new, :)) ./ Hs
        elseif filter == 3
            x̄[:] = (H * view(x_new, :)) ./ Hs
            x̃ₑ[:] = @. 1 - exp(-β * x̄) + x̄ * exp(-β)
        end

        x̃ₑ[passive .== 1] .= 0
        x̃ₑ[passive .== 2] .= 1

        if sum(x̃ₑ) > volfrac * nx * ny
            λ₁ = λₘ
        else
            λ₂ = λₘ
        end
    end
    return x_new, x̃ₑ, x̄
end

function optimal_crit(nx::Int, ny::Int, x, x̃ₑ, x̄, volfrac, β, ∂c, ∂V, h, Hs,
                      filter, passive, ::FilterConv)
    λ₁ = 0.0;
    λ₂ = 1e9;
    move = 0.2;
    x_new = similar(x)
    while (λ₂ - λ₁) / (λ₁ + λ₂) > 1e-3
        λₘ = 0.5 * (λ₂ + λ₁);
        # x_new .= max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-∂c ./ ∂V / λₘ)))))
        xₑBₑ = @. x * sqrt(-∂c / ∂V / λₘ)
        x_new .= criteria.(x, move, xₑBₑ)
        if filter == 1
            x̃ₑ = x_new
        elseif filter == 2
            x̃ₑ[:] = fastconv_2(x_new, h) ./ Hs
        elseif filter == 3
            x̄[:] = fastconv_2(x_new, h) ./ Hs
            x̃ₑ[:] = @. 1 - exp(-β * x̄) + x̄ * exp(-β)
        end

        x̃ₑ[passive .== 1] .= 0
        x̃ₑ[passive .== 2] .= 1

        if sum(x̃ₑ) > volfrac * nx * ny
            λ₁ = λₘ
        else
            λ₂ = λₘ
        end
    end
    return x_new, x̃ₑ, x̄
end

function criteria(x, move, xₑBₑ)
    if xₑBₑ ≤ max(0.0, x - move)
        x_new = max(0.0, x - move)
    elseif xₑBₑ ≥ min(1.0, x + move)
        x_new = min(1.0, x + move)
    else
        x_new = xₑBₑ
    end
    return x_new
end

function solve_fe!(fe::FEProblem, prob::TopOptProblem, res::TopOptResult, p, E₀, E_min)
    sK = reshape(fe.Kₑ[:] * (E_min .+ res.x̃[:]'.^p .* (E₀ - E_min)), 64 * prob.nx * prob.ny)
    K = sparse(fe.iK, fe.jK, sK)
    fe.U[fe.freedofs] = Symmetric(K[fe.freedofs, fe.freedofs]) \ F[fe.freedofs]
    nothing
end


function element_K(E₀::Float64, ν::Float64)
    A₁₁ = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12]
    A₁₂ = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6]
    B₁₁ = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4]
    B₁₂ = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2]
    Kₑ = 1 / (1 - ν^2) / 24 .* ([A₁₁ A₁₂; A₁₂' A₁₁] + ν * [B₁₁ B₁₂; B₁₂' B₁₁]);
    return Kₑ
end

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


end

using .TopOpt88

nx = 100;
ny = 120;
volfrac = 0.15;
p = 3.0;
r_min = nx * 0.04;

prob = TopOptProblemBase(nx, ny)
line_load!(prob, ny ÷ 3)
fix_dof!(prob, :bottom)
fix_dof!(prob, :right)
# fix_dof!(prob, [2*(ny+1) + 2*(ny+1)*(nx÷2), 2*(ny+1) + 2*(ny+1)*(nx÷2)-1])
# fix_dof!(prob, [2*(ny+1)÷2 + 2*(ny+1)*(nx+1), 2*(ny+1)÷2 + 2*(ny+1)*(nx+1)-1])
symmetry_axis!(prob, :left)
use_elements!(prob, :hline; y=ny ÷ 3)

@time topopt(prob, volfrac, p, r_min, Δ=0.05, filter=2, filter_type=1);
