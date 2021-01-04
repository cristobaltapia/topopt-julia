module TopOpt88

include("FastConv.jl/src/FastConv.jl")

using LinearAlgebra, SparseArrays
using Plots, PlotUtils, Statistics
import .FastConv.fastconv

BLAS.set_num_threads(1)

abstract type Filter end
abstract type TopOptProblem end
abstract type TopOptResult end

include("utils.jl")

struct FilterBase <: Filter; end
struct FilterConv <: Filter; end

export TopOptProblemBase, TopOptProblemConv
export optimize
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

    ret = fastconv(E, k)[ki ÷ 2 + 1:i + ki ÷ 2, kj ÷ 2 + 1:j + kj ÷ 2]

    return ret
end

struct FEProblem
    F::Array{Float64}
    U::Array{Float64}
    Kₑ::Array{Float64}
    iK::Array{Int}
    jK::Array{Int}
    edofMat::Array{Int}
    freedofs::Array{Integer}
    nx::Int
    ny::Int
end

mutable struct TopOptProblemBase <: TopOptProblem
    nx::Int
    ny::Int
    rmin::Float64
    volfrac::Float64
    p::Float64
    dofs::Int
    F::Array{Float64}
    passive::Array{Int,2}
    fixed_dofs::Array{Int}

    function TopOptProblemBase(nx, ny, rmin, volfrac, p)
        dofs::Int = 2 * (nx + 1) * (ny + 1)
        F::Array{Float64} = zeros(dofs)
        passive::Array{Int,2} = zeros(ny, nx)
        fixed_dofs::Array{Int} = Array([])
        new(nx, ny, rmin, volfrac, p, dofs, F, passive, fixed_dofs)
    end
end

mutable struct TopOptProblemConv <: TopOptProblem
    nx::Int
    ny::Int
    rmin::Float64
    volfrac::Float64
    p::Float64
    dofs::Int
    F::Array{Float64}
    passive::Array{Int,2}
    fixed_dofs::Array{Int}

    function TopOptProblemConv(nx, ny, rmin, volfrac, p)
        dofs::Int = 2 * (nx + 1) * (ny + 1)
        F::Array{Float64} = zeros(dofs)
        passive::Array{Int,2} = zeros(ny, nx)
        fixed_dofs::Array{Int} = Array([])
        new(nx, ny, rmin, volfrac, p, dofs, F, passive, fixed_dofs)
    end
end

mutable struct TopOptResultBase{T} <: TopOptResult
    prob::TopOptProblemBase
    iter::Int
    obj::Float64
    vol::Float64
    x::T
    x̃::T
    H::SparseMatrixCSC
    Hs::Array

    function TopOptResultBase(prob, x)
        nx = prob.nx
        ny = prob.ny
        iter = 0
        obj = maxintfloat(Float64)
        vol = 1.0
        x̃ = copy(x)
        H = sparse([], [], [], nx*ny, nx*ny)
        Hs = Array{Float64}([])
        new{typeof(x)}(prob, iter, obj, vol, x, x̃, H, Hs)
    end
end

mutable struct TopOptResultConv{T} <: TopOptResult
    prob::TopOptProblemConv
    iter::Int
    obj::Float64
    vol::Float64
    x::T
    x̃::T
    x̂::T
    Hs::Array
    h::Array

    function TopOptResultConv(prob, x)
        nx = prob.nx
        ny = prob.ny
        iter = 0
        obj = maxintfloat(Float64)
        vol = 1.0
        x̃ = copy(x)
        x̂ = copy(x)
        Hs = Array{Float64}([])
        kern_size = length(-ceil(prob.rmin) + 1:ceil(prob.rmin) - 1)
        h = zeros(kern_size, kern_size)
        new{typeof(x)}(prob, iter, obj, vol, x, x̃, x̂, Hs, h)
    end
end

TopOptResult(prob::TopOptProblemBase, x) = TopOptResultBase(prob, x)
TopOptResult(prob::TopOptProblemConv, x) = TopOptResultConv(prob, x)

function optimize(prob::TopOptProblem; Δ=0.01, filter=1)
    # Material properties
    E₀ = 1.0
    E_min = 1e-9
    ν = 0.3

    nx::Int = prob.nx
    ny::Int = prob.ny
    rmin = prob.rmin
    #
    nodenrs::Array{Int} = reshape(1:(1 + nx) * (1 + ny), 1 + ny, 1 + nx);
    edofVec::Array{Int} = reshape(2 .* nodenrs[1:end - 1,1:end - 1] .+ 1, nx * ny, 1);

    # Array with indices relative to the first dof
    rel_ind = [0 1 (2 * ny .+ [2 3 0 1]) -2 -1]
    edofMat = repeat(edofVec, 1, 8) .+ repeat(rel_ind, nx * ny);

    iK::Array{Int} = reshape((edofMat ⊗ ones(8, 1))', 64 * nx * ny);
    jK::Array{Int} = reshape((edofMat ⊗ ones(1, 8))', 64 * nx * ny);

    dim = prob.dofs
    # Define loads and supports (half MBB-beam)
    alldofs = Array(1:dim)
    freedofs = setdiff(alldofs, prob.fixed_dofs)

    # Get element stiffness matrix
    Kₑ = element_K(E₀, ν)
    F = prob.F
    U = zeros(dim)
    # Define FE-problem
    fe_prob = FEProblem(prob.F, U, Kₑ, iK, jK, edofMat, freedofs, nx, ny)
    # Intatiate result object
    x = prob.volfrac .* ones(ny, nx)
    res = TopOptResult(prob, x)
    # Prepare filter
    set_filter!(res, prob, rmin)

    # Initial value for β
    β = 1.0

    # Initilaize sensitivities
    ∂c = similar(x)
    ∂V = zeros(ny, nx)

    loop = 0
    loopbeta = 0
    change = 1.0
    p = prob.p

    # Start iteration
    while change > Δ
        loop += 1
        loopbeta += 1
        # FE-analysis
        solve_fe!(fe_prob, res.x̃, p, E₀, E_min)
        # Objective function and sensitivity analysis
        cₑ = compliance(fe_prob, E₀, E_min)
        c = sum((E_min .+ res.x̃.^p .* (E₀ - E_min)) .* cₑ)
        # Update ∂c and ∂V
        sensitivities!(∂c, ∂V, fe_prob, res, β, p, E₀, E_min, cₑ, filter)
        # Optimality criteria update of design variables and physical densities
        # and filtering/modification of sensitivities
        x_new = optimal_crit(prob, res, β, ∂c, ∂V, filter)
        # Compute the hange
        change = maximum(abs.(x_new - res.x))
        # Update result
        res.x .= x_new
        # Update β parameter
        if filter == 3 && β < 512 && ((loopbeta ≥ 50) || change ≤ 0.01)
            β = 2 * β
            loopbeta = 0
            change = 1
            println("Parameter β increased to $β")
        end
        # Print results
        vol_x = mean(res.x̃)
        println("It. $loop  Obj.: $c  Vol.: $vol_x  ch.: $change")
        # Plot the result
        cmap = cgrad(:greys, rev=true)
        plot_1 = heatmap(res.x̃, c=cmap, aspect_ratio=:equal, yflip=true,
                        grid=false, axis=false, colorbar=false)
        display(plot_1)

    end
    # Mirror array to show the full structure
    x_full = [res.x̃[:, end:-1:1] res.x̃]
    cmap = cgrad(:greys, rev=true)
    plot_1 = heatmap(x_full, c=cmap, aspect_ratio=:equal, yflip=true,
                    grid=false, axis=false, colorbar=false)
    savefig(plot_1, "plot_88.pdf")
    display(plot_1)
    return res
end

function set_filter!(res::TopOptResult, prob::TopOptProblemBase, rmin)
    nx = prob.nx
    ny = prob.ny
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
    H = sparse(iH, jH, sH)
    res.H = H
    res.Hs = sum(H, dims=2)
    nothing
end

function set_filter!(res::TopOptResult, prob::TopOptProblemConv, rmin)
    nx = prob.nx
    ny = prob.ny

    kern_size = length(-ceil(rmin) + 1:ceil(rmin) - 1)
    dy = -ceil(rmin) + 1:ceil(rmin) - 1
    dx = -ceil(rmin) + 1:ceil(rmin) - 1

    dy = dy' .* ones(kern_size)
    dx = ones(kern_size)' .* dx

    res.h = max.(0, rmin .- sqrt.(dx.^2 + dy.^2))
    res.Hs = fastconv_2(ones(ny, nx), res.h)
    nothing
end

# Optimal criterion for the normal filter
function optimal_crit(prob::TopOptProblemBase, res::TopOptResult, β::Float64,
                      ∂c::Array{Float64}, ∂V::Array{Float64}, filter)
    nx = prob.nx
    ny = prob.ny
    λ₁::Float64 = 0.0
    λ₂::Float64 = 1e9
    move::Float64 = 0.2
    x_new = similar(res.x)
    tot_volfrac = porb.volfrac * nx * ny

    while (λ₂ - λ₁) / (λ₁ + λ₂) > 1e-3
        λₘ = 0.5 * (λ₂ + λ₁);
        xₑBₑ = @. res.x * sqrt(-∂c / ∂V / λₘ)
        x_new .= criteria.(res.x, move, xₑBₑ)
        if filter == 1
            res.x̃ = x_new
        elseif filter == 2
            res.x̃[:] = (res.H * view(x_new, :)) ./ res.Hs
        elseif filter == 3
            res.x̂[:] = (res.H * view(x_new, :)) ./ res.Hs
            res.x̃[:] = @. 1 - exp(-β * res.x̂) + res.x̂ * exp(-β)
        end

        # Consider passibe elements
        res.x̃[prob.passive .== 1] .= 0
        res.x̃[prob.passive .== 2] .= 1

        if sum(res.x̃) > tot_volfrac
            λ₁ = λₘ
        else
            λ₂ = λₘ
        end
    end

    return x_new
end

# Optimal criterion using convolution
function optimal_crit(prob::TopOptProblemConv, res::TopOptResult, β::Float64,
                      ∂c::Array{Float64}, ∂V::Array{Float64}, filter)
    nx = prob.nx
    ny = prob.ny
    λ₁::Float64 = 0.0
    λ₂::Float64 = 1e9
    move::Float64 = 0.2
    x_new = similar(res.x)
    tot_volfrac = prob.volfrac * nx * ny

    while (λ₂ - λ₁) / (λ₁ + λ₂) > 1e-3
        λₘ = 0.5 * (λ₂ + λ₁);
        # x_new .= max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-∂c ./ ∂V / λₘ)))))
        xₑBₑ = @. res.x * sqrt(-∂c / ∂V / λₘ)
        x_new .= criteria.(res.x, move, xₑBₑ)
        if filter == 1
            res.x̃ = x_new
        elseif filter == 2
            res.x̃[:] = fastconv_2(x_new, res.h) ./ res.Hs
        elseif filter == 3
            res.x̂[:] = fastconv_2(x_new, res.h) ./ res.Hs
            res.x̃[:] = @. 1 - exp(-β * res.x̂) + res.x̂ * exp(-β)
        end

        res.x̃[prob.passive .== 1] .= 0
        res.x̃[prob.passive .== 2] .= 1

        if sum(res.x̃) > tot_volfrac
            λ₁ = λₘ
        else
            λ₂ = λₘ
        end
    end

    return x_new
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

function solve_fe!(fe::FEProblem, x̃::Array, p, E₀, E_min)
    sK = reshape(fe.Kₑ[:] * (E_min .+ x̃[:]'.^p .* (E₀ - E_min)), 64 * fe.nx * fe.ny)
    # Global stiffness matrix
    K = sparse(fe.iK, fe.jK, sK)

    freedofs = fe.freedofs
    F = fe.F
    U = fe.U
    # Solve system
    fe.U[freedofs] = Symmetric(K[freedofs, freedofs]) \ F[freedofs]
    nothing
end

function compliance(fe::FEProblem, E₀, E_min)
    nx = fe.nx
    ny = fe.ny
    edofMat = fe.edofMat

    cₑ = zeros(ny, nx)

    for ely in 1:ny, elx in 1:nx
        nₑ = ely + ny * (elx - 1)
        cₑ[ely, elx] = sum(fe.U[view(edofMat, nₑ, :)]' * fe.Kₑ .* fe.U[view(edofMat, nₑ, :)]')
    end

    return cₑ
end

function sensitivities!(∂c, ∂V, fe::FEProblem, res::TopOptResultBase, β, p, E₀, E_min, cₑ, filter)
    nx = fe.nx
    ny = fe.ny
    Hs = res.Hs
    H = res.H

    ∂c .= -p .* (E₀ - E_min) .* res.x̃.^(p - 1) .* cₑ
    ∂V .= ones(ny, nx);
    # Filtering/modification of sensitivities
    if filter == 1
        ∂c[:] = H * (res.x[:] .* ∂c[:]) ./ Hs ./ max.(1e-3, res.x)
    elseif filter == 2
        ∂c[:] = H * (∂c[:] ./ Hs)
        ∂V[:] = H * (∂V[:] ./ Hs)
    elseif filter == 3
        ∂x = @. β * exp(-β * x̄) + exp(-β)
        ∂c[:] = H * (∂c[:] .* ∂x[:] ./ Hs)
        ∂V[:] = H * (∂V[:] .* ∂x[:] ./ Hs)
    end

    return ∂c, ∂V
end

function sensitivities!(∂c, ∂V, fe::FEProblem, res::TopOptResultConv, β, p, E₀, E_min, cₑ, filter)

    nx = fe.nx
    ny = fe.ny
    Hs = res.Hs
    h = res.h

    ∂c .= -p .* (E₀ - E_min) .* res.x̃.^(p - 1) .* cₑ
    ∂V .= ones(ny, nx);
    # Filtering/modification of sensitivities
    if filter == 1
        ∂c[:] = fastconv_2(∂c .* res.x̃, res.h) ./ res.Hs ./ max.(1e-3, res.x̃)
    elseif filter == 2
        ∂c[:] = fastconv_2(∂c ./ res.Hs, res.h)
        ∂V[:] = fastconv_2(∂V ./ res.Hs, res.h)
    elseif filter == 3
        ∂x = @. β * exp(-β * res.x̂) + exp(-β)
        ∂c[:] = fastconv_2(∂c .* ∂x ./ res.Hs, res.h)
        ∂V[:] = fastconv_2(∂V .* ∂x ./ res.Hs, res.h)
    end

    return ∂c, ∂V
end

function element_K(E₀::Float64, ν::Float64)
    A₁₁ = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12]
    A₁₂ = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6]
    B₁₁ = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4]
    B₁₂ = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2]
    Kₑ = 1 / (1 - ν^2) / 24 .* ([A₁₁ A₁₂; A₁₂' A₁₁] + ν * [B₁₁ B₁₂; B₁₂' B₁₁]);
    return Kₑ
end

end
