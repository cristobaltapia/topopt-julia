include("topOpt_88.jl")
using .TopOpt88

nx = 200;
ny = 240;
volfrac = 0.15;
p = 3.0;
r_min = nx * 0.03;
#
prob = TopOptProblemConv(nx, ny, r_min, volfrac, p)
line_load!(prob, ny ÷ 3)
fix_dof!(prob, :bottom)
fix_dof!(prob, :right)
symmetry_axis!(prob, :left)
use_elements!(prob, :hline; y=ny ÷ 3)

@time res = minimize(prob, Δ=0.01, filter=3);
