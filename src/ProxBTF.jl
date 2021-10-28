module ProxBTF

export pbtfProblem,pbtf,pbsrtfProblem,pbsrtf,visualize

using SparseArrays: SparseMatrixCSC,spdiagm
using Roots: find_zero, Bisection
using Lasso: fit, FusedLasso
using Gurobi: Env, GRBsetdblparam, Optimizer
using Convex: Variable, minimize, sumsquares, norm, solve!, evaluate
import LogDensityProblems.dimension,LogDensityProblems.capabilities
import LogDensityProblems.logdensity_and_gradient,LogDensityProblems.LogDensityOrder
using Parameters: @unpack
using Random: GLOBAL_RNG
using DynamicHMC: mcmc_with_warmup, default_warmup_stages, ProgressMeterReport, NoProgressReport
using DynamicHMC.Diagnostics: summarize_tree_statistics
using DataFrames: DataFrame
using MCMCChains: Chains,summarize,quantile
using Statistics: mean,norm,var
using StatsPlots: plot, scatter!, plot!

include("utils.jl")
include("projectors.jl")
include("model_pbtf.jl")
include("model_pbsrtf.jl")
include("mcmc.jl")
include("visualization.jl")
end # module
