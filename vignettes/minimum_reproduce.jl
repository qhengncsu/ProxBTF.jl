#reproducing part of our results is as easy as
#running this script from start to end in Julia (>=1.6)
using Pkg
Pkg.add("SparseArrays")
Pkg.add("Roots")
Pkg.add(name="Lasso", version="0.6.2")
Pkg.add("LogDensityProblems")
Pkg.add("Parameters")
Pkg.add("Random")
Pkg.add(name="DynamicHMC",version="3.2.1")
Pkg.add("DataFrames")
Pkg.add("MCMCChains")
Pkg.add("Statistics")
Pkg.add("StatsPlots")

using SparseArrays: SparseMatrixCSC,spdiagm
using Roots: find_zero, Bisection
using Lasso: fit, FusedLasso
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

function getD(k::Int,n::Int,x::AbstractArray{T}=[1.:1.:n;]) where T<:Real
    D = spdiagm(n-1,n,0=>fill(-1.,n-1),1=>fill(1.,n-1))
    if k>= 1
        for i in 1:k
            D1 = spdiagm(n-i-1,n-i,0=>fill(-1.,n-i-1),1=>fill(1.,n-i-1))
            diffx = x[(i+1):n]-x[1:(n-i)]
            xdiag = spdiagm(0=> i ./ diffx)
            D = D1*xdiag*D
        end
    end
    D
end

function getTM(k::Int,n::Int,x::Vector{T}=[1.:1.:n;]) where T<:Real
    if (k==1) & (n<=200)
        D = getD(k,n,x)
        return [spdiagm(k+1,n,0=>fill(1.,k+1));D]
    else
        D = getD(k-1,n,x)
        diffx = x[(k+1):n]-x[1:(n-k)]
        xdiag = spdiagm(0=> k ./ diffx)
        return [spdiagm(k,n,0=> fill(1.,k));xdiag*D]
    end
end

function forward_solve!(β::AbstractArray{T},bw::Int,TM::SparseMatrixCSC{T,Int},θ::AbstractArray{T}) where T<:Real
    n = length(θ)
    copyto!(β,θ)
    @inbounds for i in 1:n
        for j in 1:min(i-1,bw-1)
            β[i] -= TM[i,i-j]*β[i-j]
        end
        β[i] /= TM[i,i]
    end
    return zero(T)
end

function back_solve!(∇β::AbstractArray{T},bw::Int,TMᵀ::SparseMatrixCSC{T,Int},x::AbstractArray{T}) where T<:Real
    n = length(x)
    copyto!(∇β,x)
    @inbounds for i in n:-1:1
        for j in 1:min(n-i,bw-1)
            ∇β[i] -= TMᵀ[i,i+j]*∇β[i+j]
        end
        ∇β[i] /= TMᵀ[i,i]
    end
    return zero(T)
end

function thin(xgrid::AbstractArray{T},w::AbstractArray{T},ybar::AbstractArray{T},sse::T,nbins::Int) where T<:Real
    xmin = minimum(xgrid)
    range = maximum(xgrid)-xmin
    gap = range/nbins
    w_new = zeros(T,nbins)
    xgrid_new = zeros(T,nbins)
    ybar_new = zeros(T,nbins)
    sse_new = 0.0
    for i in 1:nbins
        subset = nothing
        if i==1
            subset = xmin .<= xgrid .<= xmin+gap
        else
            subset = xmin+gap*(i-1) .< xgrid .<= xmin+i*gap
        end
        w_new[i] = sum(w[subset])
        if w_new[i]>0
            xgrid_new[i] = sum(w[subset].*xgrid[subset])/w_new[i]
            ybar_new[i] = sum(w[subset].*ybar[subset])/w_new[i]
            sse_new += sum(w[subset].* abs2.(ybar[subset].-ybar_new[i]))
        else
            xgrid_new[i] = xmin+gap*(i-0.5)
            ybar_new[i] = mean(y)
        end
    end
    sse_new += sse
    return xgrid_new,w_new,ybar_new,sse_new
end

function F₀(x::AbstractArray{T}, α::T, λ::T) where T<:Real
    out = zero(T)
    @inbounds for i in 1:length(x)
        δᵢ = max(abs(x[i])-λ,0)
        out += δᵢ
    end
    out -= α+λ
    out
end

function F₁(x::AbstractArray{T}, α::T, λ::T) where T<:Real
    if λ==0
        return sum(abs,diff(x))
    else
        x_prox = fit(FusedLasso,x,λ).β
        return sum(abs,diff(x_prox))-α-λ
    end
end

function absepi_projection!(θ_proj::AbstractArray{T}, skip::Int, θ::AbstractArray{T}, α::T) where T<:Real
    copyto!(θ_proj,θ)
    n = length(θ)
    @views if sum(abs, θ[(skip+1):n]) <= α
        return α
    else
        λstar = find_zero(λ -> F₀(θ[(skip+1):n],α,λ),(0,maximum(abs,θ[(skip+1):n])+max(0,-α)),Bisection())
        @inbounds for i in 1:(n-skip)
            θ_proj[skip+i] = sign(θ[skip+i])*max(abs(θ[skip+i])-λstar,0)
        end
        return α + λstar
    end
end

function diffepi_projection!(θ_proj::AbstractArray{T}, skip::Int, θ::AbstractArray{T}, α::T, inv_DDᵀ_D::AbstractArray{T}) where T<:Real
    copyto!(θ_proj,θ)
    n = length(θ)
    if sum(abs, diff(θ[(skip+1):n])) <= α
        return α
    else
        λₘ = maximum(abs, inv_DDᵀ_D*θ[(skip+1):n])
        λstar = find_zero(λ -> F₁(θ[(skip+1):n],α,λ),(0,λₘ+max(0,-α)),Bisection())
        θ_proj[(skip+1):n] = fit(FusedLasso,θ[(skip+1):n],λstar).β
        return α + λstar
    end
end

mutable struct pbtfProblem{T<:Real}
    y       :: Vector{T}               # response
    x       :: Vector{T}               # predictor
    k       :: Int                     # order of desired polynomial, only 1 or 2 is allowed
    λ       :: Union{T,Nothing}        # Moreau-Yosida envelope parameter
    s       :: T                       # IG(s, r) prior for σ²
    r       :: T
    s₂      :: Union{T,Nothing}        # β'(n-k,s₂) prior for α
    m       :: Int                     # total number of observations
    n       :: Int                     # number of grid locations
    xgrid   :: Vector{T}               # strictly increasing grid locations (length n)
    w       :: Vector{T}               # number of observations at each location (length n)
    ybar    :: Vector{T}               # the average of y at each location (length n)
    sse     :: T                       # sum of squared errors
    TM      :: SparseMatrixCSC{T,Int}  # n×n transformation matrix
    TMᵀ     :: SparseMatrixCSC{T,Int}  # n×n transformation matrix transposed
    D       :: SparseMatrixCSC{T,Int}  # (n-k-1)×n difference matrix
    inv_DDᵀ_D :: Matrix{T}             # used to determine the bisection interval
    θ_proj  :: Vector{T}               # the following arrays are pre-created containers to avoid creating
    β       :: Vector{T}               # new arrays during the course of MCMC sampling
    res     :: Vector{T}
end

function pbtfProblem(y::AbstractArray{T}, x::AbstractArray{T}, k::Int;
                     thinning::Bool=false,nbins::Int=100,λ::Union{T,Nothing}=nothing,
                     s::T=1e-3,r::T=1e-3,s₂::Union{T,Nothing}=nothing) where T<: Real
    if (k!=1) & (k!=2)
        throw(ArgumentError("Currenly only k=1 and k=2 are accepted for pbtf."))
    end
    m = length(x)
    xgrid = sort(unique(x))
    n = length(xgrid)
    w = zeros(T,n)
    ybar = zeros(T,n)
    sse = 0.0
    for i in 1:n
        w[i] = length(x[x.==xgrid[i]])
        ybar[i] = mean(y[x.==xgrid[i]])
        sse += sum(abs2, y[x.==xgrid[i]].-ybar[i])
    end
    if thinning
        xgrid,w,ybar,sse = thin(xgrid,w,ybar,sse,nbins)
        n = nbins
    end
    if λ==nothing
        λ = min(1/n^2,1e-4*var(y))
    end
    if s₂==nothing
        s₂ = sqrt(n)
    end
    TM = getTM(k,n,xgrid)
    TMᵀ = SparseMatrixCSC(TM')
    D1 = getD(0,n-k)
    inv_DDᵀ_D = inv(Matrix(D1*D1'))*D1
    D = getD(k,n,xgrid)
    θ_proj = zeros(T,n)
    β = zeros(T,n)
    res = zeros(T,n)
    pbtfProblem{T}(y,x,k,λ,s,r,s₂,m,n,xgrid,w,ybar,sse,TM,TMᵀ,D,inv_DDᵀ_D,θ_proj,β,res)
end

dimension(problem::pbtfProblem) = problem.n + 2
capabilities(::Type{<:pbtfProblem}) = LogDensityOrder{1}()

function logdensity_and_gradient(problem::pbtfProblem, z)
    @unpack y,x,k,λ,s,r,s₂,m,n,xgrid,w,ybar,sse,TM,TMᵀ,D,inv_DDᵀ_D,θ_proj,β,res=problem
    θ         = view(z,1:n)
    logσ²     = z[n+1]
    σ²        = exp(logσ²)
    logα      = z[n+2]
    α         = exp(logα)
    α_proj    = 0.0
    if (k==1) & (n<=200)
        α_proj = absepi_projection!(θ_proj,k+1,θ,α)
        forward_solve!(β,k+2,TM,θ)
    else
        α_proj = diffepi_projection!(θ_proj,k,θ,α,inv_DDᵀ_D)
        forward_solve!(β,k+1,TM,θ)
    end
    res      .= ybar .- β
    ssm       = sum(res .* w .* res)
    qf        = (ssm+sse+2*r)/(2*σ²)
    dist²     = norm(θ.-θ_proj,2)^2+(α-α_proj)^2
    logl      = -dist²/(2*λ)-qf-(m/2+s)*logσ²
    ∇         = similar(z)
    if (k==1) & (n<=200)
        back_solve!(view(∇,1:n),k+2,TMᵀ,w.*res)
    else
        back_solve!(view(∇,1:n),k+1,TMᵀ,w.*res)
    end
    logl += logα-(n-k+s₂)*log(1+α)
    ∇[n+2]  = 1 -α/λ*(α-α_proj)-(n-k+s₂)*α/(1+α)
    ∇[1:n]  ./= σ²
    ∇[1:n]  .-= (θ.-θ_proj)./λ
    ∇[n+1]    = qf-(m/2+s)
    logl, ∇
end

function pbtf(
    y::Vector{T},                 # response
    x::Vector{T},                 # predictor
    k::Int;                       # order of desired polynomial, only 1 or 2 is acceptable
    thinning::Bool=false,         # whether to apply the thinning technique
    nbins::Int=100,               # number of bins if thinning is applied
    λ::Union{T,Nothing}=nothing,  # the Moreau envelope parameter
    s::T=1e-3,                    # IG(s,r) prior for σ²
    r::T=1e-3,
    s₂::Union{T,Nothing}=nothing, # β'(n-k,s₂) prior for α
    nsample::Int=3000,            # number of MCMC sample iterations
    nthin::Int=5,                 # thinning the chain after MCMC sampling
    verbose::Bool=true            # whether to display progress bar
    ) where T<: Real
    problem = pbtfProblem(y,x,k,thinning=thinning,nbins=nbins,λ=λ,s=s,r=r,s₂=s₂)
    X = [ones(T,problem.n) problem.xgrid]
    q₀ = [problem.TM*X*(X\problem.ybar);0;0]
    if verbose
        elapse_time = @elapsed results = mcmc_with_warmup(GLOBAL_RNG, problem, nsample, initialization=(ϵ=0.01,q=q₀),
        warmup_stages=default_warmup_stages(;stepsize_search=nothing),reporter=ProgressMeterReport())
    else
        elapse_time = @elapsed results = mcmc_with_warmup(GLOBAL_RNG, problem, nsample, initialization=(ϵ=0.01,q=q₀),
        warmup_stages=default_warmup_stages(;stepsize_search=nothing),reporter=NoProgressReport())
    end
    println(summarize_tree_statistics(results.tree_statistics))
    result_matrix = reshape([(results.chain...)...], length(results.chain[1]), length(results.chain))
    result_matrix[1:problem.n,:] = problem.TM \ result_matrix[1:problem.n,:]
    result_matrix[end-1:end,:] = exp.(result_matrix[end-1:end,:])
    chn = Chains(transpose(result_matrix),["β[" .* string.(1:problem.n) .*"]"; "σ²";"α"],thin=nthin)
    result_summary = DataFrame(summarize(chn))
    result_quantile = DataFrame(quantile(chn))
    return result_summary,result_quantile,result_matrix,problem,elapse_time
end

function visualize(result_quantile::DataFrame,problem::pbtfProblem;legend_position::Symbol=:topleft)
    plt=plot(size=(800,600))
    scatter!(problem.x,problem.y,markersize=5,markercolor=:green,label="data",legendfontsize=20,
             xtickfontsize=20,ytickfontsize=20,markerstrokewidth=0,legend=legend_position)
    plot!(subplot=1,problem.xgrid,result_quantile[1:problem.n, Symbol("50.0%")],linewidth=5,label="posterior",c=:blue,
          ribbon=(result_quantile[1:problem.n, Symbol("50.0%")] .- result_quantile[1:problem.n, Symbol("2.5%")],
          result_quantile[1:problem.n, Symbol("97.5%")] .- result_quantile[1:problem.n, Symbol("50.0%")]),fillalpha=0.3)
end

x = [1:1.:100;]
fx = map(xi -> 0 ≤ xi ≤ 35 ? xi : 35 < xi ≤ 70 ? 70-xi : 0.5xi-35, x)
σ = 3.0
y = fx .+  σ.*randn(100)
result_summary,result_quantile,problem,elapse_time = pbtf(y,x,1)
visualize(result_quantile,problem)

x = [1:1.:100;]
σ = 3.0
fx  = 13 .* sin.((4*π/100).*x)
y = fx .+  σ.*randn(100)
result_summary,result_quantile,problem,elapse_time=pbtf(y,x,2)
visualize(result_quantile,problem)
