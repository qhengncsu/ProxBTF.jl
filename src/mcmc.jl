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
    results_matrix = reshape([(results.chain...)...], length(results.chain[1]), length(results.chain))
    results_matrix[1:problem.n,:] = problem.TM \ results_matrix[1:problem.n,:]
    results_matrix[end-1:end,:] = exp.(results_matrix[end-1:end,:])
    chn = Chains(transpose(results_matrix),["β[" .* string.(1:problem.n) .*"]"; "σ²";"α"],thin=nthin)
    result_summary = DataFrame(summarize(chn))
    result_quantile = DataFrame(quantile(chn))
    return result_summary,result_quantile,problem,elapse_time
end

function pbsrtf(
    y::Vector{T},                # response
    x::Vector{T},                # predictor
    k::Int,                      # order of desired polynomial
    restriction::String; # shape restriction
    lb::Union{Vector{T},Nothing}=nothing, #lower bound for β
    ub::Union{Vector{T},Nothing}=nothing, #upper bound for β
    scale::Bool=true,            # whether to scale x and y to the range of 0-10
    thinning::Bool=false,        # whether to apply the thinning technique
    nbins::Int=100,              # number of bins if thinning is applied
    λ::Union{T,Nothing}=nothing, # Moreau envelope parameter
    s::T=1e-3,                   # IG(s,r) prior for σ²
    r::T=1e-3,
    μ::Union{T,Nothing}=nothing, # exp(μ) prior for α
    nsample::Int=1000,           # number of MCMC sample iterations
    nthin::Int=1,                # thinning the chain after MCMC sampling
    verbose::Bool=true           # whether to display progress bar
    ) where T<: Real
    xmin = ymin = 0.0
    xscale = yscale = 1.0
    if scale
        xmin = minimum(x)
        ymin = minimum(y)
        xscale = (maximum(x)-minimum(x))/10.0
        yscale = (maximum(y)-minimum(y))/10.0
        y = (y .- ymin)./yscale
        x = (x .- xmin)./xscale
    end
    problem = pbsrtfProblem(y,x,k,restriction,lb=lb,ub=ub,λ=λ,s=s,r=r,μ=μ,thinning=thinning,nbins=nbins)
    if verbose
        elapse_time = @elapsed results = mcmc_with_warmup(GLOBAL_RNG, problem, nsample, initialization=(ϵ=0.01,),
        warmup_stages=default_warmup_stages(;stepsize_search=nothing,
        init_steps=50,middle_steps=50,doubling_stages=3,terminating_steps=50),reporter=ProgressMeterReport())
        println(summarize_tree_statistics(results.tree_statistics))
    else
        elapse_time = @elapsed results = mcmc_with_warmup(GLOBAL_RNG, problem, nsample, initialization=(ϵ=0.01,),
        warmup_stages=default_warmup_stages(;stepsize_search=nothing,
        init_steps=50,middle_steps=50,doubling_stages=3,terminating_steps=50),reporter=NoProgressReport())
    end
    results_matrix = reshape([(results.chain...)...], length(results.chain[1]), length(results.chain))
    results_matrix[end-1:end,:] = exp.(results_matrix[end-1:end,:])
    problem.x = problem.x .*xscale .+ xmin
    problem.y = problem.y .*yscale .+ ymin
    problem.xgrid = problem.xgrid .*xscale .+ xmin
    problem.ybar = problem.ybar .*yscale .+ ymin
    results_matrix[1:end-2,:] .*= yscale
    results_matrix[1:end-2,:] .+= ymin
    results_matrix[end-1,:] .*= yscale^2
    chn = Chains(transpose(results_matrix),["β[" .* string.(1:problem.n) .*"]"; "σ²";"α"],thin=nthin)
    result_summary = DataFrame(summarize(chn))
    result_quantile = DataFrame(quantile(chn))
    return result_summary,result_quantile,problem,elapse_time
end
