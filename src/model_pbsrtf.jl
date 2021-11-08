mutable struct pbsrtfProblem{T<:Real}
    y       :: Vector{T}              # response
    x       :: Vector{T}              # design
    k       :: Int
    restriction :: String             # 'increasing','convex','decreasing','concave'
                                      # 'inc-convex','inc-concave','dec-convex','dec-concave'
    lb      :: Union{Vector{T},Nothing} # lower bound vector for β
    ub      :: Union{Vector{T},Nothing} # upper bound vector for β
    λ       :: Union{T,Nothing}       # Moreau-Yosida envelope parameter
    s       :: T                      # IG(s, r) prior for σ²
    r       :: T
    μ       :: Union{T,Nothing}       # exp(μ) prior for α
    m       :: Int                    # number of observations
    n       :: Int                    # number of grid locations
    xgrid   :: Vector{T}              # strictly increasing grid locations (length n)
    w       :: Vector{T}              # number of observations at each location (length n)
    ybar    :: Vector{T}              # the average of y at each location (length n)
    sse     :: T                      # sum of squared errors
    D¹      :: SparseMatrixCSC{T,Int}
    D²      :: SparseMatrixCSC{T,Int}
    D       :: SparseMatrixCSC{T,Int}
    GRB_ENV :: Any                    # gurobi environment
end

function pbsrtfProblem(y::Vector{T}, x::Vector{T}, k::Int, restriction::String;
                       lb::Union{Vector{T},Nothing}=nothing,
                       ub::Union{Vector{T},Nothing}=nothing,
                       thinning::Bool=false,nbins::Int=100,λ::Union{T,Nothing}=nothing,
                       s::T=1e-3,r::T=1e-3,μ::Union{T,Nothing}=nothing) where T<: Real
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
    if λ == nothing
        λ = var(y)*1e-3
    end
    if μ == nothing
       μ = 3.0
    end
    D¹ = getD(0,n,xgrid)
    D² = getD(1,n,xgrid)
    D  = getD(k,n,xgrid)
    GRB_ENV = Env()
    GRBsetdblparam(GRB_ENV,"FeasibilityTol",1e-6)
    GRBsetdblparam(GRB_ENV,"OptimalityTol",1e-6)
    pbsrtfProblem{T}(y,x,k,restriction,lb,ub,λ,s,r,μ,m,n,xgrid,w,ybar,sse,D¹,D²,D,GRB_ENV)
end

dimension(problem::pbsrtfProblem) = problem.n + 2
capabilities(::Type{<:pbsrtfProblem}) = LogDensityOrder{1}()

function logdensity_and_gradient(problem::pbsrtfProblem, z)
    @unpack y,x,k,restriction,lb,ub,λ,s,r,μ,m,n,xgrid,w,ybar,sse,D¹,D²,D,GRB_ENV=problem
    β        = view(z,1:n)
    logσ²    = z[n+1]
    σ²       = exp(logσ²)
    logα     = z[n+2]
    α        = exp(logα)
    η = Variable(n)
    t = Variable()
    #p_gurobi = minimize(sumsquares(β-η)+sumsquares(α-t),norm(D*η,1)<=t)
    if restriction=="increasing"
        p_gurobi = minimize(sumsquares(β-η)+sumsquares(α-t)+1000*norm(pos(-D¹*η),1),norm(D*η,1)<=t)
        #p_gurobi.constraints += D¹*η>=0
    elseif restriction=="decreasing"
        p_gurobi.constraints += D¹*η<=0
    elseif restriction=="convex"
        p_gurobi.constraints += D²*η>=0
    elseif restriction=="concave"
        p_gurobi.constraints += D²*η<=0
    elseif restriction=="inc-convex"
        p_gurobi.constraints += [D¹*η>=0,D²*η>=0]
    elseif restriction=="dec-convex"
        p_gurobi.constraints += [D¹*η<=0,D²*η>=0]
    elseif restriction=="inc-concave"
        p_gurobi.constraints += [D¹*η>=0,D²*η<=0]
    elseif restriction=="dec-concave"
        p_gurobi.constraints += [D¹*η<=0,D²*η<=0]
    end
    if !isnothing(lb)
        p_gurobi.constraints += η>=lb
    end
    if !isnothing(ub)
        p_gurobi.constraints += η<=ub
    end
    solve!(p_gurobi,Optimizer(GRB_ENV),silent_solver=true,warmstart=true)
    #while p_gurobi.status!=Convex.MOI.OPTIMAL
        #β = β .+ 1e-3 .*randn(n)
        #p_gurobi.objective = sumsquares(β-η)+sumsquares(α-t)
        #solve!(p_gurobi,Gurobi.Optimizer(GRB_ENV),silent_solver=true,warmstart=true)
        #println("Warning, solution failed!")
    #end
    β_proj = evaluate(η)
    α_proj = evaluate(t)
    res      = ybar .- β
    ssm      = sum(abs2.(res).*w)
    qf       = (ssm+sse+2*r)/(2*σ²)
    logl     = -qf-(m/2+s)*logσ²+logα-μ*α
    dist²    = norm(β.-β_proj)^2+(α-α_proj)^2
    logl    -= dist²/(2*λ)
    ∇        = similar(z)
    ∇[1:n]   = (w.*res)./σ².-((β.-β_proj)./λ)
    ∇[n+1]   = qf-(m/2+s)
    ∇[n+2]   = 1-α/λ*(α-α_proj)-μ*α
    logl, ∇
end
