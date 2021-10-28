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
