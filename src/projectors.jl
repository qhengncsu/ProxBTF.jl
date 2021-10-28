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
