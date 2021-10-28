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
