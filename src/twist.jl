
function rand_twist_mvnormal(
    rng    ::Random.AbstractRNG,
    twist,
    μs     ::AbstractMatrix,
    Σ      ::Union{<:Diagonal, <:FillArrays.Eye, <:PDMats.ScalMat},
)
    (; a, b,)  = twist
    A          = Diagonal(a)
    K          = Diagonal(inv(2*A + inv(Σ)))
    μs_twisted = K*(Σ\μs .- b)
    μs_twisted + unwhiten(K, randn(rng, size(μs)))
end

function twist_mvnormal_logmarginal(
    twist,
    μ     ::AbstractArray,
    Σ     ::Union{<:Diagonal, <:FillArrays.Eye, <:PDMats.ScalMat},
)
    (; a, b, c)  = twist
    A     = Diagonal(a)
    K     = Diagonal(inv(2*A + inv(Σ)))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z     = Σ\μ .- b
    ((-ℓdetΣ + ℓdetK) .+ (quad(K, z) - invquad(Σ, μ)))/2 .- c
end

# function rand_twist_mvnormal(
#     rng    ::Random.AbstractRNG,
#     twist,
#     μs     ::AbstractMatrix,
#     Σ      ::AbstractMatrix,
# )
#     (; a, b,)  = twist
#     K          = inv(2*Diagonal(a) + inv(PDMats.PDMat(Σ)))
#     μs_twisted = K*(Σ\μs .- b)
#     μs_twisted + unwhiten(K, randn(rng, size(μs)))
# end

# function twist_mvnormal_logmarginal(
#     twist,
#     μ     ::AbstractArray,
#     Σ     ::AbstractMatrix,
# )
#     (; a, b, c)  = twist
#     K     = inv(2*Diagonal(a) + inv(PDMats.PDMat(Σ)))
#     ℓdetΣ = logdet(Σ)
#     ℓdetK = logdet(K)
#     z     = Σ\μ .- b
#     ((-ℓdetΣ + ℓdetK) .+ (quad(K, z) - invquad(Σ, μ)))/2 .- c
# end

function twist_logdensity(twist, x)
    (; a, b, c) = twist
    -quad(Diagonal(a), x) - (b'*x)[1,:] .- c   
end

function fit_quadratic(x::AbstractMatrix{T}, y::AbstractVector{T}) where {T <: Real}
    d, n = size(x,1), size(x,2)
    @assert size(x,2) == length(y)

    X   = vcat(x.^2, x, ones(T, 1, n))' |> Array
    β   = ones(2*d + 1)
    ϵ   = 1e-5 
    Xty = X'*y
    XtX = Hermitian(X'*X)

    func(β_) = sum(abs2, X*β_ - y)
    function grad!(g, β_)
        g[:] = 2*(XtX*β_ - Xty)
    end
    function hess!(H, β_)
        H[:,:] = 2*XtX
    end
    df    = TwiceDifferentiable(func, grad!, hess!, β)

    lower = vcat(fill(ϵ, d), fill(typemin(T), d+1))
    upper = fill(typemax(T), 2*d+1)
    dfc   = TwiceDifferentiableConstraints(lower, upper)
    res   = optimize(df, dfc, β, IPNewton())

    β = Optim.minimizer(res)
    a = β[1:d]
    b = β[d+1:2*d]
    c = β[end]
    
    a, b, c, sum(abs2, X*β - y)
end
