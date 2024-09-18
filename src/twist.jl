
function twist_mvnormal_rand(
    rng::Random.AbstractRNG,
    twist,
    μs::AbstractMatrix,
    Σ::Union{<:Diagonal,<:FillArrays.Eye,<:PDMats.ScalMat},
)
    (; a, b)   = twist
    A          = Diagonal(a)
    K          = Diagonal(inv(2 * A + inv(Σ)))
    μs_twisted = K * (Σ \ μs .- b)
    return μs_twisted + unwhiten(K, randn(rng, size(μs)))
end

function twist_mvnormal_logmarginal(
    twist, μ::AbstractArray, Σ::Union{<:Diagonal,<:FillArrays.Eye,<:PDMats.ScalMat}
)
    (; a, b, c) = twist
    A = Diagonal(a)
    K = Diagonal(inv(2 * A + inv(Σ)))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z = Σ \ μ .- b
    return ((-ℓdetΣ + ℓdetK) .+ (PDMats.quad(K, z) - PDMats.invquad(Σ, μ))) / 2 .- c
end

function twist_bivariate_mvnormal_logmarginal(
    twist,
    μ::AbstractArray,
    Σ::Union{<:Diagonal,<:FillArrays.Eye,<:PDMats.ScalMat}
)
    (; a, b, c) = twist
    A = Diagonal(a)
    K = Diagonal(inv(2 * A + inv(Σ)))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z = Σ \ μ .- b
    return ((-ℓdetΣ + ℓdetK) .+ (PDMats.quad(K, z) - PDMats.invquad(Σ, μ))) / 2 .- c
end

function twist_logdensity(twist, x)
    (; a, b, c) = twist
    return -PDMats.quad(Diagonal(a), x) - (b' * x)[1, :] .- c
end
