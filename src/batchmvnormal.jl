
struct BatchMvNormal{M, S}
    μ::M
    Σ::S
end

function Distributions.logpdf(
    p::BatchMvNormal{<:BatchVectors2, <:BlockPDMat2by2},
    x::BatchVectors2,
)
    (; μ, Σ)   = p
    (; x1, x2) = x
    d     = size(x1, 1)
    ℓdetΣ = logdet(Σ)
    r2    = invquad(Σ, x - μ)
    return @. (r2 + ℓdetΣ + 2 * d * log(2π)) / -2
end

function Base.rand(
    rng::Random.AbstractRNG,
    p::BatchMvNormal{<:BatchVectors2, <:BlockPDMat2by2}
)
    (; μ, Σ) = p
    (; L)    = Σ

    d, n = size(μ.x1, 1), size(μ.x1, 2)

    ϵ1 = randn(rng, d, n)
    ϵ2 = randn(rng, d, n)
    ϵ  = BatchVectors2(ϵ1, ϵ2)
    return L*ϵ + μ
end
