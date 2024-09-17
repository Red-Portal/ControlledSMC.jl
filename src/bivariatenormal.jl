
struct BivariateMvNormal{M,L}
    μ1::M
    μ2::M
    L11::L
    L12::L
    L22::L
    Linv11::L
    Linv12::L
    Linv22::L
end

function bivariate_logpdf(
    p::BivariateMvNormal{<:AbstractArray,<:Real}, x1::AbstractArray, x2::AbstractArray
)
    (; μ1, μ2, L11, L22, Linv11, Linv12, Linv22) = p

    d = size(x1, 1)

    x1_centered = x1 - μ1
    x2_centered = x2 - μ2

    x1_std = Linv11 * x1_centered
    x2_std = Linv12 * x1_centered + Linv22 * x2_centered

    ℓdetΣ = 2 * (d * log(L11) + d * log(L22))

    r21 = sum(abs2, x1_std; dims=1)[1, :]
    r22 = sum(abs2, x2_std; dims=1)[1, :]
    @. (r21 + r22 + ℓdetΣ + 2 * d * log(2π)) / -2
end

function bivariate_logpdf(
    p::BivariateMvNormal{<:AbstractArray,<:Diagonal}, x1::AbstractArray, x2::AbstractArray
)
    (; μ1, μ2, L11, L22, Linv11, Linv12, Linv22) = p

    d = size(x1, 1)

    x1_centered = x1 - μ1
    x2_centered = x2 - μ2

    x1_std = Linv11 * x1_centered
    x2_std = Linv12 * x1_centered + Linv22 * x2_centered

    ℓdetΣ = 2 * (logdet(L11) + logdet(L22))

    r21 = sum(abs2, x1_std; dims=1)[1, :]
    r22 = sum(abs2, x2_std; dims=1)[1, :]
    @. (r21 + r22 + ℓdetΣ + 2 * d * log(2π)) / -2
end

function bivariate_rand(rng::Random.AbstractRNG, p::BivariateMvNormal)
    (; μ1, μ2, L11, L12, L22) = p
    d, n = size(μ1, 1), size(μ1, 2)

    ϵ1 = randn(rng, d, n)
    ϵ2 = randn(rng, d, n)
    x1 = L11 * ϵ1 + μ1
    x2 = L12 * ϵ1 + L22 * ϵ2 + μ2
    return x1, x2
end
