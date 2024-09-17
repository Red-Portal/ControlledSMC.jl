
struct BivariateMvNormal{M, Chol}
    μ1  ::M
    μ2  ::M
    L   ::Chol
    Linv::Chol
end

function bivariate_logpdf(
    p::BivariateMvNormal{<:AbstractArray, <:BlockCholesky2by2{<:Real}},
    x1::AbstractArray,
    x2::AbstractArray
)
    (; μ1, μ2, Linv)       = p
    Linv11, Linv12, Linv22 = Linv.L11, Linv.L12, Linv.L22

    d = size(x1, 1)

    x1_centered = x1 - μ1
    x2_centered = x2 - μ2

    x1_std = Linv11 * x1_centered
    x2_std = Linv12 * x1_centered + Linv22 * x2_centered

    ℓdetΣ = 2 * (d * -log(Linv11) + d * -log(Linv22))

    r21 = sum(abs2, x1_std; dims=1)[1, :]
    r22 = sum(abs2, x2_std; dims=1)[1, :]
    @. (r21 + r22 + ℓdetΣ + 2 * d * log(2π)) / -2
end

function bivariate_logpdf(
    p::BivariateMvNormal{<:AbstractArray,<:BlockCholesky2by2{<:Diagonal}},
    x1::AbstractArray,
    x2::AbstractArray
)
    (; μ1, μ2, Linv)       = p
    Linv11, Linv12, Linv22 = Linv.L11, Linv.L12, Linv.L22

    d = size(x1, 1)

    x1_centered = x1 - μ1
    x2_centered = x2 - μ2

    x1_std = Linv11 * x1_centered
    x2_std = Linv12 * x1_centered + Linv22 * x2_centered

    ℓdetΣ = 2 * (-logdet(Linv11) + -logdet(Linv22))

    r21 = sum(abs2, x1_std; dims=1)[1, :]
    r22 = sum(abs2, x2_std; dims=1)[1, :]
    @. (r21 + r22 + ℓdetΣ + 2 * d * log(2π)) / -2
end

function bivariate_rand(
    rng::Random.AbstractRNG,
    p::BivariateMvNormal{<:AbstractArray,<:BlockCholesky2by2}
)
    (; μ1, μ2, L) = p
    L11, L12, L22 = L.L11, L.L12, L.L22
    d, n = size(μ1, 1), size(μ1, 2)

    ϵ1 = randn(rng, d, n)
    ϵ2 = randn(rng, d, n)
    x1 = L11 * ϵ1 + μ1
    x2 = L12 * ϵ1 + L22 * ϵ2 + μ2
    return x1, x2
end
