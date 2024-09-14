
function gradient_flow_euler(π, x::AbstractMatrix, h::Real, Γ)
    ∇U = -logdensity_gradient(π, x)
    return x - h * Γ * ∇U
end

function leapfrog(target, x::AbstractMatrix, v::AbstractMatrix, δ::Real, M::AbstractMatrix)
    v′ = v + δ / 2 * logdensity_gradient(target, x)
    x′ = x + δ * (M \ v′)
    v′′ = v′ + δ / 2 * logdensity_gradient(target, x′)
    return x′, v′′
end

struct KLMCKernelCov{S<:Real}
    lxx::S
    lxv::S
    lvv::S
    linvxx::S
    linvxv::S
    linvvv::S
end

struct KLMCKernel{M<:AbstractArray,S<:KLMCKernelCov}
    μx :: M
    μv :: M
    Σ  :: S
end

function klmc_logpdf(k::KLMCKernel, x::AbstractMatrix, v::AbstractMatrix)
    (; μx, μv, Σ) = k
    (; lxx, lvv, linvxx, linvxv, linvvv) = Σ

    d = size(x, 1)

    x_centered = x - μx
    v_centered = v - μv

    x_std = linvxx * x_centered
    v_std = linvxv * x_centered + linvvv * v_centered

    ℓdetΣ = 2 * (d * log(lxx) + d * log(lvv))

    r2x = sum(abs2, x_std; dims=1)[1, :]
    r2v = sum(abs2, v_std; dims=1)[1, :]
    @. (r2x + r2v + ℓdetΣ + d * log(2π)) / -2
end

function klmc_rand(rng::Random.AbstractRNG, k::KLMCKernel)
    (; μx, μv, Σ) = k
    (; lxx, lxv, lvv) = Σ
    n_dims = size(μx, 1)
    n_particles = size(μx, 2)

    ϵx = randn(rng, n_dims, n_particles)
    ϵv = randn(rng, n_dims, n_particles)
    x  = lxx * ϵx + μx
    v  = lxv * ϵx + lvv * ϵv + μv
    return x, v
end

function klmc_cov(stepsize::Real, damping::Real)
    γ, h = damping, stepsize
    η    = exp(-γ * h)

    σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
    σ2xv = 1 / γ * (1 - 2 * η + η^2)
    σ2vv = 1 - η^2

    lxx = sqrt(σ2xx)
    lxv = σ2xv / lxx
    lvv = sqrt(σ2xx * σ2vv - σ2xv * σ2xv) / lxx

    linvxx = 1 / lxx
    linvxv = -lxv / lxx / lvv
    linvvv = 1 / lvv

    return KLMCKernelCov(lxx, lxv, lvv, linvxx, linvxv, linvvv)
end

function klmc_transition_kernel(
    target,
    x::AbstractMatrix,
    v::AbstractMatrix,
    stepsize::Real,
    damping::Real,
    klmc_cov::KLMCKernelCov,
)
    γ, h = damping, stepsize
    η    = exp(-γ * h)
    ∇U   = -logdensity_gradient(target, x)

    μx = x + (1 - η) / γ * v - (h - (1 - η) / γ) / γ * ∇U
    μv = η * v + (1 - η) / γ * ∇U
    return KLMCKernel(μx, μv, klmc_cov)
end
