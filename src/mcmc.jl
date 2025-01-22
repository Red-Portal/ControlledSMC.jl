
function gradient_flow_euler(π, x::AbstractMatrix, h::Real, Γ)
    ∇U = -logdensity_gradient_safe(π, x)
    return x - h * Γ * ∇U
end

function leapfrog(target, x::AbstractMatrix, v::AbstractMatrix, δ::Real, M::AbstractMatrix)
    v′ = v + δ / 2 * logdensity_gradient_safe(target, x)
    x′ = x + δ * (M \ v′)
    v′′ = v′ + δ / 2 * logdensity_gradient_safe(target, x′)
    return x′, v′′
end

function klmc_mean(
    target, x::AbstractArray, v::AbstractArray, stepsize::Real, damping::Real
)
    γ, h = damping, stepsize
    η    = exp(-γ * h)
    ∇U   = -logdensity_gradient_safe(target, x)
    μx   = x + (1 - η) / γ * v - (h - (1 - η) / γ) / γ * ∇U
    μv   = η * v - (1 - η) / γ * ∇U
    return BatchVectors2(μx, μv)
end

function klmc_cov(d::Int, stepsize::Real, damping::Real)
    γ, h = damping, stepsize
    η    = exp(-γ * h)
    σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
    σ2xv = 1 / γ * (1 - 2 * η + η^2)
    σ2vv = 1 - η^2
    return BlockHermitian2by2(
        Diagonal(fill(σ2xx, d)), Diagonal(fill(σ2xv, d)), Diagonal(fill(σ2vv, d))
    )
end
