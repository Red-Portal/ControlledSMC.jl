
function gradient_flow_euler_batch(x::AbstractMatrix, ∇ℓπ::AbstractMatrix, h::Real, Γ)
    return x + h * Γ * ∇ℓπ
end

function leapfrog_batch(
    target,
    x::AbstractMatrix,
    v::AbstractMatrix,
    δ::Real,
    M::AbstractMatrix
)
    v′ = v + δ/2*logdensity_gradient_batch(target, x)
    x′ = x + δ*(M\v′)
    v′′ = v′ + δ/2*logdensity_gradient_batch(target, x′)
    x′, v′′
end
