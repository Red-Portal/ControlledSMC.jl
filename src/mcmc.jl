
function gradient_flow_euler_batch(x::AbstractMatrix, ∇ℓπ::AbstractMatrix, h::Real, Γ)
    return x + h * Γ * ∇ℓπ
end
