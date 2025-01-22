
function logdensity_safe(prob, x::AbstractVector)
    ℓπ = LogDensityProblems.logdensity(prob, x)
    return isfinite(ℓπ) ? ℓπ : -Inf
end

function logdensity_safe(prob, xs::AbstractMatrix)
    ℓπs = LogDensityProblems.logdensity.(Ref(prob), eachcol(xs))
    return @. ifelse(isfinite(ℓπs), ℓπs, -Inf)
end

function logdensity_gradient_safe(prob, x::AbstractVector)
    ℓπ, ∇ℓπt = LogDensityProblems.logdensity_and_gradient(prob, x)
    return ifelse(isfinite(ℓπ), ∇ℓπt, zero(∇ℓπt))
end

function logdensity_gradient_safe(prob, xs::AbstractMatrix)
    return mapslices(xi -> logdensity_gradient_safe(prob, Vector(xi)), xs; dims=1)
end
