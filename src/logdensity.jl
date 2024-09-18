
function LogDensityProblems.logdensity(prob, xs::AbstractMatrix)
    return LogDensityProblems.logdensity.(Ref(prob), eachcol(xs))
end

function logdensity_gradient(prob, x::AbstractVector)
    _, ∇ℓπt = LogDensityProblems.logdensity_and_gradient(prob, x)
    return ∇ℓπt
end

function logdensity_gradient(prob, xs::AbstractMatrix)
    return mapslices(xi -> logdensity_gradient(prob, Vector(xi)), xs; dims=1)
end
