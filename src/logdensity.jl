
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
    return ifelse(isfinite(ℓπ), ∇ℓπt, Fill(Inf, length(∇ℓπt)))
end

function logdensity_gradient_safe(prob, xs::AbstractMatrix)
    return mapslices(xi -> logdensity_gradient_safe(prob, Vector(xi)), xs; dims=1)
end

function ChainRulesCore.rrule(::typeof(logdensity_safe), prob, x::AbstractVector)
    ℓπ, ∇ℓπt = LogDensityProblems.logdensity_gradient_safe(prob, x)
    function logdensity_safe_pullback(ℓπ_bar)
        return (NoTangent(), NoTangent(), ∇ℓπt*ℓπ_bar)
    end
    return ℓπ, logdensity_safe_pullback
end

function ChainRulesCore.rrule(::typeof(logdensity_safe), prob, xs::AbstractMatrix)
    tups   = LogDensityProblems.logdensity_and_gradient.(Ref(prob), eachcol(xs))
    ℓπs   = [first(tup) for tup in tups]
    ∇ℓπs = hcat([last(tup) for tup in tups]...)

    function logdensity_safe_pullback(ℓπs_bar)
        return (NoTangent(), NoTangent(), reshape(ℓπs_bar, (1, length(ℓπs))) .* ∇ℓπs)
    end
    return ℓπs, logdensity_safe_pullback
end

function ChainRulesCore.rrule(::typeof(logdensity_gradient_safe), prob, x::AbstractVector)
    _, ∇ℓπ, ∇2ℓπ = LogDensityProblems.logdensity_gradient_and_hessian(prob, x)
    function logdensity_gradient_safe_pullback(∇ℓπs_bar)
        return (NoTangent(), NoTangent(), ∇2ℓπ*∇ℓπs_bar)
    end
    return ∇ℓπ, logdensity_gradient_safe_pullback
end

function ChainRulesCore.rrule(::typeof(logdensity_gradient_safe), prob, xs::AbstractMatrix)
    tups = LogDensityProblems.logdensity_gradient_and_hessian.(Ref(prob), eachcol(xs))
    ∇ℓπs = hcat([tup[2] for tup in tups]...)
    function logdensity_gradient_safe_pullback(∇ℓπs_bar)
        hessvec = mapreduce(hcat, enumerate(tups)) do (idx, tup)
            tup[3]*∇ℓπs_bar[:,idx]
        end
        return (NoTangent(), NoTangent(), hessvec)
    end
    return ∇ℓπs, logdensity_gradient_safe_pullback
end
