
function logdensity(prob, x::AbstractVector)
    return LogDensityProblems.logdensity(prob, x)
end

function logdensity(prob, xs::AbstractMatrix)
    return map(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(xs))
end

function logdensity_and_gradient(prob, x::AbstractVector)
    return LogDensityProblems.logdensity_and_gradient(prob, x)
end

function logdensity_and_gradient(prob, xs::AbstractMatrix)
    ℓπ_and_∇ℓπs = map(
        Base.Fix1(LogDensityProblems.logdensity_and_gradient, prob),
        map(Vector, eachcol(xs)),
    )
    ℓπs = first.(ℓπ_and_∇ℓπs)
    ∇ℓπs = hcat(last.(ℓπ_and_∇ℓπs)...)
    return ℓπs, ∇ℓπs
end

function logdensity_safe(prob, x::AbstractVector)
    ℓπ = LogDensityProblems.logdensity(prob, x)
    return isfinite(ℓπ) ? ℓπ : -Inf
end

function logdensity_safe(prob, xs::AbstractMatrix)
    # ℓπs = if threaded
    #     OhMyThreads.tmap(
    #         Base.Fix1(LogDensityProblems.logdensity, prob),
    #         map(Vector, eachcol(xs))
    #     )
    # else
    #     map(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(xs))
    # end
    ℓπs = logdensity(prob, xs)
    return @. ifelse(isfinite(ℓπs), ℓπs, -Inf)
end

function logdensity_gradient_safe(prob, xs::AbstractMatrix)
    # ℓπ_and_∇ℓπs = if nthreads() > 1
    #     OhMyThreads.tmap(
    #         Base.Fix1(LogDensityProblems.logdensity_and_gradient, prob),
    #         map(Vector, eachcol(xs))
    #     )
    # else
    #     map(Base.Fix1(LogDensityProblems.logdensity_and_gradient, prob), eachcol(xs))
    # end

    ℓπs, ∇ℓπs = logdensity_and_gradient(prob, xs)

    for i in 1:size(∇ℓπs, 2)
        if !isfinite(ℓπs[i])
            ∇ℓπs[:, i] .= Inf
        end
    end
    return ∇ℓπs
end
