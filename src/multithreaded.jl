
struct MultithreadedLogDensityProblem{Prob}
    prob::Prob
end

function logdensity(tprob::MultithreadedLogDensityProblem, xs::AbstractMatrix)
    return OhMyThreads.tmap(Base.Fix1(logdensity, tprob.prob), eachcol(xs))
end

function logdensity_and_gradient(tprob::MultithreadedLogDensityProblem, xs::AbstractMatrix)
    ℓπ_and_∇ℓπs = OhMyThreads.tmap(
        Base.Fix1(LogDensityProblems.logdensity_and_gradient, tprob.prob),
        map(Vector, eachcol(xs)),
    )
    ℓπs = first.(ℓπ_and_∇ℓπs)
    ∇ℓπs = last.(ℓπ_and_∇ℓπs)
    return ℓπs, hcat(∇ℓπs...)
end
