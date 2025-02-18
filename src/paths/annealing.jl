
abstract type AbstractAnnealing end

struct GeometricAnnealing{G<:Real} <: AbstractAnnealing
    gamma::G
end

anneal(a::GeometricAnnealing, x, y) = (1 - a.gamma) * x + a.gamma * y

struct AnnealedDensityProblem{
    A<:AbstractAnnealing,Q<:ContinuousMultivariateDistribution,P,AD<:ADTypes.AbstractADType
}
    annealing :: A
    proposal  :: Q
    problem   :: P
    adtype    :: AD
end

function logdensity(prob::AnnealedDensityProblem, xs::AbstractMatrix)
    (; annealing, proposal, problem) = prob
    ℓπ0s = map(Base.Fix1(logpdf, proposal), eachcol(xs))
    ℓπTs = logdensity(problem, xs)
    return anneal(annealing, ℓπ0s, ℓπTs)
end

function logdensity_and_gradient(prob::AnnealedDensityProblem, xs::AbstractMatrix)
    (; annealing, proposal, problem, adtype) = prob
    ℓπ0_and_∇ℓπ0 = map(eachcol(xs)) do x
        value_and_gradient(Base.Fix1(logpdf, proposal), adtype, x)
    end
    ℓπ0s = first.(ℓπ0_and_∇ℓπ0)
    ∇ℓπ0s = hcat(last.(ℓπ0_and_∇ℓπ0)...)

    ℓπTs, ∇ℓπTs = logdensity_and_gradient(problem, xs)

    ℓπts = anneal(annealing, ℓπ0s, ℓπTs)
    ∇ℓπts = anneal(annealing, ∇ℓπ0s, ∇ℓπTs)
    return ℓπts, ∇ℓπts
end
