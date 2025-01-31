
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

function LogDensityProblems.logdensity(prob::AnnealedDensityProblem, x::AbstractVector)
    (; annealing, proposal, problem) = prob
    ℓπ0 = logpdf(proposal, x)
    ℓπT = LogDensityProblems.logdensity(problem, x)
    ℓπt = anneal(annealing, ℓπ0, ℓπT)
    if isfinite(ℓπt)
        return ℓπt
    else
        return zero(ℓπt)
    end
end

function LogDensityProblems.logdensity_and_gradient(
    prob::AnnealedDensityProblem, x::AbstractVector
)
    (; annealing, proposal, problem, adtype) = prob
    ℓπ0, ∇ℓπ0 = value_and_gradient(Base.Fix1(logpdf, proposal), adtype, x)
    ℓπT, ∇ℓπT = LogDensityProblems.logdensity_and_gradient(problem, x)
    ℓπt = anneal(annealing, ℓπ0, ℓπT)
    ∇ℓπt = anneal(annealing, ∇ℓπ0, ∇ℓπT)
    if isfinite(ℓπt)
        return ℓπt, ∇ℓπt
    else
        return ℓπt, zero(∇ℓπt)
    end
end
