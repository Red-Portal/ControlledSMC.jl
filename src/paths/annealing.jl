
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
        return -Inf
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
        return ℓπt, Fill(-Inf, size(∇ℓπt))
    end
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    prob::AnnealedDensityProblem, x::AbstractVector
)
    (; annealing, proposal, problem, adtype) = prob
    ℓπ0, ∇ℓπ0, ∇2ℓπ0 = value_gradient_and_hessian(Base.Fix1(logpdf, proposal), adtype, x)
    #ℓπT, ∇ℓπT, ∇2ℓπT = LogDensityProblems.logdensity_gradient_and_hessian(problem, x)
    ℓπT, ∇ℓπT, ∇2ℓπT = value_gradient_and_hessian(Base.Fix1(LogDensityProblems.logdensity, problem), adtype, x)
    ℓπt = anneal(annealing, ℓπ0, ℓπT)
    ∇ℓπt = anneal(annealing, ∇ℓπ0, ∇ℓπT)
    ∇2ℓπt = anneal(annealing, ∇2ℓπ0, ∇2ℓπT)
    if isfinite(ℓπt)
        return ℓπt, ∇ℓπt, ∇2ℓπt
    else
        return ℓπt, Fill(-Inf, size(∇ℓπt)), Fill(-Inf, size(∇2ℓπt))
    end
end
