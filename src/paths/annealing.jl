
abstract type AbstractAnnealing end

struct GeometricAnnealing{G <: Real} <: AbstractAnnealing
    gamma::G
end

anneal(a::GeometricAnnealing, x, y) = (1 - a.gamma)*x + a.gamma*y

struct AnnealedDensityProblem{
    A  <: AbstractAnnealing,
    Q  <: ContinuousMultivariateDistribution,
    P,
    AD <: ADTypes.AbstractADType
}
    annealing::A
    proposal ::Q
    problem  ::P
    adtype   ::AD
end

function logdensity(prob::AnnealedDensityProblem, x)
    (; annealing, proposal, problem) = prob
    ℓπ0 = logpdf(proposal, x)
    ℓπT = LogDensityProblems.logdensity(problem, x)
    anneal(annealing, ℓπ0, ℓπT)
end

function logdensity_batch(prob::AnnealedDensityProblem, xs::AbstractMatrix)
    logdensity.(Ref(prob), eachcol(xs))
end

function logdensity_and_gradient(prob::AnnealedDensityProblem, x::AbstractVector)
    (; annealing, proposal, problem, adtype) = prob
    ℓπ0, ∇ℓπ0 = value_and_gradient(Base.Fix1(logpdf, proposal), adtype, x)
    ℓπT, ∇ℓπT = LogDensityProblems.logdensity_and_gradient(problem, x)
    ℓπt       = anneal(annealing, ℓπ0, ℓπT)
    ∇ℓπt      = anneal(annealing, ∇ℓπ0, ∇ℓπT)
    ℓπt, ∇ℓπt
end

function logdensity_gradient(prob::AnnealedDensityProblem, x::AbstractVector)
    _, ∇ℓπt = logdensity_and_gradient(prob, x)
    ∇ℓπt
end

function logdensity_gradient_batch(prob::AnnealedDensityProblem, xs::AbstractMatrix)
    mapslices(xi -> logdensity_gradient(prob, Vector(xi)), xs, dims=1)
end


function logdensity_and_gradient_batch(prob::AnnealedDensityProblem, xs::AbstractMatrix)
    res = logdensity_and_gradient.(Ref(prob), xs)
    first.(res), last.(res)
end
