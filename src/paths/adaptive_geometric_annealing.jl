
struct AdaptiveGeometricAnnealing{
    ESSThres <: Real,
    Sched    <: AbstractVector{<:Real},
    Prop     <: Distributions.ContinuousMultivariateDistribution,
    Prob,
    AD       <: ADTypes.AbstractADType,
} <: AbstractAnnealing
    ess_thres::ESSThres
    proposal ::Prop
    problem  ::Prob
    adtype   ::AD
end

function AdaptiveGeometricAnnealing(
    ess_thres::Real,
    proposal ::Distributions.ContinuousMultivariateDistribution,
    problem,
    adtype   ::ADTypes.AbstractADType = AutoReverseDiff(),
)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2
    
    GeometricAnnealing{
        typeof(ess_thres),
        typeof(proposal),
        typeof(problem),
        typeof(adtype),
    }(
        ess_thres,
        proposal,
        problem,
        adtype
    )
end

Base.length(path::AdaptiveGeometricAnnealing) = length(path.schedule)

function anneal(path::AdaptiveGeometricAnnealing, t::Int, x, y)
    γt = path.schedule[t]
    (1 - γt)*x + γt*y
end
