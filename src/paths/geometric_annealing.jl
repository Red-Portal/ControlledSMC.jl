
struct GeometricAnnealingPath{
    Sched <: AbstractVector{<:Real},
    Prop  <: Distributions.ContinuousMultivariateDistribution,
    Prob,
    AD    <: ADTypes.AbstractADType,
} <: AbstractPath
    schedule::Sched
    proposal::Prop
    problem ::Prob
    adtype  ::AD
end

function GeometricAnnealingPath(
    schedule::AbstractVector,
    proposal::Distributions.ContinuousMultivariateDistribution,
    problem,
    adtype  ::ADTypes.AbstractADType = AutoReverseDiff(),
)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2
    
    GeometricAnnealingPath{
        typeof(schedule),
        typeof(proposal),
        typeof(problem),
        typeof(adtype),
    }(
        schedule,
        proposal,
        problem,
        adtype
    )
end

Base.length(path::GeometricAnnealingPath) = length(path.schedule)

function step(
    path::GeometricAnnealingPath,
    t   ::Int,
    x   ::AbstractMatrix,
    w   ::AbstractVector
)
    (; schedule, proposal, problem, adtype) = path
    AnnealedDensityProblem(
        GeometricAnnealing(schedule[t]), proposal, problem, adtype
    )
end
