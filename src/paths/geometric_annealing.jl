
struct GeometricAnnealingPath{
    Sched<:AbstractVector{<:Real},
    Prop<:Distributions.ContinuousMultivariateDistribution,
    Prob,
    AD<:ADTypes.AbstractADType,
} <: AbstractPath
    schedule :: Sched
    proposal :: Prop
    problem  :: Prob
    adtype   :: AD
end

function GeometricAnnealingPath(
    schedule::AbstractVector,
    proposal::Distributions.ContinuousMultivariateDistribution,
    problem,
    adtype::ADTypes.AbstractADType=AutoReverseDiff(),
)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2

    return GeometricAnnealingPath{
        typeof(schedule),typeof(proposal),typeof(problem),typeof(adtype)
    }(
        schedule, proposal, problem, adtype
    )
end

Base.length(path::GeometricAnnealingPath) = length(path.schedule)

function get_target(path::GeometricAnnealingPath, t::Int)
    (; schedule, proposal, problem, adtype) = path
    return AnnealedDensityProblem(
        GeometricAnnealing(schedule[t]), proposal, problem, adtype
    )
end
