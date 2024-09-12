
struct GeometricAnnealing{
    Sched <: AbstractVector{<:Real},
    Prop  <: Distributions.ContinuousMultivariateDistribution,
    Prob,
    AD,
} <: AbstractPath
    schedule::Sched
    proposal::Prop
    problem ::Prob
    adtype  ::AD
end

function GeometricAnnealing(
    schedule::AbstractVector,
    proposal::Distributions.ContinuousMultivariateDistribution,
    problem,
    adtype = AutoReverseDiff(),
)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2
    
    GeometricAnnealing{
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

Base.length(path::GeometricAnnealing) = length(path.schedule)

function anneal(path::GeometricAnnealing, t::Int, x, y)
    γt = path.schedule[t]
    (1 - γt)*x + γt*y
end

function logtarget(
    path::GeometricAnnealing,
    t   ::Int,
    x   ::AbstractVector,
)
    (; proposal, problem) = path
    ℓπ0 = logpdf(proposal, x)
    ℓπT = LogDensityProblems.logdensity(problem, x)
    anneal(path, t, ℓπ0, ℓπT)
end

function logtarget_batch(
    path::GeometricAnnealing,
    t   ::Int,
    xs  ::AbstractMatrix,
)
    logtarget.(Ref(path), Ref(t), eachcol(xs))
end

function logtarget_gradient(
    path::GeometricAnnealing,
    t   ::Int,
    x   ::AbstractVector,
)
    _, ∇ℓπt = logtarget_and_gradient(path, t, x)
    ∇ℓπt
end

function logtarget_gradient_batch(
    path::GeometricAnnealing,
    t   ::Int,
    xs  ::AbstractMatrix,
)
    mapslices(xi -> logtarget_gradient(path, t, Vector(xi)), xs, dims=1)
end

function logtarget_and_gradient(
    path::GeometricAnnealing,
    t   ::Int,
    x   ::AbstractVector,
)
    (; proposal, problem, adtype) = path
    ℓπ0, ∇ℓπ0 = value_and_gradient(Base.Fix1(logpdf, proposal), adtype, x)
    ℓπT, ∇ℓπT = LogDensityProblems.logdensity_and_gradient(problem, x)
    ℓπt       = anneal(path, t, ℓπ0, ℓπT)
    ∇ℓπt      = anneal(path, t, ∇ℓπ0, ∇ℓπT)
    ℓπt, ∇ℓπt
end

function logtarget_and_gradient_batch(
    path::GeometricAnnealing,
    t   ::Int,
    xs  ::AbstractVector,
)
    res = logtarget_and_gradient.(Ref(path), Ref(t), xs)
    first.(res), last.(res)
end

