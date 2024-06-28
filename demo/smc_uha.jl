
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using PDMats
using SimpleUnPack
using FillArrays
using Random123

include("common.jl")
include("sample.jl")

struct SMCUHA{
    G    <: AbstractMatrix,
    H    <: Real,
    B    <: AbstractBackwardKernel,
    Prop <: MultivariateDistribution,
    Path <: AnnealingPath,
} <: AbstractSMC
    M        ::G
    δ0       ::H
    δT       ::H
    backward ::B
    proposal ::Prop
    path     ::Path
end

Base.length(sampler::SMCUHA) = length(sampler.path)

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::SMCUHA,
               ::Any,
    n_particles::Int,
)
    (; proposal, M) = sampler
    q = rand(rng, proposal, n_particles)
    d = size(q, 1)
    p = rand(rng, MvNormal(Zeros(d), M), n_particles)
    x = vcat(q, p)
    ℓG = zeros(n_particles)
    x, ℓG
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::SMCUHA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget,
)
    (; path, δ0, δT, M, proposal) = sampler
    logπt(x_)   = annealed_logtarget(path, t,   x_, proposal, logtarget)
    logπtm1(x_) = annealed_logtarget(path, t-1, x_, proposal, logtarget)
    δt = anneal(path, t, δ0, δT)

    d      = size(x, 1) ÷ 2
    n      = size(x, 2)
    q, p   = x[1:d,:], x[d+1:end,:]
    h      = 0.9
    phalf  = h*p + sqrt(1 - h^2)*unwhiten(M, randn(rng, size(p)))
    p_dist = MvNormal(Zeros(d), M)
    
    res = map(1:n) do i
        pi, qi  = p[:,i], q[:,i]
        pihalf  = phalf[:,i]
        qi′, pi′ = leapfrog(logπt, qi, pihalf, δt, M)
        xi′     = vcat(qi′, pi′)

        ℓGi = logπt(qi′) - logπtm1(qi) +
            logpdf(p_dist, pi′) - logpdf(p_dist, pi) +
            logpdf(MvNormal(h*pihalf, (1 - h^2)*M), pi) +
            -logpdf(MvNormal(h*pi, (1 - h^2)*M), pihalf)

        xi′, ℓGi
    end
    x′ = hcat([first(r) for r in res]...)
    ℓG = [last(r) for r in res]
    x′, ℓG, NamedTuple()
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 20
    μ            = Fill(10, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #δ0    = 5e-2
    #δT    = 5e-3
    δ0    = δT = 0.05

    Γ     = Eye(d)

    n_iters  = 16
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    sampler = SMCUHA(
        Γ,
        δ0,
        δT,
        ForwardKernel(),
        proposal,
        AnnealingPath(schedule)
    )

    particles = [32, 64, 128, 256]#, 512, 1024]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end

        logZ = [last(r) for r in res]

        violin!( fill(2*idx-1, length(logZ)), logZ, fillcolor  =:blue, alpha=0.2, label="N=$(n_particles)") |> display
        dotplot!(fill(2*idx-1, length(logZ)), logZ, markercolor=:blue, label=nothing) |> display
    end

    # sampler = SMCUHA(
    #     Γ,
    #     h0,
    #     hT,
    #     DetailedBalance(),
    #     proposal,
    #     AnnealingPath(schedule)
    # )
    # for (idx, n_particles) in enumerate(particles)
    #     res = @showprogress map(1:64) do _
    #         xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
    #         (mean(xs, dims=2)[:,1], last(stats).logZ)
    #     end
    #     logZ = [last(r) for r in res]
    #     violin!( fill(2*idx, length(logZ)), logZ, fillcolor  =:red, alpha=0.2, label="N=$(n_particles)") |> display
    #     dotplot!(fill(2*idx, length(logZ)), logZ, markercolor=:red, label=nothing) |> display
    # end
end
