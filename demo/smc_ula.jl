
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

struct SMCULA{
    G    <: AbstractMatrix,
    H    <: Real,
    B    <: AbstractBackwardKernel,
    Prop <: MultivariateDistribution,
    Path <: AnnealingPath,
} <: AbstractSMC
    Γ        ::G
    h0       ::H
    hT       ::H
    backward ::B
    proposal ::Prop
    path     ::Path
end

Base.length(sampler::SMCULA) = length(sampler.path)

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::SMCULA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget
)
    (; path, h0, hT, Γ, proposal) = sampler
    logπt(x) = annealed_logtarget(path, t, x, proposal, logtarget)
    ht = anneal(path, t, h0, hT)
    q  = mapslices(xi -> euler_fwd(logπt, xi, ht, Γ), x, dims=1)
    x′ = q + sqrt(ht)*unwhiten(Γ, randn(rng, size(q)))
    ℓG = potential(sampler, t, x′, x, logtarget)
    x′, ℓG, NamedTuple()
end

function potential(
    sampler   ::SMCULA,
    t         ::Int,
    x_curr    ::AbstractMatrix,
    x_prev    ::AbstractMatrix,
    logtarget,
)
    potential_with_backward(
        sampler, sampler.backward, t, x_curr, x_prev, logtarget,
    )
end

function potential_with_backward(
    sampler   ::SMCULA,
              ::DetailedBalance,
    t         ::Int,
    x_curr    ::AbstractMatrix,
    x_prev    ::AbstractMatrix,
    logtarget,
)
    (; proposal, path)   = sampler

    logπt(x)   = annealed_logtarget(path,   t, x, proposal, logtarget)
    logπtm1(x) = annealed_logtarget(path, t-1, x, proposal, logtarget)

    ℓπt_xtm1   = map(xi -> logπt(xi),   eachcol(x_prev))
    ℓπtm1_xtm1 = map(xi -> logπtm1(xi), eachcol(x_prev))
    ℓπt_xtm1 - ℓπtm1_xtm1
end

function potential_with_backward(
    sampler   ::SMCULA,
              ::ForwardKernel,
    t         ::Int,
    x_curr    ::AbstractMatrix,
    x_prev    ::AbstractMatrix,
    logtarget,
)
    (; proposal, h0, hT, Γ, path) = sampler

    ht   = anneal(path, t, h0, hT)
    htm1 = anneal(path, t-1, h0, hT)
    logπt(x)   = annealed_logtarget(path, t,   x, proposal, logtarget)
    logπtm1(x) = annealed_logtarget(path, t-1, x, proposal, logtarget)

    map(eachcol(x_curr), eachcol(x_prev)) do xi_curr, xi_prev
        K = MvNormal(euler_fwd(logπt,   xi_prev, ht,   Γ), ht*Γ)
        L = MvNormal(euler_fwd(logπtm1, xi_curr, htm1, Γ), htm1*Γ)
        (logπt(xi_curr) + logpdf(L, xi_prev)) -
            (logπtm1(xi_prev) + logpdf(K, xi_curr))
    end
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

    #h0    = 5e-2
    #hT    = 5e-3
    h0    = hT = 0.5

    Γ     = Eye(d)

    n_iters  = 32
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    sampler = SMCULA(
        Γ,
        h0,
        hT,
        ForwardKernel(),
        proposal,
        AnnealingPath(schedule)
    )

    particles = [32, 64, 128, 256, 512]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end

        logZ = [last(r) for r in res]

        violin!( fill(2*idx-1, length(logZ)), logZ, fillcolor  =:blue, alpha=0.2, label="N=$(n_particles)") |> display
        dotplot!(fill(2*idx-1, length(logZ)), logZ, markercolor=:blue, label=nothing) |> display
    end

    sampler = SMCULA(
        Γ,
        h0,
        hT,
        DetailedBalance(),
        proposal,
        AnnealingPath(schedule)
    )
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end
        logZ = [last(r) for r in res]
        violin!( fill(2*idx, length(logZ)), logZ, fillcolor  =:red, alpha=0.2, label="N=$(n_particles)") |> display
        dotplot!(fill(2*idx, length(logZ)), logZ, markercolor=:red, label=nothing) |> display
    end
end
