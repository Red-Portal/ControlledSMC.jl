
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using SimpleUnPack
using FillArrays
using Random123

include("common.jl")
include("sample.jl")

struct SMCULA{
    G  <: AbstractMatrix,
    GL <: AbstractMatrix,
    H  <: Real,
    B  <: AbstractBackwardKernel,
    P  <: AnnealingPath,
} <: AbstractSMC
    Γ        ::G
    Γchol    ::GL
    h0       ::H
    hT       ::H
    backward ::B
    path     ::P
end

function mutate(
    rng       ::Random.AbstractRNG,
    sampler   ::SMCULA,
    t         ::Int,
    x         ::AbstractVector,
    proposal,
    logtarget
)
    (; path, h0, hT, Γ, Γchol) = sampler
    logπt(x) = annealed_logtarget(path, t, x, proposal, logtarget)
    ht = anneal(path, t, h0, hT)
    q  = euler_fwd(logπt, x, ht, Γ)
    q + sqrt(2*ht)*Γchol*randn(rng, length(q))
end

function potential(
    sampler   ::SMCULA,
              ::DetailedBalance,
    t         ::Int,
    x_curr    ::AbstractVector,
    x_prev    ::AbstractVector,
    proposal,
    logtarget,
)
    (; path)   = sampler
    ℓπt_xtm1   = annealed_logtarget(path, t,   x_prev, proposal, logtarget)
    ℓπtm1_xtm1 = annealed_logtarget(path, t-1, x_prev, proposal, logtarget)
    ℓπt_xtm1 - ℓπtm1_xtm1
end

function potential(
    sampler   ::SMCULA,
              ::ForwardKernel,
    t         ::Int,
    x_curr    ::AbstractVector,
    x_prev    ::AbstractVector,
    proposal,
    logtarget,
)
    (; h0, hT, Γ, path) = sampler

    ht   = anneal(path, t, h0, hT)
    htm1 = anneal(path, t-1, h0, hT)

    logπt(x)   = annealed_logtarget(path, t,   x, proposal, logtarget)
    logπtm1(x) = annealed_logtarget(path, t-1, x, proposal, logtarget)
    K = MvNormal(euler_fwd(logπt,   x_prev, ht,   Γ), 2*ht*Γ)
    L = MvNormal(euler_fwd(logπtm1, x_curr, htm1, Γ), 2*htm1*Γ)
    (logπt(x_curr) + logpdf(L, x_prev)) - (logπtm1(x_prev) + logpdf(K, x_curr))
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 4
    μ            = randn(d)/2
    logtarget(x) = logpdf(MvNormal(μ, 0.01*I + 0.001*ones(d,d)), x)
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    h0    = 5e-2
    hT    = 5e-3
    h0    = hT = 5e-3

    Γ     = Eye(d)
    Γchol = Eye(d)

    n_iters  = 16
    schedule = range(0, 1; length=n_iters).^2

    hline([0.0]) |> display

    sampler = SMCULA(Γ, Γchol, h0, hT, ForwardKernel(), AnnealingPath(schedule))

    particles = [32, 64, 128, 256, 512, 1024]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, stats    = smc(rng, sampler, n_particles, 0.5, proposal, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end

        logZ = [last(r) for r in res]

        violin!( fill(2*idx-1, length(logZ)), logZ, fillcolor  =:blue, alpha=0.2) |> display
        dotplot!(fill(2*idx-1, length(logZ)), logZ, markercolor=:blue)            |> display
    end

    # sampler = SMCULA(Γ, Γchol, h0, hT, DetailedBalance(), AnnealingPath(schedule))
    # for (idx, n_particles) in enumerate(particles)
    #     res = @showprogress map(1:64) do _
    #         xs, stats    = smc(rng, sampler, n_particles, proposal, logtarget)
    #         (mean(xs, dims=2)[:,1], last(stats).logZ)
    #     end
    #     logZ = [last(r) for r in res]
    #     violin!( fill(2*idx, length(logZ)), logZ, fillcolor  =:red,  alpha=0.2) |> display
    #     dotplot!(fill(2*idx, length(logZ)), logZ, markercolor=:red)             |> display
    # end
end
