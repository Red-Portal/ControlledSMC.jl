
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
include("mcmc.jl")

struct SMCKLMC{
    G    <: AbstractPDMat,
    H    <: Real,
    B    <: AbstractBackwardKernel,
    Prop <: MultivariateDistribution,
    Path <: AnnealingPath,
} <: AbstractSMC
    h        ::H
    γ        ::H
    Σ        ::G
    backward ::B
    proposal ::Prop
    path     ::Path
end

function SMCKLMC(
    h       ::Real,
    gamma   ::Real,
    sigma2  ::Real,
    backward::AbstractBackwardKernel,
    proposal::MultivariateDistribution,
    path    ::AnnealingPath,
)
    d = length(proposal)
    Σ = klmc_cov(d, h, gamma, sigma2)
    SMCKLMC(h, gamma, PDMat(Σ), backward, proposal, path)
end

Base.length(sampler::SMCKLMC) = length(sampler.path)

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::SMCKLMC,
               ::Any,
    n_particles::Int,
)
    (; proposal,) = sampler
    x  = rand(rng, proposal, n_particles)
    d  = size(x, 1)
    v  = rand(rng, MvNormal(Zeros(d), I), n_particles)
    z  = vcat(x, v)
    ℓG = zeros(n_particles)
    z, ℓG
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::SMCKLMC,
    t         ::Int,
    z         ::AbstractMatrix,
    logtarget,
)
    (; path, h, γ, Σ, proposal) = sampler
    logπt(x_)   = annealed_logtarget(path, t,   x_, proposal, logtarget)
    logπtm1(x_) = annealed_logtarget(path, t-1, x_, proposal, logtarget)
    d           = length(proposal)
    n           = size(z, 2)
    v_dist      = MvNormal(Zeros(d), I)

    x, v = z[1:d,:], z[d+1:end,:]

    μ_fwd = klmc_fwd(logπt, x, v, h, γ)
    ϵ     = rand(rng, MvNormal(Zeros(2*d), Σ), n)
    z′    = μ_fwd + ϵ

    x′, v′ = z′[1:d,:], z′[d+1:end,:]

    ℓπt     = logπt.(eachcol(x′)) 
    ℓπtm1   = logπtm1.(eachcol(x)) 
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(v′))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(v))

    μ_bwd = klmc_fwd(logπtm1, x′, -v′, h, γ)

    ℓF = map(i -> logpdf(MvNormal(μ_fwd[:,i], Σ), z′[:,i]), 1:n) 
    ℓB = map(i -> logpdf(MvNormal(μ_bwd[:,i], Σ), z[:,i]), 1:n)
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓB - ℓF
    z′, ℓG, NamedTuple()
end

function underdamped_langevin(rng, logtarget, x0, h, gamma, sigma2, n_samples)
    d      = length(x0)
    v_dist = MvNormal(Zeros(d), I)
    v      = rand(rng, v_dist)
    Σ      = klmc_cov(d, h, gamma, sigma2) |> PDMat

    xs = zeros(d, n_samples)
    x  = x0
    for i in 1:n_samples
        μ_fwd = klmc_fwd(logtarget, x, v, h, gamma)
        ϵ     = rand(rng, MvNormal(Zeros(2*d), Σ))
        z′     = μ_fwd + ϵ
        x, v  = z′[1:d], z′[d+1:end]
        xs[:,i] = x
    end
    xs
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

    h  = 1.0
    γ  = 5.
    σ2 = 2*γ
    M  = Eye(d)

    #xs = underdamped_langevin(rng, logtarget, randn(rng, d), h, γ, σ2, 100)
    #return Plots.plot(xs[1,:], xs[2,:], marker=:circ)

    n_iters  = 16
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    sampler = SMCKLMC(
        h, γ, σ2,
        ForwardKernel(),
        proposal,
        AnnealingPath(schedule)
    )

    particles = [32, 64, 128, 256,]# 512, 1024]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:4) do _
            xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end

        logZ = [last(r) for r in res]

        violin!( fill(2*idx-1, length(logZ)), logZ, fillcolor  =:blue, alpha=0.2, label="N=$(n_particles)") |> display
        dotplot!(fill(2*idx-1, length(logZ)), logZ, markercolor=:blue, label=nothing) |> display
    end
end
