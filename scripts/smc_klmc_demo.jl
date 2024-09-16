
using ADTypes
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using Plots, StatsPlots
using ProgressMeter
using Random, Random123
using ForwardDiff, ReverseDiff, Zygote, Tapir
using PDMats

using ControlledSMC

struct Dist{D}
    dist::D
end

function LogDensityProblems.capabilities(::Type{<:Dist})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::Dist) = length(prob.dist)

function LogDensityProblems.logdensity(prob::Dist, x)
    return logpdf(prob.dist, x)
end

function klmc_mcmc(rng, π, x0, stepsize, damping, invmass, n_samples)
    n_dims = length(x0)
    v0     = randn(rng, n_dims)
    Σ_klmc = ControlledSMC.klmc_cov(stepsize, damping, invmass)
    post   = zeros(n_dims, n_samples)
    x, v = reshape(x0, (:, 1)), reshape(v0, (:, 1))
    for i in 1:n_samples
        K         = ControlledSMC.klmc_transition_kernel(π, x, v, stepsize, damping, invmass, Σ_klmc)
        x, v      = ControlledSMC.klmc_rand(rng, K)
        post[:,i] = x[:, 1]
    end
    post
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d       = 10
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    n_iters  = 32
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters)
    path     = GeometricAnnealingPath(schedule, proposal, prob_ad)

    display(hline([0.0]; label="True logZ"))

    stepsize  = 1.0
    damping   = 2.0
    invmass   = 1.0
    n_samples = 1000

    #x0   = rand(rng, proposal)
    #post = klmc_mcmc(rng, path, x0, stepsize, damping, invmass, 10^4)
    #plt1 = Plots.scatter(post[1,:], post[2,:])
    #return plt1

    #sampler = SMCKLMC(0.5, 10.0)
    #xs, _, stats = ControlledSMC.sample(rng, sampler, path, 1024, 1.0; show_progress=true)
    #return

    n_particles = 256
    dampings = [2.0, 5.0, 10.0, 50.0]

    for (idx, damping) in enumerate(dampings)
        sampler = SMCKLMC(stepsize*damping, damping, invmass)

        res = @showprogress map(1:64) do _
            xs, _, _, stats = ControlledSMC.sample(
                rng, sampler, path, n_particles, 0.5; show_progress=false
            )
            (mean(xs; dims=2)[:, 1], last(stats).log_normalizer)
        end

        logZ = [last(r) for r in res]

        display(
            violin!(
                fill(idx, length(logZ)),
                logZ;
                fillcolor=:blue,
                alpha=0.2,
                label="damping = $(damping)",
            ),
        )
        display(dotplot!(fill(idx, length(logZ)), logZ; markercolor=:blue, label=nothing))
    end
end
