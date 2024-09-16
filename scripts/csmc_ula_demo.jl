
using ADTypes
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using Plots, StatsPlots
using ProgressMeter
using Random, Random123
using ForwardDiff, ReverseDiff, Zygote, Tapir

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

    stepsize    = 0.2
    n_particles = 512
    n_episodes  = 2

    smc  = SMCULA(stepsize, stepsize, TimeCorrectForwardKernel(), Eye(d), path)

    # csmc = CSMCULA(smc, path)
    # _, _, states, stats = ControlledSMC.sample(rng, csmc, path, n_particles, 0.5; show_progress=true)

    # Plots.plot([stat.ess for stat in stats]) |> display

    # for i in 1:3:10
    # csmc = optimize_policy(csmc, states)
    # _, _, states, stats = ControlledSMC.sample(
    #     rng, csmc, path, n_particles, 0.5; show_progress=false
    # )
    # Plots.plot!([stat.ess for stat in stats]) |> display
    # end
    # return stats

    res = @showprogress map(1:32) do _
        _, _, _, stats = ControlledSMC.sample(
            rng, smc, path, n_particles, 0.5; show_progress=false
        )
        last(stats).log_normalizer
    end
    logZ = [last(r) for r in res]

    display(violin!(fill(1, length(logZ)), logZ; fillcolor=:blue, alpha=0.2, label="SMC",),)
    display(dotplot!(fill(1, length(logZ)), logZ; markercolor=:blue, label=nothing),)

    tups = @showprogress map(1:32) do _
        csmc = CSMCULA(smc, path)
        _, _, states, stats = ControlledSMC.sample(
            rng, csmc, path, n_particles, 0.5; show_progress=false
        )
        ℓZs = [last(stats).log_normalizer]

        for _ in 1:n_episodes
            csmc = optimize_policy(csmc, states)
            _, _, states, stats = ControlledSMC.sample(
                rng, csmc, path, n_particles, 0.5; show_progress=false
            )
            push!(ℓZs, last(stats).log_normalizer)
        end
        ℓZs
    end

    for j in 0:n_episodes
        ℓZs = [tup[j + 1] for tup in tups]
        display(violin!(fill(j + 2, length(ℓZs)),  ℓZs; fillcolor=:red, alpha=0.2, label="CSMC J = $(j)",),)
        display(dotplot!(fill(j + 2, length(ℓZs)), ℓZs; markercolor=:red, label=nothing),)
    end
end
