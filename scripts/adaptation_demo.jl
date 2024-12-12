
using Accessors
using ADTypes
using ControlledSMC
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using Plots, StatsPlots
using ProgressMeter
using Random, Random123
using ReverseDiff

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

function run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    log_normalizer_est = []
    _, _, sampler, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    push!(log_normalizer_est, last(stats).log_normalizer)

    for i in 1:n_episodes
        schedule, _, _               = ControlledSMC.update_schedule(path.schedule, stats, n_steps)
        path                         = @set path.schedule = schedule
        _, _, sampler, states, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=true
        )

        push!(log_normalizer_est, last(stats).log_normalizer)
    end
    log_normalizer_est
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d       = 10
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, 0.3*I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    n_episodes  = 3
    n_particles = 512
    n_steps     = 16
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_steps)
    path        = GeometricAnnealingPath(schedule, proposal, prob_ad)

    Plots.plot() |> display

    h       = 0.1
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, NoAdaptation())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    Plots.plot!(ℓZ_ests, label="No Adaptation h = $(h)") |> display

    h       = 0.3
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, NoAdaptation())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    Plots.plot!(ℓZ_ests, label="No Adaptation h = $(h)") |> display

    h       = 0.3
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, AnnealedFlowTransport())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    Plots.plot!(ℓZ_ests, label="AFT") |> display

    h       = 0.3
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, PathForwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    Plots.plot!(ℓZ_ests, label="Path forward KL") |> display

    h       = 0.3
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, PathBackwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    Plots.plot!(ℓZ_ests, label="Path reverse KL") |> display
end
