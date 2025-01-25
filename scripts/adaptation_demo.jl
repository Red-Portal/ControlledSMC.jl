
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

function run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    _, _, sampler, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )

    # schedule, _, _               = ControlledSMC.update_schedule(path.schedule, stats, n_steps)
    # path                         = @set path.schedule = schedule
    # _, _, sampler, states, stats = ControlledSMC.sample(
    #     rng, sampler, path, n_particles, 0.5; show_progress=true
    # )

    @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)
        
        _, _, _, _, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=false
        )
        last(stats).log_normalizer
    end
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d       = 10
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, 0.3*I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    n_reps      = 32
    n_particles = 512
    n_steps     = 32
    proposal    = MvNormal(Zeros(d), 3*I)
    schedule    = range(0, 1; length=n_steps)
    path        = GeometricAnnealingPath(schedule, proposal, prob_ad)

    Plots.plot() |> display
    plot_idx = 0  

    #h       = 0.1
    #Γ       = Eye(d)
    #sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, NoAdaptation())
    #ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_episodes)
    #Plots.plot!(ℓZ_ests, label="No Adaptation h = $(h)") |> display


    h       = 0.3
    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, NoAdaptation())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="No Adaptation h = $(h)",
             ylabel="log Z estimates",
             ylims=(-75,5),
             ) |> display
    plot_idx += 1

    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, AnnealedFlowTransport())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="AFT") |> display
    plot_idx += 1

    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, ForwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="Forward KL") |> display
    plot_idx += 1

    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, BackwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="Backward KL") |> display
    plot_idx += 1

    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, PartialForwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="Partial Forward KL") |> display
    plot_idx += 1

    Γ       = Eye(d)
    sampler = SMCULA(h, n_steps, TimeCorrectForwardKernel(), Γ, PartialBackwardKLMin())
    ℓZ_ests = run_smc(rng, sampler, path, n_particles, n_steps, n_reps)
    boxplot!(fill(plot_idx, length(ℓZ_ests)), ℓZ_ests, label="Partial Backward KL") |> display
end
