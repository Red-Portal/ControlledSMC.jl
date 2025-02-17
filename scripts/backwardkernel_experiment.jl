
using ADTypes
using Accessors
using ControlledSMC
using DifferentiationInterface
using Distributions
using FillArrays
using HDF5, JLD2
using LinearAlgebra
using LogDensityProblems, StanLogDensityProblems, LogDensityProblemsAD
using PDMats
using Plots, StatsPlots
using PosteriorDB
using ProgressMeter
using Random, Random123
using ReverseDiff, Mooncake, Zygote

include("Models/Models.jl")

function run_smc(sampler, path, n_particles, n_reps)
    @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)
        
        _, _, sampler, _, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=false
        )

        sampler = @set sampler.adaptor = NoAdaptation()
        _, _, _, _, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=false
        )
        last(stats).log_normalizer
    end
end

function run_experiment()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d    = 10 
    μ    = Fill(30.0, d)

    prob = Dist(MvNormal(μ, I))
    prob = ADgradient(AutoReverseDiff(), prob; x=randn(rng, d))

    n_iters  = 64
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters)
    path     = GeometricAnnealingPath(schedule, proposal, prob)
    n_reps   = 256
    adaptor  = NoAdaptation()

    plot_idx = 0

    n_particles_range = [128, 256, 512, 1024, 2048, 4096]

    Plots.plot() |> display

    results = Dict()
    for (backname, backward, h) in [
        (:timecorrectforward, TimeCorrectForwardKernel(), 0.5),
        (:detailedbalance, DetailedBalance(), 0.5),
        (:forward, ForwardKernel(), 0.5),
        ],
        n_particles in n_particles_range

        config  = (name=backname, n_particles=n_particles)
        @info("", config...)

        sampler = SMCULA(h, n_iters, backward, Eye(d), adaptor)
        ℓZs     = run_smc(sampler, path, n_particles, n_reps)
        boxplot!(fill(plot_idx, length(ℓZs)), ℓZs, color=:blue, label=nothing) |> display
        plot_idx += 1

        results[config] = ℓZs
        JLD2.save("data/raw/backward_kernel_comparison.jld2", "data", results)
    end
end

function process_data()
    data = JLD2.load("data/raw/backward_kernel_comparison.jld2", "data")

    n_particles_range = [128, 256, 512, 1024, 2048, 4096]

    config = (name=:timecorrectforward, n_particles=128)

    Plots.plot() |> display

    h5open("data/pro/backward_kernel_comparison", "w") do h5
        for backname in [
            :timecorrectforward,
            :detailedbalance,
            :forward,
        ]
            tups = map(n_particles_range) do n_particles
                config  = (name=backname, n_particles=n_particles)
                quantile(data[config], (0.1, 0.5, 0.9))
            end
            ls = [tup[1] for tup in tups]
            ms = [tup[2] for tup in tups]
            us = [tup[3] for tup in tups]

            Plots.plot!(
                n_particles_range, ms,
                ribbon=(ms - ls, us - ms),
                label=string(backname),
                xscale=:log10
            ) |> display

            write(h5, string(backname) * "_x", n_particles_range)
            write(h5, string(backname) * "_y", hcat(ms, us - ms, ms - ls)' |> Array)
        end
    end
end
