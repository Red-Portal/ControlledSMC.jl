
using Accessors
using ADTypes
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using Plots, StatsPlots
using ProgressMeter
using Random, Random123
using ForwardDiff, ReverseDiff, Zygote
using LogExpFunctions

using ControlledSMC

struct Funnel
    n_dims
end

function LogDensityProblems.capabilities(::Type{<:Funnel})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::Funnel) = prob.n_dims

function LogDensityProblems.logdensity(prob::Funnel, z)
    x, y = z[2:end], z[1]
    ℓp_y = logpdf(Normal(0, 3), y)
    ℓp_x = logpdf(MvNormal(zeros(length(x)), exp(y/2)), x)
    ℓp_x + ℓp_y
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d           = 5
    #μ           = Fill(10, d)
    prob        = Funnel(d) #Dist(MvNormal(μ, 0.1*I))
    prob_ad     = ADgradient(AutoReverseDiff(), prob)
    proposal    = MvNormal(Zeros(d), 3*I)

    n_iters  = 64
    schedule = range(0, 1; length=n_iters).^2
    path     = GeometricAnnealingPath(schedule, proposal, prob_ad)
    Γ        = Eye(d)


    # Obtain a sensible schedule
    h           = 0.03
    n_particles = 256
    sampler = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ)
    _, _, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    for _ in 1:3
        schedule, local_barrier, _ = ControlledSMC.update_schedule(schedule, stats, n_iters)

        path    = @set path.schedule = schedule
        sampler = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ)
        _, _, states, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=true
        )
    end

    # Run Experiment
    n_particles = 1
    n_reps      = 4096
    h_range     = 10.0.^range(-4, 0; length=64)

    ℓZ_hs_samples = @showprogress map(h_range) do h
        pmap(1:n_reps) do key
            seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
            rng  = Philox4x(UInt64, seed, 8)
            set_counter!(rng, key)

            sampler = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ)
            _, _, _, stats = ControlledSMC.sample(
                rng, sampler, path, n_particles, 0; show_progress=false
            )
            last(stats).log_normalizer
        end
    end

    chisqs = map(ℓZ_hs_samples) do ℓZ_samples
        EℓZ = logsumexp(-ℓZ_samples) - log(length(ℓZ_samples))
        logexpm1(EℓZ)
    end
    Plots.plot(h_range, chisqs, xscale=:log10, yscale=:log10, ylims=[1.0, 10^4], label="log Chi(Q,P)", ylabel="Divergence")

    kls = map(ℓZ_hs_samples) do ℓZ_samples
        mean(-ℓZ_samples)
    end
    Plots.plot!(h_range, kls, xscale=:log10, yscale=:log10, ylims=[1.0, 10^4], label="KL(Q,P)")
end
