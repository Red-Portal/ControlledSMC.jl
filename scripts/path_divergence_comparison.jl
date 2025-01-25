
using ADTypes
using Accessors
using Distributed
using Distributions
using FillArrays
using ForwardDiff, ReverseDiff, Zygote
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using LogExpFunctions
using Plots, StatsPlots
using ProgressMeter
using Random, Random123

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
    proposal    = MvNormal(Zeros(d), 2*I)

    n_iters  = 4
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
    h_range     = 10.0.^range(-4, -0.3; length=64)

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

    ℓchisqs = map(ℓZ_hs_samples) do ℓZ_samples
        ℓEZ = logsumexp(-ℓZ_samples) - log(length(ℓZ_samples))
        logexpm1(ℓEZ)
    end
    Plots.plot(h_range, ℓchisqs, xscale=:log10, yscale=:log10, color=:blue, ylims=[1.0, 10^4], label="log Chi(Q,P)", ylabel="Divergence")
    Plots.vline!([h_range[argmin(ℓchisqs)]], color=:blue, linestyle=:dash, label=nothing)

    kls = map(ℓZ_hs_samples) do ℓZ_samples
        mean(-ℓZ_samples)
    end
    Plots.plot!(h_range, kls, xscale=:log10, yscale=:log10, color=:red, ylims=[1.0, 10^4], label="KL(Q,P)")
    Plots.vline!([h_range[argmin(kls)]], color=:red, linestyle=:dash, label=nothing)

    ℓvar = map(ℓZ_hs_samples) do ℓZ_samples
        ℓEZ  = logsumexp(-ℓZ_samples)   - log(length(ℓZ_samples))
        ℓEZ2 = logsumexp(-2*ℓZ_samples) - log(length(ℓZ_samples))
        logsubexp(ℓEZ2, 2*ℓEZ) 
    end
    Plots.plot!(h_range, ℓvar, xscale=:log10, yscale=:log10, color=:green, ylims=[1.0, 10^4], label="log variance")
    Plots.vline!([h_range[argmin(ℓvar)]], color=:green, linestyle=:dash, label=nothing)
end
