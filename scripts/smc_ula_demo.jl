
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

    d       = 3
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    n_iters  = 64
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters)
    path     = GeometricAnnealingPath(schedule, proposal, prob_ad)

    h0 = 0.5
    hT = 0.5
    Γ  = Eye(d)

    display(hline([0.0]; label="True logZ"))

    #sampler = SMCULA(h0, hT, ForwardKernel(), Γ, path)
    #res = @showprogress map(1:32) do _
    #    xs, _, _, stats = ControlledSMC.sample(rng, sampler, path, 256, 1.0; show_progress=true)
    #    last(stats).log_normalizer
    #end
    #return res

    #
    # SMC-ULA with reverse kernel set as:
    #     L_{t-1}(x_{t-1}, x_t) = K_{t-1}(x_{t-1}, x_t)
    #
    sampler = SMCULA(h0, hT, TimeCorrectForwardKernel(), Γ, path)
    particles = [32, 64, 128]#, 256, 512]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, _, stats = ControlledSMC.sample(
                rng, sampler, path, n_particles, 0.5; show_progress=false
            )
            (mean(xs; dims=2)[:, 1], last(stats).log_normalizer)
        end

        logZ = [last(r) for r in res]

        display(
            violin!(
                fill(3 * idx - 2, length(logZ)),
                logZ;
                fillcolor=:blue,
                alpha=0.2,
                label=" N=$(n_particles)",
            ),
        )
        display(
            dotplot!(
                fill(3 * idx - 2, length(logZ)), logZ; markercolor=:blue, label=nothing
            ),
        )
    end

    #
    # SMC-ULA with reverse kernel set as:
    #     L_{t-1}(x_{t-1}, x_t) = K_t(x_{t-1}, x_t)
    #
    sampler = SMCULA(h0, hT, ForwardKernel(), Γ, path)
    particles = [32, 64, 128]#, 256, 512]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, _, stats = ControlledSMC.sample(
                rng, sampler, path, n_particles, 0.5; show_progress=false
            )
            (mean(xs; dims=2)[:, 1], last(stats).log_normalizer)
        end

        logZ = [last(r) for r in res]

        display(
            violin!(
                fill(3 * idx - 1, length(logZ)),
                logZ;
                fillcolor=:green,
                alpha=0.2,
                label="N=$(n_particles)",
            ),
        )
        display(
            dotplot!(
                fill(3 * idx - 1, length(logZ)), logZ; markercolor=:green, label=nothing
            ),
        )
    end

    #
    # SMC-ULA with reverse kernel set as:
    #     L_{t-1}(x_{t-1}, x_t) = \pi_{t}(x_{t}) K_t(x_t, x_{t-1}) / \pi_{t-1}(x_{t-1})
    #
    sampler = SMCULA(h0, hT, DetailedBalance(), Γ, path)
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, _, stats = ControlledSMC.sample(
                rng, sampler, path, n_particles, 0.5; show_progress=false
            )
            (mean(xs; dims=2)[:, 1], last(stats).log_normalizer)
        end
        logZ = [last(r) for r in res]
        display(
            violin!(
                fill(3 * idx, length(logZ)),
                logZ;
                fillcolor=:red,
                alpha=0.2,
                label="N=$(n_particles)",
            ),
        )
        display(
            dotplot!(fill(3 * idx, length(logZ)), logZ; markercolor=:red, label=nothing)
        )
    end
end
