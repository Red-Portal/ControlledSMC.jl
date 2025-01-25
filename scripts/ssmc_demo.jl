
using ADTypes
using Accessors
using ControlledSMC
using Distributions
using DifferentiationInterface
using FillArrays
using LinearAlgebra
using LogDensityProblems, StanLogDensityProblems, LogDensityProblemsAD
using ReverseDiff, Mooncake, Zygote
using PDMats
using Plots, StatsPlots
using PosteriorDB
using ProgressMeter
using Random, Random123

include("Models/Models.jl")

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

function run_smc(sampler, path, n_particles, n_reps)
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

    # if !isdir(".stan")
    #     mkdir(".stan")
    # end
    # pdb  = PosteriorDB.database()
    # post = PosteriorDB.posterior(pdb, "dogs-dogs_hierarchical")
    # prob = StanProblem(post, ".stan/"; force=true)

    d    = 10
    μ    = Fill(10.0, d)
    #prob = Dist(MvNormal(μ, I))
    prob = Dist(MvTDist(d + 3, μ, PDMats.PDMat(Matrix(0.3*I, (d, d)))))
    #prob = LogisticRegressionSonar()
    #prob = LogGaussianCoxProcess()
    #prob = BrownianMotion()

    d    = LogDensityProblems.dimension(prob)
    #prob = ADgradient(AutoMooncake(; config=Mooncake.Config()), prob; x=randn(rng, d))
    prob = ADgradient(AutoReverseDiff(), prob; x=randn(rng, d))

    n_iters     = 128
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters)
    path        = GeometricAnnealingPath(schedule, proposal, prob)
    n_particles = 512
    n_reps      = 32

    h = 0.1
    Γ = Eye(d)
    
    #adaptor = NoAdaptation()
    #adaptor = BackwardKLMin(128)
    #adaptor = ForwardKLMin(128)
    #adaptor = CondESSMax(128)
    adaptor = AnnealedFlowTransport(128)
    #adaptor = ChiSquareMin(128)
    sampler = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ, adaptor)
    #sampler = SMCUHMC(h, 0.1, n_iters, Γ, adaptor)
    #sampler = SMCUBarker(h, n_iters, TimeCorrectForwardKernel(), adaptor)

    #adaptor = ESJDMax(128)
    #sampler = SMCMALA(h, n_iters, Γ, adaptor, 4)
    #sampler = SMCBarker(h, n_iters, adaptor, 4)

    plot_idx = 1
    #Plots.plot() |> display
    
    x, _, sampler, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    #Plots.plot(sampler.stepsizes) |> display

    sampler = @set sampler.adaptor = NoAdaptation() 

    ℓZs = run_smc(sampler, path, n_particles, n_reps)
    boxplot!(fill(plot_idx, length(ℓZs)), ℓZs, color=:blue, label=nothing, ylabel="log Z", xlabel="SMC Episode") |> display
    plot_idx += 1
    return

    x, _, sampler, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    schedule, local_barrier, _ = ControlledSMC.update_schedule(schedule, stats, n_iters)
    path                       = @set path.schedule = schedule

    #Plots.plot([stat.ess for stat in stats]) |> display
    Plots.plot(schedule) |> display
    #Plots.plot(schedule[2:end], local_barrier[2:end], xscale=:log10) |> display
    #Plots.plot!(schedule) |> display

    #ℓZs = run_smc(sampler, path, n_particles, n_reps)
    #boxplot!(fill(plot_idx, length(ℓZs)), ℓZs, color=:blue, label=nothing) |> display
    #plot_idx += 1

    for i in 1:2 #10
        sampler = @set sampler.adaptor = adaptor
        _, _, sampler, states, stats = ControlledSMC.sample(
             rng, sampler, path, n_particles, 0.5; show_progress=true
        )

        if i == 2
            return sampler
        end

        #Plots.plot!(sampler.stepsizes) |> display

        sampler = @set sampler.adaptor = NoAdaptation() 
        _, _, sampler, states, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=true
        )
        schedule, local_barrier, _ = ControlledSMC.update_schedule(schedule, stats, n_iters)
        path                       = @set path.schedule = schedule

        #ℓZs = run_smc(sampler, path, n_particles, n_reps)
        #boxplot!(fill(plot_idx, length(ℓZs)), ℓZs, color=:blue, label=nothing) |> display
        #plot_idx += 1

        #println(sum([stat.resampled for stat in stats]))
        #Plots.plot!(schedule[2:end], local_barrier[2:end], xscale=:log10) |> display
        Plots.plot!(schedule) |> display
        #Plots.plot!([stat.ess for stat in stats]) |> display
    end
end
