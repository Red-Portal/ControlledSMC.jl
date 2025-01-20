
using ADTypes
using Accessors
using Chain
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

mutable struct TrackedLogDensityProblem{Prob}
    n_density_evals  :: Int
    n_gradient_evals :: Int
    prob             :: Prob
end

function TrackedLogDensityProblem(prob)
    TrackedLogDensityProblem{typeof(prob)}(0, 0, prob)
end

function LogDensityProblems.capabilities(::Type{TrackedLogDensityProblem{Prob}}) where {Prob}
    return LogDensityProblems.capabilities(Prob)
end

LogDensityProblems.dimension(prob::TrackedLogDensityProblem) = length(prob.prob)

function LogDensityProblems.logdensity(prob::TrackedLogDensityProblem, x)
    prob.n_density_evals += 1
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::TrackedLogDensityProblem, x)
    prob.n_gradient_evals += 1
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end

function get_stan_model(model_name::String)
    id = myid()
    if !isdir(".stan_$id")
        mkdir(".stan_$id")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, model_name)
    StanProblem(post, ".stan_$id/"; force=true)
end

function prepare_sampler(sampler::SMCULA, d, n_iters)
    @chain sampler begin 
        @set _.precond   = Eye(d)
        @set _.stepsizes = ones(n_iters)
    end
end

function prepare_sampler(sampler::SMCUHMC, d, n_iters)
    @chain sampler begin 
        @set _.mass_matrix = Eye(d)
        @set _.stepsizes   = ones(n_iters)
        @set _.dampings    = ones(n_iters)
    end
end

function run_smc(
    prob,
    adtype,
    sampler,
    n_particles,
    n_iters,
    n_reps;
    show_progress=false,
)
    @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        prob_ad  = if prob isa String
            get_stan_model(prob)
        else
            d = LogDensityProblems.dimension(prob)
            ADgradient(adtype, prob; x=randn(rng, d))
        end
        prob_ad_tracked = TrackedLogDensityProblem(prob_ad)
        d               = LogDensityProblems.dimension(prob_ad)
        sampler         = prepare_sampler(sampler, d, n_iters)
        proposal        = MvNormal(Zeros(d), I)
        schedule        = range(0, 1; length=n_iters).^2
        path            = GeometricAnnealingPath(schedule, proposal, prob_ad)
        path_tracked    = GeometricAnnealingPath(schedule, proposal, prob_ad_tracked)
        ℓZs             = Float64[]
        n_grad_evals    = Int64[]
        n_density_evals = Int64[]

        local_barrier  = nothing
        global_barrier = nothing

        # Run with initial n_iters
        _, _, sampler, _, stats = ControlledSMC.sample(
            rng, sampler, path_tracked, n_particles, 0.5;
            show_progress = (key == 1) ? show_progress : false
        )
        sampler_test = @set sampler.adaptor = NoAdaptation()
        _, _, _, _, stats_test = ControlledSMC.sample(
            rng, sampler_test, path, n_particles, 0.5;
            show_progress = (key == 1) ? show_progress : false
        )
        push!(n_grad_evals,    prob_ad_tracked.n_gradient_evals)
        push!(n_density_evals, prob_ad_tracked.n_density_evals)
        push!(ℓZs, last(stats_test).log_normalizer)

        for i in 1:2
            # Compute global communication barrier and update schedule
            _, _, global_barrier = ControlledSMC.update_schedule(
                schedule, stats, length(schedule)
            )
            n_iters_updated = ceil(Int, 2*global_barrier)
            schedule, local_barrier, global_barrier = ControlledSMC.update_schedule(
                schedule, stats, n_iters_updated
            )
            path_tracked = @set path_tracked.schedule = schedule
            path         = @set path.schedule         = schedule
            sampler      = prepare_sampler(sampler, d, n_iters_updated)

            # Run with updated n_iters
            _, _, sampler, _, stats = ControlledSMC.sample(
                rng, sampler, path_tracked, n_particles, 0.5;
                show_progress = (key == 1) ? show_progress : false
            )
            sampler_test = @set sampler.adaptor = NoAdaptation()
            _, _, _, _, stats_test = ControlledSMC.sample(
                rng, sampler_test, path, n_particles, 0.5;
                show_progress = (key == 1) ? show_progress : false
            )
            push!(n_grad_evals,    prob_ad_tracked.n_gradient_evals)
            push!(n_density_evals, prob_ad_tracked.n_density_evals)
            push!(ℓZs, last(stats_test).log_normalizer)
        end

        return (
            sampler         = sampler,
            local_barrier   = local_barrier,
            global_barrier  = global_barrier,
            schedule        = schedule,
            n_grad_evals    = n_grad_evals,
            n_density_evals = n_density_evals,
            log_normalizer  = ℓZs,
        )
    end
end

function run_posteriordb_experiments()
    fname = "data/raw/posteriordb_experiments.jld2"

    n_particles = 1024
    n_reps      = 32
    n_iters     = 64
    adtype      = AutoMooncake(; config=Mooncake.Config())

    metadata = (
        samplers    = [:SMCULA, :SMCUHMC],
        n_iters     = [64],
        n_particles = [1024],
        n_subsample = [128],
    )

    if !isfile(fname)
        JLD2.save(fname, "data", Dict(), "metadata", metadata)
    end

    for name in reverse([
        "lsat_data-lsat_model",
        "radon_all-radon_variable_intercept_slope_centered",
        "radon_all-radon_variable_intercept_slope_noncentered",
        "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
        "timssAusTwn_irt-gpcm_latent_reg_irt",
        "three_men2-ldaK2",
        "three_men3-ldaK2",
        "three_men1-ldaK2",
        "Mth_data-Mth_model", # 4 minutes , T = 128 not enough
        "radon_all-radon_hierarchical_intercept_centered",
        "radon_all-radon_hierarchical_intercept_noncentered",
        "radon_all-radon_variable_intercept_centered",
        "radon_all-radon_variable_intercept_noncentered",
        "radon_all-radon_variable_slope_centered",
        "radon_all-radon_variable_slope_noncentered",
        "radon_all-radon_partially_pooled_centered",
        "radon_all-radon_partially_pooled_noncentered",
        "election88-election88_full",
        "radon_mod-radon_county",
        "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
        "Mh_data-Mh_model",
        "radon_all-radon_county_intercept",
        "GLMM_data-GLMM1_model",
        "radon_mn-radon_variable_intercept_slope_centered",
        "radon_mn-radon_variable_intercept_slope_noncentered",
        "Mtbh_data-Mtbh_model",
        "irt_2pl-irt_2pl",
        "butterfly-multi_occupancy",
        "radon_mn-radon_hierarchical_intercept_centered", 
        "radon_mn-radon_hierarchical_intercept_noncentered",
        "radon_mn-radon_variable_intercept_centered",
        "radon_mn-radon_variable_intercept_noncentered",
        "radon_mn-radon_variable_slope_centered",
        "radon_mn-radon_variable_slope_noncentered",
        "radon_mn-radon_partially_pooled_centered",
        "radon_mn-radon_partially_pooled_noncentered",
        "radon_mn-radon_county_intercept",
        "rats_data-rats_model",
        "rstan_downloads-prophet",
        "GLMM_Poisson_data-GLMM_Poisson_model",
        "iohmm_reg_simulated-iohmm_reg",
        "diamonds-diamonds",
        "seeds_data-seeds_centered_model",
        "seeds_data-seeds_model",
        "seeds_data-seeds_stanified_model",
        "pilots-pilots",
        "loss_curves-losscurve_sislob",
        "hmm_gaussian_simulated-hmm_gaussian",
        "bones_data-bones_model",
        "surgical_data-surgical_model",
        ]),
        (n_particles, n_subsample) in [ (1024, 128), ],
        (samplername, sampler) in [
            (:SMCULA, SMCULA(
                1.0, 1, TimeCorrectForwardKernel(), Eye(1),
                AnnealedFlowTransport(; n_subsample=128, regularization=0.1)
            )),
            (:SMCUHMC, SMCUHMC(
                1.0, 1.0, 1, Eye(1),
                AnnealedFlowTransport(; n_subsample=128, regularization=10.0)
            ))
        ]
        
        config = (
            name        = name,
            sampler     = samplername,
            n_iters     = n_iters,
            n_particles = n_particles,
            n_subsample = n_subsample,
        )

        data  = JLD2.load(fname, "data")
        if haskey(data, config)
            continue
        else
            @info("Running", config...)
            try
                data[config] = :running
                res          = run_smc(name, adtype, sampler, n_particles, n_iters, n_reps; show_progress=true)
                data         = JLD2.load(fname, "data")
                data[config] = res
                JLD2.save(fname, "data", data, "metadata", metadata)
            catch e
                @warn "$(name) failed:\n$(e)"
                data         = JLD2.load(fname, "data")
                data[config] = e
                JLD2.save(fname, "data", data, "metadata", metadata)
            end
        end
    end
end

function main()
    run_posteriordb_experiments()
end
