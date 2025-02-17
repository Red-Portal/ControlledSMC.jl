
using ADTypes
using Accessors
using Base.GC
using BridgeStan
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

function LoadStanProblem(
    post::PosteriorDB.Posterior, path::AbstractString; force::Bool=false, kwargs...
)
    model = PosteriorDB.model(post)
    data = PosteriorDB.load(PosteriorDB.dataset(post), String)
    lib = joinpath(path, "$(model.name)_model.so")
    if isfile(lib)
        return StanLogDensityProblems.StanProblem(lib, data; kwargs...)
    else
        stan_file = PosteriorDB.path(PosteriorDB.implementation(model, "stan"))
        stan_file_new = joinpath(path, basename(stan_file))
        cp(stan_file, stan_file_new; force=force)
        return StanLogDensityProblems.StanProblem(stan_file_new, data; kwargs...)
    end
end

function get_stan_model(model_name::String)
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, model_name)
    return LoadStanProblem(
        post, ".stan/"; force=true, nan_on_error=true, make_args=["STAN_THREADS=true"]
    )
end

function run_adaptive_smc(
    prob,
    adtype,
    sampler_type,
    adaptor,
    n_particles,
    n_iters,
    n_reps;
    n_rounds=2,
    adapt_n_iters_factor=2,
    show_progress=false,
    enable_threading=false,
    prepare_gradient=true,
)
    if prob isa String
        @info("loading stan models...")
        for pid in procs()
            task = @spawnat pid begin
                get_stan_model(prob)
            end
            fetch(task)
        end
        @info("loading stan models... - done!")
    end

    @showprogress pmap(1:n_reps) do key
        seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng  = Philox4x(UInt64, seed, 8)
        set_counter!(rng, key)

        prob_ad  = if prob isa String
            get_stan_model(prob)
        elseif prepare_gradient
            d = LogDensityProblems.dimension(prob)
            ADgradient(adtype, prob; x=randn(rng, d))
        else
            d = LogDensityProblems.dimension(prob)
            ADgradient(adtype, prob)
        end
        prob_ad_tracked = TrackedLogDensityProblem(prob_ad)

        prob_ad_tracked_thread = if enable_threading
            MultithreadedLogDensityProblem(prob_ad_tracked)
        else
            prob_ad_tracked
        end

        prob_ad_thread = if enable_threading
            MultithreadedLogDensityProblem(prob_ad)
        else
            prob_ad
        end

        d               = LogDensityProblems.dimension(prob_ad)
        proposal        = MvNormal(Zeros(d), I)
        schedule        = range(0, 1; length=n_iters).^2
        path            = GeometricAnnealingPath(schedule, proposal, prob_ad_thread)
        path_tracked    = GeometricAnnealingPath(schedule, proposal, prob_ad_tracked_thread)
        ℓZs             = Float64[]
        n_grad_evals    = Int64[]
        n_density_evals = Int64[]
        schedules       = []
        stepsizes       = []
        infos           = []

        local_barrier  = nothing
        global_barrier = nothing

        sampler = sampler_type(path_tracked, adaptor)

        # Run with initial n_iters
        _, _, sampler, _, info = ControlledSMC.sample(
            rng, sampler, n_particles, 0.5;
            show_progress = (key == 1) ? show_progress : false
        )

        sampler_test = @chain sampler begin
            @set _.adaptor = nothing
            @set _.path    = path
        end
        _, _, _, _, info_test = ControlledSMC.sample(
            rng, sampler_test, n_particles, 0.5;
            show_progress = (key == 1) ? show_progress : false
        )
        push!(n_grad_evals,    prob_ad_tracked.n_gradient_evals)
        push!(n_density_evals, prob_ad_tracked.n_density_evals)
        push!(ℓZs,             last(info_test).log_normalizer)
        push!(infos,           info)
        push!(schedules,       schedule)
        push!(stepsizes,       sampler.stepsizes)

        for i in 1:n_rounds
            # Compute global communication barrier and update schedule
            _, _, global_barrier = ControlledSMC.update_schedule(
                schedule, info, length(schedule)
            )
            n_iters_updated = ceil(Int, adapt_n_iters_factor*global_barrier)
            schedule, local_barrier, global_barrier = ControlledSMC.update_schedule(
                schedule, info, n_iters_updated
            )
            path_tracked = @set path_tracked.schedule = schedule
            path         = @set path.schedule         = schedule
            sampler      = sampler_type(path_tracked, adaptor)

            # Run with updated n_iters
            _, _, sampler, _, info = ControlledSMC.sample(
                rng, sampler, n_particles, 0.5;
                show_progress = (key == 1) ? show_progress : false
            )

            sampler_test = @chain sampler begin
                @set _.adaptor = nothing
                @set _.path    = path
            end
            _, _, _, _, info_test = ControlledSMC.sample(
                rng, sampler_test, n_particles, 0.5;
                show_progress = (key == 1) ? show_progress : false
            )
            push!(schedules,       schedule)
            push!(stepsizes,       sampler.stepsizes)
            push!(n_grad_evals,    prob_ad_tracked.n_gradient_evals)
            push!(n_density_evals, prob_ad_tracked.n_density_evals)
            push!(ℓZs,             last(info_test).log_normalizer)
            push!(infos,           info)
        end

        GC.gc()

        return (
            stepsizes       = stepsizes,
            local_barrier   = local_barrier,
            global_barrier  = global_barrier,
            schedules       = schedules,
            n_grad_evals    = n_grad_evals,
            n_density_evals = n_density_evals,
            log_normalizer  = ℓZs,
            infos           = infos,
        )
    end
end

const posteriordb_problems = [
    "lsat_data-lsat_model",
    "radon_all-radon_variable_intercept_slope_centered",
    "radon_all-radon_variable_intercept_slope_noncentered",
    "Mth_data-Mth_model",
    "radon_all-radon_hierarchical_intercept_centered",
    "radon_all-radon_hierarchical_intercept_noncentered",
    "radon_all-radon_variable_intercept_centered",
    "radon_all-radon_variable_intercept_noncentered",
    "radon_all-radon_variable_slope_centered",
    "radon_all-radon_variable_slope_noncentered",
    "radon_all-radon_partially_pooled_centered",
    "radon_all-radon_partially_pooled_noncentered",
    "radon_mod-radon_county",
    "radon_all-radon_county_intercept",
    "radon_mn-radon_variable_intercept_slope_centered",
    "radon_mn-radon_variable_intercept_slope_noncentered",
    "Mtbh_data-Mtbh_model",
    "radon_mn-radon_variable_intercept_centered",
    "radon_mn-radon_variable_intercept_noncentered",
    "radon_mn-radon_variable_slope_centered",
    "radon_mn-radon_variable_slope_noncentered",
    "radon_mn-radon_partially_pooled_centered",
    "radon_mn-radon_partially_pooled_noncentered",
    "radon_mn-radon_hierarchical_intercept_noncentered",
    "radon_mn-radon_county_intercept",
    "election88-election88_full",
    "three_men2-ldaK2",
    "three_men3-ldaK2",
    "three_men1-ldaK2",
    "state_wide_presidential_votes-hierarchical_gp",
    "sat-hier_2pl",
    "hmm_gaussian_simulated-hmm_gaussian",
    "iohmm_reg_simulated-iohmm_reg",
    "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
    "timssAusTwn_irt-gpcm_latent_reg_irt",
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
    "science_irt-grsm_latent_reg_irt",
    "Mh_data-Mh_model",
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
    "GLMM_data-GLMM1_model",
    "irt_2pl-irt_2pl",
    "butterfly-multi_occupancy",
    "rats_data-rats_model",
    "rstan_downloads-prophet",
    "radon_mn-radon_hierarchical_intercept_centered", 
    "diamonds-diamonds",
    "seeds_data-seeds_centered_model",
    "seeds_data-seeds_model",
    "seeds_data-seeds_stanified_model",
    "pilots-pilots",
    "loss_curves-losscurve_sislob",
    "bones_data-bones_model",
    "surgical_data-surgical_model",
]

function run_groundtruth()
    path = "data/raw"

    n_reps = 1
    adtype = AutoMooncake(; config=Mooncake.Config())
    for (name, prob) in vcat(
            [
                ("funnel",   Funnel(10)),
                ("sonar",    LogisticRegressionSonar()),
                ("brownian", BrownianMotion()),
                ("lgcp",     LogGaussianCoxProcess()),
            ],
            collect(
                zip(reverse(posteriordb_problems),
                    reverse(posteriordb_problems))
            )
        )
        failed = false

        fname = joinpath(path, "groundtruth_$(name).jld2")
        if isfile(fname)
            continue
        end

        metadata = (
            name        = name,
            samplers    = [:SMCUHMC],
            n_iters     = [512],
            n_particles = [16384],
            n_subsample = [128],
        )

        data = Dict()
        for n_iters in [512],
            (n_particles, n_subsample) in [(16384, 128)],
            (samplername, sampler_type, adaptor) in [
                (:SMCUHMC, SMCUHMC, BackwardKLMin(; n_subsample=n_subsample, regularization=1.0)),
            ]
        
            config = (
                name        = name,
                sampler     = samplername,
                n_iters     = n_iters,
                n_particles = n_particles,
                n_subsample = n_subsample,
            )

            @info("Running", config...)
            try
                data[config] = run_adaptive_smc(
                    prob,
                    adtype,
                    sampler_type,
                    adaptor,
                    n_particles,
                    n_iters,
                    n_reps;
                    show_progress=true,
                    n_rounds=0,
                    enable_threading=false,
                    prepare_gradient=false,
                )
            catch e
                if e isa InterruptException
                    throw(e)
                end
                @warn "$(name) failed:\n$(e)"
                failed = true
                break
            end
        end
        if !failed
            JLD2.save(fname, "data", data, "metadata", metadata)
        end
    end
end

function run_adaptivesmc_posteriordb_experiments(; show_progress=true)
    path = "data/raw"

    n_reps = 32
    adtype = AutoMooncake(; config=Mooncake.Config())

    for name in reverse(posteriordb_problems)
        failed = false

        fname = joinpath(path, "adaptivesmc_$(name).jld2")
        if isfile(fname)
            continue
        end

        metadata = (
            name        = name,
            samplers    = [:SMCULA, :SMCUHMC],
            n_iters     = [16, 32, 64],
            n_particles = [1024, 256],
            n_subsample = [128, 32],
        )

        data = Dict()
        for n_iters in [16, 32, 64],
            (n_particles, n_subsample) in [(256, 32),
                                           (1024, 128)],
            (samplername, sampler_type, adaptor) in [
                (:SMCULA, SMCULA, BackwardKLMin(; n_subsample=n_subsample, regularization=0.1)),
                (:SMCUHMC, SMCUHMC, BackwardKLMin(; n_subsample=n_subsample, regularization=5.0)),
            ]
        
            config = (
                name        = name,
                sampler     = samplername,
                n_iters     = n_iters,
                n_particles = n_particles,
                n_subsample = n_subsample,
            )

            @info("Running", config...)
            try
                data[config] = run_adaptive_smc(
                    name,
                    adtype,
                    sampler_type,
                    adaptor,
                    n_particles,
                    n_iters,
                    n_reps;
                    show_progress
                )
            catch e
                if e isa InterruptException
                    throw(e)
                end
                @warn "$(name) failed:\n$(e)"
                failed = true
                break
            end
        end
        if !failed
            JLD2.save(fname, "data", data, "metadata", metadata)
        end
    end
end

function run_adaptivesmc_logdensityproblems_experiments(; show_progress=true)
    path = "data/raw"

    n_reps = 32
    adtype = AutoMooncake(; config=Mooncake.Config())

    for (probname, prob) in [
            ("funnel",   Funnel(10)),
            ("sonar",    LogisticRegressionSonar()),
            ("brownian", BrownianMotion()),
            ("lgcp",     LogGaussianCoxProcess()),
        ]
        failed = false

        metadata = (
            name        = probname, 
            samplers    = [:SMCULA, :SMCUHMC],
            n_iters     = [16, 32, 64],
            n_particles = [1024, 256],
            n_subsample = [128, 32],
        )

        fname = joinpath(path, "adaptivesmc_$(probname).jld2")
        if isfile(fname)
            continue
        end

        data = Dict()
        for  n_iters in [16, 32, 64],
            (n_particles, n_subsample) in [
                (256, 32),
                (1024, 128)
            ],
            (samplername, sampler_type, adaptor) in [
                (:SMCULA, SMCULA, BackwardKLMin(; n_subsample=n_subsample, regularization=0.1)),
                (:SMCUHMC, SMCUHMC, BackwardKLMin(; n_subsample=n_subsample, regularization=5.0)),
            ]

            config = (
                name        = probname,
                sampler     = samplername,
                n_iters     = n_iters,
                n_particles = n_particles,
                n_subsample = n_subsample,
            )

            @info("Running", config...)
            try
                data[config] = run_adaptive_smc(
                    prob,
                    adtype,
                    sampler_type,
                    adaptor,
                    n_particles,
                    n_iters,
                    n_reps;
                    show_progress
                )
            catch e
                if e isa InterruptException
                    throw(e)
                end
                @warn "$(name) failed:\n$(e)"
                failed = true
            end
        end
        if !failed
            JLD2.save(fname, "data", data, "metadata", metadata)
        end
    end
end

function process_data()
    for name in [
        "brownian",
        "funnel",
        "sonar",
        "lgcp",
        "bones_data-bones_model",
        "surgical_data-surgical_model",
        "GLMM_data-GLMM1_model",
        "irt_2pl-irt_2pl",
        "butterfly-multi_occupancy",
        "rats_data-rats_model",
        "rstan_downloads-prophet",
        "radon_mn-radon_hierarchical_intercept_centered", 
        "radon_mn-radon_hierarchical_intercept_noncentered",
        "seeds_data-seeds_centered_model",
        "seeds_data-seeds_model",
        "seeds_data-seeds_stanified_model",
        "pilots-pilots",
        "loss_curves-losscurve_sislob",
        "hmm_gaussian_simulated-hmm_gaussian",
        ],
        sampler in [:SMCUHMC, :SMCULA]

        data  = JLD2.load("data/raw/adaptivesmc_$(name).jld2", "data")
        truth = JLD2.load("data/raw/groundtruth_$(name).jld2", "data")

        @info("", name, sampler)
        
        filt(entry) = begin
            k = first(entry)
            true &&
                k.name        == name    &&
                k.sampler     == sampler &&
                k.n_iters     == 64      &&
                k.n_particles == 1024
        end

        rows = filter(filt, data)
        row  = values(rows) |> only

        h5open("data/pro/adaptivesmc_$(name)_$(sampler).h5", "w") do h5
            begin "logZ estimates"
                tups = map(1:3) do i
                    x = quantile([r.n_grad_evals[i]   for r in row], [0.1, 0.5, 0.9])
                    y = quantile([r.log_normalizer[i] for r in row], [0.1, 0.5, 0.9])
                    x, y
                end

                xs = [tup[1] for tup in tups] 
                ys = [tup[2] for tup in tups] 

                xs_veusz = hcat([[x[2], x[3] - x[2], x[2] - x[1]] for x in xs]...)
                ys_veusz = hcat([[y[2], y[3] - y[2], y[2] - y[1]] for y in ys]...)
                
                plot([x[1] for x in xs_veusz], [y[1] for y in ys_veusz]) |> display

                write(h5, "x", xs_veusz)
                write(h5, "y", ys_veusz)

                ℓZ_true = (truth |> values |> first |> first).log_normalizer |> first

                write(h5, "logZ_true", [ℓZ_true])
            end

            begin "stepsizes"
                stepsizes = [last(r.stepsizes)       for r in row]
                schedules = [last(r.schedules) for r in row]
                for (i, id) in enumerate(1:4:length(stepsizes))
                    write(h5, "stepsize_$(i)_x", schedules[id][2:end])
                    write(h5, "stepsize_$(i)_y", stepsizes[id][2:end])
                end
            end

            begin "schedule"
                schedules = [last(r.schedules) for r in row]
                for (i, id) in enumerate(1:4:length(schedules))
                    write(h5, "schedule_$i", schedules[id])
                end
            end

            begin "local barrier"
               local_barriers = [r.local_barrier    for r in row]
               schedules      = [r.schedules[end-1] for r in row]
               for (i, id) in enumerate(1:4:length(local_barriers))
                   write(h5, "localbarrier_$(i)_x", schedules[id])
                   write(h5, "localbarrier_$(i)_y", local_barriers[id])
               end
            end
        end

        #x = [1e-7, 100]
        #y = @chain [out[2], out[3] - out[2], out[1] - out[2]]  begin
        #    reshape(_, (3, 1))
        #    repeat(_, inner=(1, 2))
        #end

        #write(h5, "adaptive_smc_x", x)
        #write(h5, "adaptive_smc_y", y)
    end
end

