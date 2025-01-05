
struct SMCUBarker{
    Stepsizes<:AbstractVector,Backward<:AbstractBackwardKernel,Adaptor<:AbstractAdaptor
} <: AbstractSMC
    stepsizes :: Stepsizes
    backward  :: Backward
    adaptor   :: Adaptor
end

function SMCUBarker(
    stepsize::Real, n_steps::Int, backward::AbstractBackwardKernel, adaptor::AbstractAdaptor
)
    stepsizes = Fill(stepsize, n_steps)
    return SMCUBarker{typeof(stepsizes),typeof(backward),typeof(adaptor)}(
        stepsizes, backward, adaptor
    )
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCUBarker, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes,) = sampler
    σt = stepsizes[t]

    ∇ℓπt_x = logdensity_gradient(πt, xtm1)
    z = σt * randn(rng, size(xtm1))
    p_dir = @. logistic(z * ∇ℓπt_x)
    b = @. 2 * rand(rng, Bernoulli(p_dir)) - 1
    xt = xtm1 + b .* z
    ℓG = potential(sampler, t, πt, πtm1, xt, xtm1)
    return xt, ℓG, NamedTuple()
end

function potential(
    sampler::SMCUBarker, t::Int, πt, πtm1, xt::AbstractMatrix, xtm1::AbstractMatrix
)
    return potential_with_backward(sampler, sampler.backward, t, πt, πtm1, xt, xtm1)
end

function potential_with_backward(
    ::SMCUBarker,
    ::DetailedBalance,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
)
    return LogDensityProblems.logdensity(πt, xtm1) -
           LogDensityProblems.logdensity(πtm1, xtm1)
end

function potential_with_backward(
    sampler::SMCUBarker,
    ::TimeCorrectForwardKernel,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
)
    (; stepsizes,) = sampler
    σt = stepsizes[t]
    ℓπt_xt = LogDensityProblems.logdensity(πt, xt)
    ∇ℓπt_xtm1 = logdensity_gradient(πt, xtm1)
    ℓk = sum(
        (@. loglogistic(∇ℓπt_xtm1 * (xt - xtm1)) +
            logpdf(Normal(0, σt), xt - xtm1) +
            log(2));
        dims=1,
    )[
        1, :,
    ]

    if t == 2
        return ℓπt_xt - ℓk
    else
        σtm1       = stepsizes[t - 1]
        ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
        ∇ℓπtm1_xt  = logdensity_gradient(πtm1, xt)

        ℓl = sum(
            (@. loglogistic(∇ℓπtm1_xt * (xtm1 - xt)) +
                logpdf(Normal(0, σtm1), xtm1 - xt) +
                log(2));
            dims=1,
        )[
            1, :,
        ]
        return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
    end
end

function potential_with_backward(
    sampler::SMCUBarker,
    ::ForwardKernel,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
)
    (; stepsizes,) = sampler
    σt = stepsizes[t]
    ℓπt_xt = LogDensityProblems.logdensity(πt, xt)
    ∇ℓπt_xtm1 = logdensity_gradient(πt, xtm1)
    ℓk = sum(
        (@. loglogistic(∇ℓπt_xtm1 * (xt - xtm1)) +
            logpdf(Normal(0, σt), xt - xtm1) +
            log(2));
        dims=1,
    )[
        1, :,
    ]

    if t == 2
        return ℓπt_xt - ℓk
    else
        σtm1       = stepsizes[t - 1]
        ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
        ∇ℓπt_xt    = logdensity_gradient(πt, xt)

        ℓl = sum(
            (@. loglogistic(∇ℓπt_xt * (xtm1 - xt)) +
                logpdf(Normal(0, σtm1), xtm1 - xt) +
                log(2));
            dims=1,
        )[
            1, :,
        ]
        return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
    end
end

using Plots

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCUBarker,
    t::Int,
    πt,
    πtm1,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if sampler.adaptor isa NoAdaptation
        return sampler, NamedTuple()
    end

    # Subsample particles to reduce adaptation overhead
    n_particles = size(xtm1, 2)
    idx_sub     = StatsBase.sample(rng, 1:n_particles, sampler.adaptor.n_subsample; replace=false)
    xtm1_sub    = xtm1[:, idx_sub]
    ℓwtm1_sub   = ℓwtm1[idx_sub]

    if t == 2
        function obj_init(ℓh′)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub) +
                   0.01 * abs2(ℓh′)
        end

        ℓh_lower_guess = -15.0

        ## Find any point between 1e-10 and 2e-16 that is not degenerate
        ℓh_decrease_stepsize = log(0.5)
        ℓh_lower, n_feasible_evals = find_feasible_point(
            obj_init, ℓh_lower_guess, ℓh_decrease_stepsize, log(eps(Float64))
        )

        ## Find remaining endpoint of an interval containing a (possibly local) minima
        ℓh_upper_increase_ratio = 1.2
        n_interval_max_iters = ceil(Int, log(ℓh_upper_increase_ratio, 20))
        ℓh_upper, _, n_interval_evals = find_golden_section_search_interval(
            obj_init, ℓh_lower, ℓh_upper_increase_ratio, 1; n_max_iters=n_interval_max_iters
        )

        # Properly optimize the objective
        ℓh, n_gss_iters = golden_section_search(obj_init, ℓh_lower, ℓh_upper; abstol=1e-2)
        h = exp(ℓh)

        stats = (
            feasible_search_objective_evaluations       = n_feasible_evals,
            initialization_objective_evaluations        = n_interval_evals,
            golden_section_search_objective_evaluations = n_gss_iters,
            ubarker_stepsize                            = h,
        )
        sampler = @set sampler.stepsizes[2] = h
        return sampler, stats
    else
        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1

        function obj(ℓh′)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub) +
                   0.01 * abs2(ℓh′ - ℓh_prev)
        end

        ℓh, n_gss_iters = golden_section_search(obj, ℓh_lower, ℓh_upper; abstol=1e-2)

        h       = exp(ℓh)
        sampler = @set sampler.stepsizes[t] = h
        stats   = (golden_section_search_iterations=n_gss_iters, ubarker_stepsize=h)

        return sampler, stats
    end
end
