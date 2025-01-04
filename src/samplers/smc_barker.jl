
struct SMCBarker{Stepsizes<:AbstractVector,Adaptor<:AbstractAdaptor} <: AbstractSMC
    stepsizes    :: Stepsizes
    adaptor      :: Adaptor
    n_mcmc_steps :: Int
end

function SMCBarker(
    stepsize::Real,
    n_steps::Int,
    adaptor::Union{<:NoAdaptation,<:AcceptanceRateCtrl,<:ESJDMax},
    n_mcmc_steps::Int=1,
)
    stepsizes = Fill(stepsize, n_steps)
    return SMCBarker{typeof(stepsizes),typeof(adaptor)}(stepsizes, adaptor, n_mcmc_steps)
end

function transition_barker(rng::Random.AbstractRNG, σ, π, x::AbstractMatrix)
    n_particles = size(x, 2)

    ∇ℓπ_x = logdensity_gradient(π, x)
    ℓπ_x = LogDensityProblems.logdensity(π, x)
    z = σ * randn(rng, size(x))
    p_dir = @. logistic(z * ∇ℓπ_x)
    b = @. 2 * rand(rng, Bernoulli(p_dir)) - 1
    y = x + b .* z
    ∇ℓπ_y = logdensity_gradient(π, y)
    ℓπ_y = LogDensityProblems.logdensity(π, y)

    ℓq_fwd = sum((@. log1pexp((y - x) * ∇ℓπ_y)); dims=1)[1, :]
    ℓq_bwd = sum((@. log1pexp((x - y) * ∇ℓπ_x)); dims=1)[1, :]

    ℓα = @. min(ℓπ_y - ℓπ_x + ℓq_bwd - ℓq_fwd, 0)
    ℓu = -Random.randexp(rng, n_particles)

    x_next = mapreduce(hcat, 1:size(x, 2)) do n
        if ℓα[n] > ℓu[n]
            y[:, n]
        else
            x[:, n]
        end
    end
    α = exp(logsumexp(ℓα) - log(n_particles))
    return x_next, α
end

function potential(::SMCBarker, t::Int, πt, πtm1, xtm1::AbstractMatrix)
    ℓπt_xtm1   = LogDensityProblems.logdensity(πt, xtm1)
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    return ℓπt_xtm1 - ℓπtm1_xtm1
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCBarker, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes, n_mcmc_steps) = sampler
    σt = stepsizes[t]

    xt = xtm1
    for _ in 1:n_mcmc_steps
        xt, _ = transition_barker(rng, σt, πt, xt)
    end
    ℓG = potential(sampler, t, πt, πtm1, xtm1)
    return xt, ℓG, NamedTuple()
end

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCBarker,
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
            xt_sub, α = transition_barker(rng_fixed, exp(ℓh′), πt, xtm1_sub)
            esjd      = mean(sum(abs2.(xt_sub - xtm1_sub); dims=1))
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓwtm1_sub, α, esjd) +
                   0.01 * abs2(ℓh′)
        end

        ℓh_lower_guess = -20.0

        ## Find any point between 1e-10 and 2e-16 that is not degenerate
        ℓh_decrease_stepsize = log(0.5)
        ℓh_lower, n_feasible_evals = find_feasible_point(
            obj_init, ℓh_lower_guess, ℓh_decrease_stepsize, log(eps(Float64))
        )

        ## Find an interval that contains a (possibly local) minima
        ℓh_upper_increase_ratio = (1 + √5) / 2
        n_interval_max_iters = ceil(Int, log(ℓh_upper_increase_ratio, 40))
        ℓh_upper, _, n_interval_evals = find_golden_section_search_interval(
            obj_init, ℓh_lower, ℓh_upper_increase_ratio, 1; n_max_iters=n_interval_max_iters
        )

        ## Properly optimize objective
        gss_abstol = 1e-2
        ℓh, n_gss_iters = golden_section_search(
            obj_init, ℓh_lower, ℓh_upper; abstol=gss_abstol
        )
        h = exp(ℓh)

        _, α = transition_barker(rng, h, πt, xtm1_sub)

        stats = (
            feasible_lowerbound_search_obj_evals = n_feasible_evals,
            bracketing_interval_search_obj_evals = n_interval_evals,
            golden_section_search_iters          = n_gss_iters,
            barker_stepsize                      = h,
        )
        sampler = @set sampler.stepsizes[t] = exp(ℓh)
        return sampler, stats
    else
        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1

        function obj(ℓh′)
            rng_fixed = copy(rng)
            xt_sub, α = transition_barker(rng_fixed, exp(ℓh′), πt, xtm1_sub)
            esjd      = mean(sum(abs2.(xt_sub - xtm1_sub); dims=1))
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓwtm1_sub, α, esjd) +
                   0.01 * abs2(ℓh′ - ℓh_prev)
        end

        gss_abstol = 1e-2
        ℓh, n_gss_iters = golden_section_search(obj, ℓh_lower, ℓh_upper; abstol=gss_abstol)
        h = exp(ℓh)

        _, α = transition_barker(rng, exp(ℓh), πt, xtm1_sub)

        sampler = @set sampler.stepsizes[t] = h
        stats   = (golden_section_search_iterations=n_gss_iters, barker_stepsize=h, acceptance_rate=α)
        return sampler, stats
    end
end
