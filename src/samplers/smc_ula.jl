
struct SMCULA{
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Precond<:AbstractMatrix,
    Adaptor<:AbstractAdaptor,
} <: AbstractSMC
    stepsizes :: Stepsizes
    backward  :: Backward
    precond   :: Precond
    adaptor   :: Adaptor
end

function SMCULA(
    stepsize::Real,
    n_steps::Int,
    backward::AbstractBackwardKernel,
    precond::AbstractMatrix,
    adaptor::AbstractAdaptor,
)
    stepsizes = Fill(stepsize, n_steps)
    return SMCULA{typeof(stepsizes),typeof(backward),typeof(precond),typeof(adaptor)}(
        stepsizes, backward, precond, adaptor
    )
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes, precond) = sampler
    ht, Γ = stepsizes[t], precond
    q = gradient_flow_euler(πt, xtm1, ht, Γ)
    xt = q + sqrt(2 * ht) * unwhiten(Γ, randn(rng, eltype(q), size(q)))
    ℓG = potential(sampler, t, πt, πtm1, xt, xtm1)
    return xt, ℓG, (q=q,)
end

function potential(
    sampler::SMCULA, t::Int, πt, πtm1, xt::AbstractMatrix, xtm1::AbstractMatrix
)
    return potential_with_backward(sampler, sampler.backward, t, πt, πtm1, xt, xtm1)
end

function potential_with_backward(
    ::SMCULA, ::DetailedBalance, t::Int, πt, πtm1, xt::AbstractMatrix, xtm1::AbstractMatrix
)
    return logdensity_safe(πt, xtm1) - logdensity_safe(πtm1, xtm1)
end

function potential_with_backward(
    sampler::SMCULA,
    ::TimeCorrectForwardKernel,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
)
    (; stepsizes, precond) = sampler

    ht, Γ  = stepsizes[t], precond
    ℓπt_xt = logdensity_safe(πt, xt)
    q_fwd  = gradient_flow_euler(πt, xtm1, ht, Γ)
    K      = BatchMvNormal(q_fwd, 2 * ht * Γ)
    ℓk     = logpdf(K, xt)

    if t == 2
        return ℓπt_xt - ℓk
    else
        htm1       = stepsizes[t - 1]
        ℓπtm1_xtm1 = logdensity_safe(πtm1, xtm1)
        q_bwd      = gradient_flow_euler(πtm1, xt, htm1, Γ)
        L          = BatchMvNormal(q_bwd, 2 * htm1 * Γ)
        ℓl         = logpdf(L, xtm1)
        return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
    end
end

function potential_with_backward(
    sampler::SMCULA,
    ::ForwardKernel,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
)
    (; stepsizes, precond) = sampler

    ht, Γ  = stepsizes[t], precond
    ℓπt_xt = logdensity_safe(πt, xt)
    q_fwd  = gradient_flow_euler(πt, xtm1, ht, Γ)
    K      = BatchMvNormal(q_fwd, 2 * ht * Γ)
    ℓk     = logpdf(K, xt)

    if t == 2
        return ℓπt_xt - ℓk
    else
        ℓπtm1_xtm1 = logdensity_safe(πtm1, xtm1)
        q_bwd      = gradient_flow_euler(πt, xt, ht, Γ)
        L          = BatchMvNormal(q_bwd, 2 * ht * Γ)
        ℓl         = logpdf(L, xtm1)
        return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
    end
end

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCULA,
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
    w_norm    = exp.(ℓwtm1 .- logsumexp(ℓwtm1))
    n_sub     = sampler.adaptor.n_subsample
    sub_idx   = systematic_sampling(rng, w_norm, n_sub)
    xtm1_sub  = xtm1[:, sub_idx]
    ℓdPdQ_sub = ℓwtm1[sub_idx]
    ℓwtm1_sub = fill(-log(n_sub), n_sub)

    if t == 2
        function obj_init(ℓh′)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) +
                sampler.adaptor.regularization * abs2(ℓh′)
        end

        ℓh_lower_guess = -15.0

        ## Find any point between 1e-10 and 2e-16 that is not degenerate
        ℓh_decrease_stepsize = -1
        ℓh_lower, n_feasible_evals = find_feasible_point(
            obj_init, ℓh_lower_guess, ℓh_decrease_stepsize, log(eps(Float64))
        )

        ## Find remaining endpoint of an interval containing a (possibly local) minima
        ℓh_upper_increase_coeff = 0.1
        ℓh_upper_increase_ratio = 1.5
        ℓh_upper, _, n_interval_evals = find_golden_section_search_interval(
            obj_init, ℓh_lower, ℓh_upper_increase_coeff, ℓh_upper_increase_ratio
        )

        # Properly optimize the objective
        ℓh, n_gss_iters = golden_section_search(obj_init, ℓh_lower, ℓh_upper; abstol=1e-2)
        h = exp(ℓh)

        # ℓh_range = range(-15, 0; length=32)
        # obj_range = @showprogress map(obj_init, ℓh_range)
        # Plots.plot(ℓh_range, obj_range, yscale=:log10) |> display
        # Plots.vline!([ℓh_lower, ℓh_upper]) |> display
        # Plots.vline!([ℓh]) |> display
        # throw()

        stats = (
            feasible_search_objective_evaluations       = n_feasible_evals,
            initialization_objective_evaluations        = n_interval_evals,
            golden_section_search_objective_evaluations = n_gss_iters,
            ula_stepsize                                = h,
        )

        sampler = @set sampler.stepsizes[2] = h
        return sampler, stats
    else
        ℓh_prev = log(sampler.stepsizes[t - 1])

        function obj(ℓh′)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) +
                   sampler.adaptor.regularization * abs2(ℓh′ - ℓh_prev)
        end

        ℓh_change_coeff = 0.1
        ℓh_change_ratio = 1.5
        ℓh_upper, _, n_upper_bound_evals = find_golden_section_search_interval(
            obj, ℓh_prev, ℓh_change_coeff, ℓh_change_ratio
        )
        ℓh_lower, _, n_lower_bound_evals = find_golden_section_search_interval(
            obj, ℓh_upper, -ℓh_change_coeff, ℓh_change_ratio
        )

        ℓh, n_gss_iters = golden_section_search(obj, ℓh_lower, ℓh_upper; abstol=1e-2)

        h       = exp(ℓh)
        sampler = @set sampler.stepsizes[t] = h
        stats   = (
            h_upper_bound                    = exp(ℓh_upper),
            h_lower_bound                    = exp(ℓh_lower),
            interval_objective_evaluations   = n_upper_bound_evals + n_lower_bound_evals,
            golden_section_search_iterations = n_gss_iters,
            ula_stepsize                     = h
        )

        return sampler, stats
    end
end
