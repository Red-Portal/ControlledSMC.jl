
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
    return LogDensityProblems.logdensity(πt, xtm1) -
           LogDensityProblems.logdensity(πtm1, xtm1)
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
    ht, htm1, Γ = stepsizes[t], stepsizes[t - 1], precond
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt = LogDensityProblems.logdensity(πt, xt)
    q_fwd = gradient_flow_euler(πt, xtm1, ht, Γ)
    q_bwd = gradient_flow_euler(πtm1, xt, htm1, Γ)
    K = MvNormal.(eachcol(q_fwd), Ref(2 * ht * Γ))
    L = MvNormal.(eachcol(q_bwd), Ref(2 * htm1 * Γ))
    ℓk = logpdf.(K, eachcol(xt))
    ℓl = logpdf.(L, eachcol(xtm1))
    return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
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
    ht, Γ = stepsizes[t], precond
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt = LogDensityProblems.logdensity(πt, xt)
    q_fwd = gradient_flow_euler(πt, xtm1, ht, Γ)
    q_bwd = gradient_flow_euler(πt, xt, ht, Γ)
    K = MvNormal.(eachcol(q_fwd), Ref(2 * ht * Γ))
    L = MvNormal.(eachcol(q_bwd), Ref(2 * ht * Γ))
    ℓK = logpdf.(K, eachcol(xt))
    ℓL = logpdf.(L, eachcol(xtm1))
    return ℓπt_xt + ℓL - ℓπtm1_xtm1 - ℓK
end

using Plots

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
    n_particles = size(xtm1, 2)
    idx_sub     = StatsBase.sample(rng, 1:n_particles, sampler.adaptor.n_subsample; replace=false)
    xtm1_sub    = xtm1[:, idx_sub]
    ℓwtm1_sub   = ℓwtm1[idx_sub]

    function obj(ℓh′)
        rng_fixed = copy(rng)
        sampler′ = @set sampler.stepsizes[t] = exp(only(ℓh′))

        # If t == 2, also optimize the stepsize at t = 1.
        # For simplicity, we just set h[1] = h[2], which is suboptimal,
        # but shouldn't be too critical.
        if t == 2
            sampler′ = @set sampler′.stepsizes[1] = exp(only(ℓh′))
        end
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub)
    end

    # ℓh_range = range(log(1e-20), log(1); length=256)
    # obj_range = map(ℓh_range) do ℓh′
    #     obj(ℓh′)
    # end
    # Plots.plot(ℓh_range, obj_range) |> display

    if t == 2
        ℓh0 = 0.0
        r   = 0.5

        ## Find any point that is not degenerate
        ℓh, n_feasible_evals = find_feasible_point(obj, ℓh0, log(r), log(1e-10))

        ℓh0 = ℓh

        ## One-step of gradient descent with line-search
        # Approximate gradient
        # Backward difference tests on a smaller stepsize so proabbly more stable:
        # We know that ℓh is feasible but ℓh + δ may not be.
        δ = 1e-5
        ∇obj = (obj(ℓh) - obj(ℓh - δ)) / δ

        stepsize, n_evals_aels = approx_exact_linesearch(obj, ℓh, 1.0, -∇obj)

        # Perform gradient descent
        ℓh = ℓh - stepsize * ∇obj

        ℓh_gd = ℓh

        ## Refine with golden section search
        ℓh_lower, ℓh_upper = ℓh - 2, ℓh + 2
        n_max_iters        = 64
        ℓh, n_gss_iters    = golden_section_search(obj, ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2)

        h     = exp(ℓh)
        stats = (feasible_init_objective_evaluations = n_feasible_evals, linesearch_objective_evaluations    = n_evals_aels, golden_section_search_iterations    = n_gss_iters, ula_stepsize                        = h)

        sampler = @set sampler.stepsizes[1] = h
        sampler = @set sampler.stepsizes[t] = h

        # Plots.vline!([ℓh0],   label="initial") |> display
        # Plots.vline!([ℓh_gd], label="GD+AELS") |> display
        # Plots.vline!([ℓh],    label="GSS")     |> display
        # println(stats)
        # if readline() == "n"
        #     throw()
        # end

        return sampler, stats
    else
        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1
        n_max_iters        = 64

        ℓh, n_gss_iters = golden_section_search(
            obj, ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
        )

        h       = exp(ℓh)
        sampler = @set sampler.stepsizes[t] = h
        stats   = (golden_section_search_iterations=n_gss_iters, ula_stepsize=h)

        # Plots.vline!([ℓh]) |> display
        # if readline() == "n"
        #     throw()
        # end

        return sampler, stats
    end
end
