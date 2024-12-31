
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

    # ℓh_range = range(log(1e-20), log(1); length=256)
    # obj_range = map(ℓh_range) do ℓh′
    #     obj(ℓh′)
    # end
    # Plots.plot(ℓh_range, obj_range) |> display

    if t == 2
        function obj_init(ℓh12)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[1] = exp(ℓh12[1])
            sampler′ = @set sampler′.stepsizes[2] = exp(ℓh12[2])
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub)
        end

        ℓh_init = 0.0
        r       = 0.5

        # Find any point that is not degenerate
        ℓh_init, n_feasible_evals = find_feasible_point(
            ℓh′ -> obj_init([ℓh′, ℓh′]), ℓh_init, log(r), log(1e-10)
        )

        # Properly optimize the objective
        res = Optim.optimize(
            obj_init,
            [ℓh_init, ℓh_init],
            GradientDescent(),
            Optim.Options(; x_tol=1e-2, iterations=30),
        )
        ℓh12 = Optim.minimizer(res)

        stats = (
            initialization_objective_evaluations   = n_feasible_evals,
            gradient_descent_objective_evaluations = Optim.f_calls(res),
            ula_stepsize                           = exp(ℓh12[2]),
        )

        sampler = @set sampler.stepsizes[1] = exp(ℓh12[1])
        sampler = @set sampler.stepsizes[2] = exp(ℓh12[2])
        return sampler, stats
    else
        function obj(ℓh′)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(only(ℓh′))
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub)
        end

        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1
        n_max_iters        = 64

        ℓh, n_gss_iters = golden_section_search(
            obj, ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
        )

        h       = exp(ℓh)
        sampler = @set sampler.stepsizes[t] = h
        stats   = (golden_section_search_iterations=n_gss_iters, ula_stepsize=h)

        return sampler, stats
    end
end
