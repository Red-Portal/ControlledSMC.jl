
struct SMCUBarker{
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Adaptor<:AbstractAdaptor,
} <: AbstractSMC
    stepsizes :: Stepsizes
    backward  :: Backward
    adaptor   :: Adaptor
end

function SMCUBarker(
    stepsize::Real,
    n_steps::Int,
    backward::AbstractBackwardKernel,
    adaptor::AbstractAdaptor,
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
    z       = σt * randn(rng, size(xtm1))
    p_dir   = @. logistic(z * ∇ℓπt_x)
    b       = @. 2 * rand(rng, Bernoulli(p_dir)) - 1
    xt      = xtm1 + b .* z
    ℓG      = potential(sampler, t, πt, πtm1, xt, xtm1)
    return xt, ℓG, NamedTuple()
end

function potential(
    sampler::SMCUBarker, t::Int, πt, πtm1, xt::AbstractMatrix, xtm1::AbstractMatrix
)
    return potential_with_backward(sampler, sampler.backward, t, πt, πtm1, xt, xtm1)
end

function potential_with_backward(
    ::SMCUBarker, ::DetailedBalance, t::Int, πt, πtm1, xt::AbstractMatrix, xtm1::AbstractMatrix
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
    σt, σtm1 = stepsizes[t], stepsizes[t - 1]
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt     = LogDensityProblems.logdensity(πt, xt)
    ∇ℓπt_xtm1 = logdensity_gradient(πt, xtm1)
    ∇ℓπtm1_xt = logdensity_gradient(πtm1, xt)

    ℓk = sum(
        (@. loglogistic(∇ℓπt_xtm1*(xt - xtm1)) + logpdf(Normal(0, σt), xt - xtm1) + log(2)),
        dims=1
    )[1,:]
    ℓl = sum(
        (@. loglogistic(∇ℓπtm1_xt*(xtm1 - xt)) + logpdf(Normal(0, σtm1), xtm1 - xt) + log(2)),
        dims=1
    )[1,:]
    return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
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
    σt, σtm1 = stepsizes[t], stepsizes[t - 1]
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt     = LogDensityProblems.logdensity(πt, xt)
    ∇ℓπt_xtm1 = logdensity_gradient(πt, xtm1)
    ∇ℓπt_xt   = logdensity_gradient(πt, xt)

    ℓk = sum(
        (@. loglogistic(∇ℓπt_xtm1*(xt - xtm1)) + logpdf(Normal(0, σt), xt - xtm1) + log(2)),
        dims=1
    )[1,:]
    ℓl = sum(
        (@. loglogistic(∇ℓπt_xt*(xtm1 - xt)) + logpdf(Normal(0, σtm1), xtm1 - xt) + log(2)),
        dims=1
    )[1,:]
    return ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
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
            ubarker_stepsize                       = exp(ℓh12[2]),
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
        stats   = (golden_section_search_iterations=n_gss_iters, ubarker_stepsize=h)

        return sampler, stats
    end
end
