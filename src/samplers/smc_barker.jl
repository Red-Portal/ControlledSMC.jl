
struct SMCBarker{
    Stepsizes<:AbstractVector,Precond<:AbstractMatrix,Adaptor<:AbstractAdaptor
} <: AbstractSMC
    stepsizes :: Stepsizes
    precond   :: Precond
    adaptor   :: Adaptor
end

function SMCBarker(
    stepsize::Real,
    n_steps::Int,
    precond::AbstractMatrix,
    adaptor::Union{<:NoAdaptation,<:AcceptanceRateCtrl,<:ESJDMax},
)
    stepsizes = Fill(stepsize, n_steps)
    return SMCBarker{typeof(stepsizes),typeof(precond),typeof(adaptor)}(
        stepsizes, precond, adaptor
    )
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
    (; stepsizes,) = sampler
    σt = stepsizes[t]

    xt, _ = transition_barker(rng, σt, πt, xtm1)
    ℓG    = potential(sampler, t, πt, πtm1, xtm1)
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

    Γ = sampler.precond

    function obj(ℓh′)
        rng_fixed = copy(rng)
        xt_sub, α = transition_mala(rng_fixed, exp(ℓh′), Γ, πt, xtm1_sub)
        esjd      = mean(sum(abs2.(xt_sub - xtm1_sub); dims=1))
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓwtm1_sub, α, esjd)
    end

    if t == 2
        ℓh0 = 0.0
        r   = 0.5

        ## Find any point that is not degenerate
        ℓh, n_feasible_evals = find_feasible_point(obj, ℓh0, log(r), log(1e-10))

        # Properly optimize the objective
        res = Optim.optimize(
            params -> obj(only(params)),
            [ℓh],
            GradientDescent(),
            Optim.Options(; x_tol=1e-2, iterations=30),
        )
        ℓh = only(Optim.minimizer(res))
        h = exp(ℓh)

        _, α = transition_mala(rng, h, Γ, πt, xtm1_sub)

        stats = (
            initialization_objective_evaluations   = n_feasible_evals,
            gradient_descent_objective_evaluations = Optim.f_calls(res),
            ula_stepsize                           = h,
        )
        sampler = @set sampler.stepsizes[t] = exp(ℓh)
        return sampler, stats
    else
        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1
        n_max_iters        = 64

        ℓh, n_gss_iters = golden_section_search(
            obj, ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
        )

        h = exp(ℓh)

        _, α = transition_mala(rng, exp(ℓh), Γ, πt, xtm1_sub)

        sampler = @set sampler.stepsizes[t] = h
        stats   = (golden_section_search_iterations=n_gss_iters, mala_stepsize=h, acceptance_rate=α)
        return sampler, stats
    end
end
