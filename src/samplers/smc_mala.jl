
struct SMCMALA{
    Path<:AbstractPath,
    Stepsizes<:AbstractVector,
    Precond<:Union{<:AbstractMatrix,<:UniformScaling},
    Adaptor<:Union{<:AbstractAdaptor,Nothing},
} <: AbstractSMC
    path         :: Path
    stepsizes    :: Stepsizes
    precond      :: Precond
    adaptor      :: Adaptor
    n_mcmc_steps :: Int
end

Base.length(sampler::SMCMALA) = length(sampler.path)

function SMCMALA(
    path::AbstractPath,
    adaptor::Union{<:AcceptanceRateControl,<:ESJDMax};
    precond::Union{<:AbstractMatrix,<:UniformScaling}=I,
    n_mcmc_steps::Int=1,
)
    stepsizes = zeros(length(path))
    return SMCMALA{typeof(path),typeof(stepsizes),typeof(precond),typeof(adaptor)}(
        path, stepsizes, precond, adaptor, n_mcmc_steps
    )
end

function SMCMALA(
    path::AbstractPath,
    stepsize::Union{Real,<:AbstractVector};
    precond::Union{<:AbstractMatrix,<:UniformScaling}=I,
    n_mcmc_steps::Int=1,
)
    if stepsize isa Real
        stepsize = fill(stepsize, length(path))
    end
    @assert length(stepsize) == length(path)
    @assert all(@. 0 < stepsize)
    return SMCMALA{typeof(path),typeof(stepsize),typeof(precond),Nothing}(
        path, stepsize, precond, nothing, n_mcmc_steps
    )
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::SMCMALA, n_particles::Int
)
    x  = rand(rng, sampler.path.proposal, n_particles)
    ℓG = zeros(eltype(eltype(x)), n_particles)
    return x, ℓG
end

function transition_mala(rng::Random.AbstractRNG, h, Γ, πt, x::AbstractMatrix)
    n_particles = size(x, 2)

    μ_fwd  = gradient_flow_euler(πt, x, h, Γ)
    x_prop = μ_fwd + sqrt(2 * h) * unwhiten(Γ, randn(rng, eltype(μ_fwd), size(μ_fwd)))

    μ_bwd = gradient_flow_euler(πt, x_prop, h, Γ)

    ℓπt_prop = logdensity_safe(πt, x_prop)
    ℓπt      = logdensity_safe(πt, x)

    q_fwd  = BatchMvNormal(μ_fwd, 2 * h * Γ)
    q_bwd  = BatchMvNormal(μ_bwd, 2 * h * Γ)
    ℓq_fwd = logpdf(q_fwd, x_prop)
    ℓq_bwd = logpdf(q_bwd, x)

    ℓα = @. min(ℓπt_prop - ℓπt + ℓq_bwd - ℓq_fwd, 0)
    ℓu = -Random.randexp(rng, length(ℓα))

    x_next = mapreduce(hcat, 1:size(x, 2)) do n
        if ℓα[n] > ℓu[n]
            x_prop[:, n]
        else
            x[:, n]
        end
    end
    ℓEα = logsumexp(ℓα) - log(n_particles)
    return x_next, ℓEα
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCMALA, t::Int, xtm1::AbstractMatrix
)
    (; path, stepsizes, precond, n_mcmc_steps) = sampler
    πt = get_target(path, t)
    πtm1 = get_target(path, t - 1)
    ht = stepsizes[t]
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond

    xt = xtm1
    for _ in 1:n_mcmc_steps
        xt, _ = transition_mala(rng, ht, Γ, πt, xt)
    end
    ℓG = potential(sampler, t, πt, πtm1, xtm1)
    return xt, ℓG, NamedTuple()
end

function potential(::SMCMALA, t::Int, πt, πtm1, xtm1::AbstractMatrix)
    return logdensity_safe(πt, xtm1) - logdensity_safe(πtm1, xtm1)
end

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCMALA,
    t::Int,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if isnothing(sampler.adaptor)
        return sampler, NamedTuple()
    end
    path = sampler.path
    πt   = get_target(path, t)

    # Subsample particles to reduce adaptation overhead
    n_particles = size(xtm1, 2)
    idx_sub     = StatsBase.sample(rng, 1:n_particles, sampler.adaptor.n_subsample; replace=false)
    xtm1_sub    = xtm1[:, idx_sub]

    precond = sampler.precond
    Γ       = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond

    τ = sampler.adaptor.regularization

    function obj(ℓh′)
        rng_fixed  = copy(rng)
        xt_sub, ℓα = transition_mala(rng_fixed, exp(ℓh′), Γ, πt, xtm1_sub)
        esjd       = mean(sum(abs2.(xt_sub - xtm1_sub); dims=1))
        reg        = if t == 1
            τ * abs2(ℓh′)
        else
            ℓh_prev = log(sampler.stepsizes[t - 1])
            τ * abs2(ℓh′ - ℓh_prev)
        end
        return adaptation_objective(sampler.adaptor, ℓα, esjd) + reg
    end

    r = 2.0
    c = 0.1
    ϵ = 1e-2
    δ = -1
    ℓh_guess = -10.0
    n_evals_total = 0

    ℓh = if t == 1
        ℓh, n_evals = find_feasible_point(obj, ℓh_guess, δ, log(eps(eltype(xtm1))))
        n_evals_total += n_evals
        ℓh
    else
        log(sampler.stepsizes[t - 1])
    end
    ℓh, n_evals = minimize(obj, ℓh, c, r, ϵ)
    n_evals_total += n_evals

    h     = exp(ℓh)
    _, ℓα = transition_mala(rng, exp(ℓh), Γ, πt, xtm1_sub)

    stats = (mala_stepsize=h, acceptance_rate=exp(ℓα), n_objective_evals=n_evals_total)

    # Consume rng states so that the actual mutation step is less biased.
    rand(rng, size(xtm1))

    sampler = @set sampler.stepsizes[t] = h
    return sampler, stats
end
