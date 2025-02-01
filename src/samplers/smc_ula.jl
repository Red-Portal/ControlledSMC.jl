
struct SMCULA{
    Path<:AbstractPath,
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Precond<:Union{<:AbstractMatrix,<:UniformScaling},
    Adaptor<:AbstractAdaptor,
} <: AbstractSMC
    path      :: Path
    stepsizes :: Stepsizes
    backward  :: Backward
    precond   :: Precond
    adaptor   :: Adaptor
end

function SMCULA(
    path::AbstractPath;
    precond::Union{<:AbstractMatrix,<:UniformScaling}=I,
    backward::AbstractBackwardKernel=TimeCorrectForwardKernel(),
    adaptor::AbstractAdaptor,
)
    stepsizes = zeros(Float64, length(path))
    return SMCULA{
        typeof(path),typeof(stepsizes),typeof(backward),typeof(precond),typeof(adaptor)
    }(
        path, stepsizes, backward, precond, adaptor
    )
end

function SMCULA(
    path::AbstractPath,
    stepsize::Union{Real,<:AbstractVector};
    backward::AbstractBackwardKernel=TimeCorrectForwardKernel(),
    precond::Union{<:AbstractMatrix,<:UniformScaling}=I,
)
    if stepsize isa Real
        stepsize = fill(stepsize, length(path))
    end
    @assert length(stepsize) == length(path)
    adaptor = NoAdaptation()
    return SMCULA{
        typeof(path),typeof(stepsize),typeof(backward),typeof(precond),typeof(adaptor)
    }(
        path, stepsize, backward, precond, adaptor
    )
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes, precond) = sampler
    ht = stepsizes[t]
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond
    q = gradient_flow_euler(πt, xtm1, ht, Γ)
    xt = q + sqrt(2 * ht) * unwhiten(Γ, randn(rng, eltype(q), size(q)))
    ℓG = potential(sampler, t, πt, πtm1, xt, xtm1, q)
    return xt, ℓG, (q=q,)
end

function potential(
    sampler::SMCULA,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
    q_fwd::AbstractMatrix,
)
    return potential_with_backward(sampler, sampler.backward, t, πt, πtm1, xt, xtm1, q_fwd)
end

function potential_with_backward(
    ::SMCULA,
    ::DetailedBalance,
    t::Int,
    πt,
    πtm1,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
    ::AbstractMatrix,
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
    q_fwd::AbstractMatrix,
)
    (; stepsizes, precond) = sampler
    ht = stepsizes[t]
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond
    ℓπt_xt = logdensity_safe(πt, xt)
    K = BatchMvNormal(q_fwd, 2 * ht * Γ)
    ℓk = logpdf(K, xt)

    if t == 1
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
    q_fwd::AbstractMatrix,
)
    (; stepsizes, precond) = sampler

    ht, Γ  = stepsizes[t], precond
    ℓπt_xt = logdensity_safe(πt, xt)
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
    w_norm    = exp.(ℓwtm1 .- logsumexp(ℓwtm1))
    n_sub     = sampler.adaptor.n_subsample
    sub_idx   = systematic_sampling(rng, w_norm, n_sub)
    xtm1_sub  = xtm1[:, sub_idx]
    ℓdPdQ_sub = ℓwtm1[sub_idx]
    ℓwtm1_sub = fill(-log(n_sub), n_sub)

    τ = sampler.adaptor.regularization

    function obj_init(ℓh′)
        rng_fixed    = copy(rng)
        sampler′     = @set sampler.stepsizes[t] = exp(ℓh′)
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
        reg          = if t == 1
            τ * abs2(ℓh′)
        else
            ℓh_prev = log(sampler.stepsizes[t - 1])
            τ * abs2(ℓh′ - ℓh_prev)
        end
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) + reg
    end

    r = 1.5
    c = 0.3
    ϵ = 1e-2
    δ = -1
    ℓh_guess = -15.0
    n_evals_total = 0

    ℓh = if t == 1
        ℓh, n_evals = find_feasible_point(obj_init, ℓh_guess, δ, log(eps(eltype(xtm1))))
        n_evals_total += n_evals
        ℓh
    else
        sampler.stepsizes[t - 1]
    end
    ℓh, n_evals = minimize(obj_init, ℓh, c, r, ϵ)
    n_evals_total += n_evals

    h = exp(ℓh)

    stats = (ula_stepsize=h, n_objective_evals=n_evals_total)

    # Consume rngs so that the actual mutation step is less biased.
    rand(rng, size(xtm1))

    sampler = @set sampler.stepsizes[t] = h
    return sampler, stats
end
