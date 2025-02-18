
struct SMCULA{
    Path<:AbstractPath,
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Precond<:Union{<:AbstractMatrix,<:UniformScaling},
    Adaptor<:Union{Nothing,<:AbstractAdaptor},
} <: AbstractSMC
    path      :: Path
    stepsizes :: Stepsizes
    backward  :: Backward
    precond   :: Precond
    adaptor   :: Adaptor
end

Base.length(sampler::SMCULA) = length(sampler.path)

function SMCULA(
    path::AbstractPath,
    adaptor::AbstractAdaptor;
    precond::Union{<:AbstractMatrix,<:UniformScaling}=I,
    backward::AbstractBackwardKernel=TimeCorrectForwardKernel(),
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
    @assert all(@. 0 < stepsize)
    return SMCULA{typeof(path),typeof(stepsize),typeof(backward),typeof(precond),Nothing}(
        path, stepsize, backward, precond, nothing
    )
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, n_particles::Int
)
    x  = rand(rng, sampler.path.proposal, n_particles)
    ℓG = zeros(eltype(eltype(x)), n_particles)
    return x, ℓG
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, t::Int, xtm1::AbstractMatrix
)
    (; path, stepsizes, precond) = sampler
    πt = get_target(path, t)
    ht = stepsizes[t]
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond

    q  = gradient_flow_euler(πt, xtm1, ht, Γ)
    K  = BatchMvNormal(q, 2 * ht * Γ)
    xt = rand(rng, K)
    ℓG = potential(sampler, t, xt, xtm1, K)
    return xt, ℓG, (q=q,)
end

function potential(
    sampler::SMCULA, t::Int, xt::AbstractMatrix, xtm1::AbstractMatrix, K::BatchMvNormal
)
    return potential_with_backward(sampler, sampler.backward, t, xt, xtm1, K)
end

function potential_with_backward(
    sampler::SMCULA,
    ::DetailedBalance,
    t::Int,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
    ::BatchMvNormal,
)
    πt   = get_target(sampler.path, t)
    πtm1 = get_target(sampler.path, t - 1)
    return logdensity_safe(πt, xtm1) - logdensity_safe(πtm1, xtm1)
end

function potential_with_backward(
    sampler::SMCULA,
    ::TimeCorrectForwardKernel,
    t::Int,
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
    K::BatchMvNormal,
)
    (; path, stepsizes, precond) = sampler
    πt = get_target(path, t)
    πtm1 = get_target(path, t - 1)
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond
    ℓπt_xt = logdensity_safe(πt, xt)
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
    xt::AbstractMatrix,
    xtm1::AbstractMatrix,
    K::BatchMvNormal,
)
    (; path, stepsizes, precond) = sampler
    ht = stepsizes[t]
    Γ = precond isa UniformScaling ? precond(size(xtm1, 1)) : precond
    πt = get_target(path, t)
    πtm1 = get_target(path, t - 1)

    ℓπt_xt = logdensity_safe(πt, xt)
    ℓk     = logpdf(K, xt)

    if t == 1
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
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if isnothing(sampler.adaptor)
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

    function obj(ℓh′)
        rng_fixed    = copy(rng)
        sampler′     = @set sampler.stepsizes[t] = exp(ℓh′)
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, xtm1_sub)
        reg          = if t == 1
            τ * abs2(ℓh′)
        else
            ℓh_prev = log(sampler.stepsizes[t - 1])
            τ * abs2(ℓh′ - ℓh_prev)
        end
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) + reg
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

    h = exp(ℓh)

    stats = (ula_stepsize=h, n_objective_evals=n_evals_total)

    sampler = @set sampler.stepsizes[t] = h
    return sampler, stats
end
