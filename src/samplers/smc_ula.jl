
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

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCULA,
    t::Int,
    πt::Any,
    πtm1::Any,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if sampler.adaptor isa NoAdaptation
        return sampler, NamedTuple()
    end

    ℓh_lower, ℓh_upper = if t == 2
        log(1e-7), log(10)
    else
        ℓh_prev = log(sampler.stepsizes[t - 1])
        ℓh_prev - 1, ℓh_prev + 1
    end

    n_max_iters = (2 == 1) ? 64 : 16

    ℓh, n_gss_iters = golden_section_search(
        ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
    ) do ℓh′
        rng_fixed = copy(rng)
        sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)

        # If t == 2, also optimize the stepsize at t = 1
        if t == 2
            sampler′ = @set sampler.stepsizes[1] = exp(ℓh′)
        end
        _, ℓG, _  = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1)
        adaptation_objective(sampler.adaptor, ℓwtm1, ℓG)
    end

    h       = exp(ℓh)
    sampler = @set sampler.stepsizes[t] = exp(ℓh)
    stats   = (
        golden_section_search_iterations = n_gss_iters,
        ula_stepsize                     = h
    )
    return sampler, stats
end
