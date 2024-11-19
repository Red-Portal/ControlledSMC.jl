
struct SMCULA{
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Precond<:AbstractMatrix,
} <: AbstractSMC
    stepsizes :: Stepsizes
    backward  :: Backward
    precond   :: Precond
end

function SMCULA(stepsize::Real, n_steps::Int, backward::AbstractBackwardKernel, precond::AbstractMatrix)
    stepsizes = Fill(stepsize, n_steps)
    return SMCULA{typeof(stepsizes), typeof(backward), typeof(precond)}(
        stepsizes, backward, precond
    )
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes, precond) = sampler
    ht, Γ  = stepsizes[t], precond
    q      = gradient_flow_euler(πt, xtm1, ht, Γ)
    xt     = q + sqrt(2 * ht) * unwhiten(Γ, randn(rng, eltype(q), size(q)))
    ℓG     = potential(sampler, t, πt, πtm1, xt, xtm1)
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
    ℓπtm1_xtm1  = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt      = LogDensityProblems.logdensity(πt, xt)
    q_fwd       = gradient_flow_euler(πt, xtm1, ht, Γ)
    q_bwd       = gradient_flow_euler(πtm1, xt, htm1, Γ)
    K           = MvNormal.(eachcol(q_fwd), Ref(2 * ht * Γ))
    L           = MvNormal.(eachcol(q_bwd), Ref(2 * htm1 * Γ))
    ℓk          = logpdf.(K, eachcol(xt))
    ℓl          = logpdf.(L, eachcol(xtm1))
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
    ht, Γ      = stepsizes[t], precond
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt     = LogDensityProblems.logdensity(πt, xt)
    q_fwd      = gradient_flow_euler(πt, xtm1, ht, Γ)
    q_bwd      = gradient_flow_euler(πt, xt, ht, Γ)
    K          = MvNormal.(eachcol(q_fwd), Ref(2 * ht * Γ))
    L          = MvNormal.(eachcol(q_bwd), Ref(2 * ht * Γ))
    ℓK         = logpdf.(K, eachcol(xt))
    ℓL         = logpdf.(L, eachcol(xtm1))
    return ℓπt_xt + ℓL - ℓπtm1_xtm1 - ℓK
end
