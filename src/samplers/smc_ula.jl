
struct SMCULA{
    Stepsize<:Real,
    Precond<:AbstractMatrix,
    Backward<:AbstractBackwardKernel,
    Path<:GeometricAnnealingPath,
} <: AbstractSMC
    stepsize_proposal :: Stepsize
    stepsize_problem  :: Stepsize
    backward          :: Backward
    precond           :: Precond
    path              :: Path
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCULA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsize_proposal, stepsize_problem, precond, path) = sampler
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)

    q  = gradient_flow_euler(πt, xtm1, ht, Γ)
    xt = q + sqrt(2 * ht) * unwhiten(Γ, randn(rng, eltype(q), size(q)))
    ℓG = potential(sampler, t, πt, πtm1, xt, xtm1)
    return xt, ℓG, NamedTuple()
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
    (; stepsize_proposal, stepsize_problem, precond, path) = sampler
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)
    htm1 = anneal(GeometricAnnealing(path.schedule[t - 1]), h0, hT)

    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xt     = LogDensityProblems.logdensity(πt, xt)
    q_fwd      = gradient_flow_euler(πt, xtm1, ht, Γ)
    q_bwd      = gradient_flow_euler(πtm1, xt, htm1, Γ)
    K          = MvNormal.(eachcol(q_fwd), Ref(2 * ht * Γ))
    L          = MvNormal.(eachcol(q_bwd), Ref(2 * htm1 * Γ))
    ℓk         = logpdf.(K, eachcol(xt))
    ℓl         = logpdf.(L, eachcol(xtm1))
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
    (; stepsize_proposal, stepsize_problem, precond, path) = sampler
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)

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
