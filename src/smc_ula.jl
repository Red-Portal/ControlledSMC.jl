
struct SMCULA{
    G    <: AbstractMatrix,
    H    <: Real,
    B    <: AbstractBackwardKernel,
} <: AbstractSMC
    Γ       ::G
    h0      ::H
    hT      ::H
    backward::B
end

function mutate_with_potential(
    rng    ::Random.AbstractRNG,
    sampler::SMCULA,
    path   ::AbstractPath,
    t      ::Int,
    x      ::AbstractMatrix,
)
    (; h0, hT, Γ) = sampler
    ht   = anneal(path, t, h0, hT)
    ∇ℓπt = logtarget_gradient_batch(path, t, x)
    q    = gradient_flow_euler_batch(x, ∇ℓπt, ht, Γ)
    x′   = q + sqrt(ht)*unwhiten(Γ, randn(rng, eltype(q), size(q)))
    ℓG   = potential(sampler, path, t, x′, x)
    x′, ℓG, NamedTuple()
end

function potential(
    sampler::SMCULA,
    path   ::AbstractPath,
    t      ::Int,
    x_curr ::AbstractMatrix,
    x_prev ::AbstractMatrix,
)
    potential_with_backward(sampler, path, sampler.backward, t, x_curr, x_prev)
end

function potential_with_backward(
          ::SMCULA,
    path  ::AbstractPath,
          ::DetailedBalance,
    t     ::Int,
    x_curr::AbstractMatrix,
    x_prev::AbstractMatrix,
)
    ℓπt_xtm1   = logtarget.(Ref(path), Ref(t),     eachcol(x_prev))
    ℓπtm1_xtm1 = logtarget.(Ref(path), Ref(t - 1), eachcol(x_prev))
    ℓπt_xtm1 - ℓπtm1_xtm1
end

function potential_with_backward(
    sampler::SMCULA,
    path   ::AbstractPath,
           ::TimeCorrectForwardKernel,
    t      ::Int,
    x_curr ::AbstractMatrix,
    x_prev ::AbstractMatrix,
)
    (; h0, hT, Γ) = sampler

    ht   = anneal(path, t, h0, hT)
    htm1 = anneal(path, t-1, h0, hT)

    ∇ℓπtm1_xt = logtarget_gradient_batch(path, t - 1, x_curr)
    ∇ℓπt_xtm1 = logtarget_gradient_batch(path, t,     x_prev)
    ℓπtm1_xtm1 = logtarget_batch(path, t - 1, x_prev)
    ℓπt_xt     = logtarget_batch(path, t,     x_curr)

    q_fwd = gradient_flow_euler_batch(x_prev, ∇ℓπt_xtm1, ht, Γ)
    q_bwd = gradient_flow_euler_batch(x_curr, ∇ℓπtm1_xt, ht, Γ)
    K     = MvNormal.(eachcol(q_fwd), Ref(ht*Γ))
    L     = MvNormal.(eachcol(q_bwd), Ref(htm1*Γ))
    ℓk    = logpdf.(K, eachcol(x_curr))
    ℓl    = logpdf.(L, eachcol(x_prev))
    ℓπt_xt + ℓl - ℓπtm1_xtm1 - ℓk
end

function potential_with_backward(
    sampler::SMCULA,
    path   ::AbstractPath,
           ::ForwardKernel,
    t      ::Int,
    x_curr ::AbstractMatrix,
    x_prev ::AbstractMatrix,
)
    (; h0, hT, Γ) = sampler

    ht = anneal(path, t, h0, hT)

    ∇ℓπt_xt   = logtarget_gradient_batch(path, t, x_curr)
    ∇ℓπt_xtm1 = logtarget_gradient_batch(path, t, x_prev)
    ℓπtm1_xtm1 = logtarget_batch(path, t - 1, x_prev)
    ℓπt_xt     = logtarget_batch(path, t,     x_curr)

    q_fwd = gradient_flow_euler_batch(x_prev, ∇ℓπt_xtm1, ht, Γ)
    q_bwd = gradient_flow_euler_batch(x_curr, ∇ℓπt_xt,   ht, Γ)
    K     = MvNormal.(eachcol(q_fwd), Ref(ht*Γ))
    L     = MvNormal.(eachcol(q_bwd), Ref(ht*Γ))
    ℓK    = logpdf.(K, eachcol(x_curr))
    ℓL    = logpdf.(L, eachcol(x_prev))
    ℓπt_xt + ℓL - ℓπtm1_xtm1 - ℓK
end
