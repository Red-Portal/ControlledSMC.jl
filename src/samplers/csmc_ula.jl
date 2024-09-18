
struct CSMCULA{SMCBase<:SMCULA,Path<:AbstractPath,Policy<:AbstractVector} <:
       AbstractControlledSMC
    smc    :: SMCBase
    path   :: Path
    policy :: Policy
end

function CSMCULA(smc::SMCULA, path::AbstractPath)
    T      = length(path)
    d      = length(path.proposal)
    F      = eltype(path.proposal)
    policy = [(a=zeros(F, d), b=zeros(F, d), c=zero(F)) for _ in 1:T]
    return CSMCULA{typeof(smc),typeof(path),typeof(policy)}(smc, path, policy)
end

function twist_double_mvnormal_logmarginal(
    sampler::CSMCULA, t::Int, ψ_first, ψ_second, state
)
    (; smc, path) = sampler
    (; stepsize_proposal, stepsize_problem, precond) = smc
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)

    q         = state.q
    (; a, b)  = ψ_first
    A         = Diagonal(a)
    K         = inv(4 * ht * A + inv(Γ))
    μ_twisted = K * (Γ \ q .- 2 * ht * b)
    Σ_twisted = 2 * ht * K
    return twist_mvnormal_logmarginal(ψ_second, μ_twisted, Σ_twisted)
end

function twist_kernel_logmarginal(csmc::CSMCULA, twist, πt, t::Int, xtm1::AbstractMatrix)
    (; stepsize_proposal, stepsize_problem, precond, path) = csmc.smc
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)

    q = gradient_flow_euler(πt, xtm1, ht, Γ)
    return twist_mvnormal_logmarginal(twist, q, 2 * ht * Γ)
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::CSMCULA, path::AbstractPath, n_particles::Int
)
    (; proposal,) = path
    (; policy,) = sampler
    ψ0 = first(policy)
    μ, Σ = mean(proposal), Distributions._cov(proposal)
    x = twist_mvnormal_rand(rng, ψ0, repeat(μ, 1, n_particles), Σ)

    ℓG0  = zero(eltype(x))
    ℓqψ0 = twist_mvnormal_logmarginal(ψ0, μ, Σ)
    ℓψ0  = twist_logdensity(ψ0, x)
    πtp1 = step(path, 2, x, x[1, :])
    ℓMψ  = twist_kernel_logmarginal(sampler, policy[2], πtp1, 2, x)
    ℓGψ  = @. ℓG0 + ℓqψ0 + ℓMψ - ℓψ0

    return x, ℓGψ
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::CSMCULA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; smc, policy, path) = sampler
    (; stepsize_proposal, stepsize_problem, precond, path) = smc
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond
    ht = anneal(GeometricAnnealing(path.schedule[t]), h0, hT)

    ψ = policy[t]
    q = gradient_flow_euler(πt, xtm1, ht, Γ)
    xt = twist_mvnormal_rand(rng, ψ, q, 2 * ht * Γ)

    ℓG  = potential(smc, t, πt, πtm1, xt, xtm1)
    ℓψ  = twist_logdensity(ψ, xt)
    T   = length(path)
    ℓGψ = if t < T
        ψtp1 = policy[t + 1]
        πtp1 = step(path, t + 1, xt, ℓG)
        ℓMψ  = twist_kernel_logmarginal(sampler, ψtp1, πtp1, t + 1, xt)
        @. ℓG + ℓMψ - ℓψ
    elseif t == T
        @. ℓG - ℓψ
    end
    return xt, ℓGψ, (q=q,)
end
