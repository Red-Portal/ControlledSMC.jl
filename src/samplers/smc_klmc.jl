
struct SMCKLMC{Stepsize<:Real,Damping<:Real,InvMass<:Real,Sigma<:KLMCKernelCov} <: AbstractSMC
    stepsize   :: Stepsize
    damping    :: Damping
    invmass    :: InvMass
    sigma_klmc :: Sigma
end

function SMCKLMC(stepsize::Real, damping::Real, invmass::Real)
    Σ_klmc = klmc_cov(stepsize, damping, invmass)
    return SMCKLMC(stepsize, damping, invmass, Σ_klmc)
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, ::SMCKLMC, path::AbstractPath, n_particles::Int
)
    (; proposal,) = path
    x = rand(rng, proposal, n_particles)
    n_dims = size(x, 1)
    v = rand(rng, MvNormal(Zeros(n_dims), I), n_particles)
    ℓG = zeros(n_particles)
    return vcat(x, v), ℓG
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCKLMC, ::Int, πt, πtm1, ztm1::AbstractMatrix
)
    (; stepsize, damping, invmass, sigma_klmc) = sampler
    h, γ, u = stepsize, damping, invmass

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), I)

    K      = klmc_transition_kernel(πt, xtm1, vtm1, h, γ, u, sigma_klmc)
    xt, vt = klmc_rand(rng, K)

    ℓπt     = logdensity(πt, xt)
    ℓπtm1   = logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = klmc_transition_kernel(πtm1, xt, -vt, h, γ, u, sigma_klmc)
    ℓk = klmc_logpdf(K, xt, vt)
    ℓl = klmc_logpdf(L, xtm1, vtm1)
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end
