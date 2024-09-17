
struct SMCKLMC{Stepsize<:Real,Chol<:BlockCholesky2by2} <: AbstractSMC
    stepsize     :: Stepsize
    damping      :: Stepsize
    chol_klmc    :: Chol
    cholinv_klmc :: Chol
end

function SMCKLMC(stepsize::Real, damping::Real)
    Σ    = klmc_cov(stepsize, damping)
    L    = cholesky2by2(Σ)
    Linv = inv2by2(L)
    return SMCKLMC(stepsize, damping, L, Linv)
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
    (; stepsize, damping, chol_klmc, cholinv_klmc) = sampler
    h, γ, L_klmc, Linv_klmc = stepsize, damping, chol_klmc, cholinv_klmc

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), I)

    μx, μv = klmc_mean(πt, xtm1, vtm1, h, γ)
    K      = BivariateMvNormal(μx, μv, L_klmc, Linv_klmc)
    xt, vt = bivariate_rand(rng, K)

    ℓπt     = logdensity(πt, xt)
    ℓπtm1   = logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    μx_back, μv_back = klmc_mean(πtm1, xt, -vt, h, γ)
    L                = BivariateMvNormal(μx_back, μv_back, L_klmc, Linv_klmc)

    ℓk = bivariate_logpdf(K, xt, vt)
    ℓl = bivariate_logpdf(L, xtm1, vtm1)
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end
