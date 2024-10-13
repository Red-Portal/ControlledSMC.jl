
struct SMCKLMC{Stepsize<:Real,Sigma<:BlockPDMat2by2} <: AbstractSMC
    stepsize   :: Stepsize
    damping    :: Stepsize
    sigma_klmc :: Sigma
end

function SMCKLMC(n_dims::Int, stepsize::Real, damping::Real)
    Σ    = klmc_cov(n_dims, stepsize, damping)
    Σ_pd = PDMats.PDMat(Σ)
    return SMCKLMC(stepsize, damping, Σ_pd)
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
    (; stepsize, damping, sigma_klmc) = sampler
    h, γ, Σ_klmc = stepsize, damping, sigma_klmc

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), I)

    μ_klmc = klmc_mean(πt, xtm1, vtm1, h, γ)
    K      = BatchMvNormal(μ_klmc, Σ_klmc)
    zt     = rand(rng, K)
    xt, vt = zt.x1, zt.x2

    ℓπt     = LogDensityProblems.logdensity(πt, xt)
    ℓπtm1   = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    μ_klmc_back = klmc_mean(πtm1, xt, -vt, h, γ)
    L           = BatchMvNormal(μ_klmc_back, Σ_klmc)

    ℓk = logpdf(K, BatchVectors2(xt, vt))
    ℓl = logpdf(L, BatchVectors2(xtm1, vtm1))
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end
