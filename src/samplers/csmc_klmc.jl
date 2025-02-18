
struct CSMCKLMC{SMCBase<:SMCKLMC,Path<:AbstractPath,Policy<:AbstractVector} <:
       AbstractControlledSMC
    smc    :: SMCBase
    path   :: Path
    policy :: Policy
end

function CSMCKLMC(smc::SMCKLMC, path::AbstractPath)
    T      = length(path)
    d      = length(path.proposal)
    F      = eltype(path.proposal)
    policy = [(a=zeros(F, 2 * d), b=zeros(F, 2 * d), c=zero(F)) for _ in 1:T]
    return CSMCKLMC{typeof(smc),typeof(path),typeof(policy)}(smc, path, policy)
end

function twist_double_mvnormal_logmarginal(
    sampler::CSMCKLMC, t::Int, ψ_first, ψ_second, state
)
    (; smc,)       = sampler
    (; sigma_klmc) = smc
    (; a, b)       = ψ_first
    (; μ_klmc)     = state
    Σ              = sigma_klmc
    d, n           = size(μ_klmc.x1, 1), size(μ_klmc.x1, 2)
    A              = BlockDiagonal2by2(Diagonal(a[1:d]), Diagonal(a[(d + 1):(2 * d)]))
    b              = BatchVectors2(repeat(b[1:d], 1, n), repeat(b[(d + 1):(2 * d)], 1, n))
    K              = control_cov(A, sigma_klmc)
    μ_twist        = K.Σ * (Σ \ μ_klmc - b)
    Σ_twist        = K
    return twist_mvnormal_logmarginal(ψ_second, μ_twist, Σ_twist)
end

function twist_kernel_logmarginal(csmc::CSMCKLMC, twist, πt, t::Int, ztm1::AbstractMatrix)
    (; stepsizes, dampings, sigma_klmc) = csmc.smc
    h, γ = stepsizes[t], dampings[t]
    d = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    μ_klmc = klmc_mean(πt, xtm1, vtm1, h, γ)
    return twist_mvnormal_logmarginal(twist, μ_klmc, sigma_klmc)
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::CSMCKLMC, path::AbstractPath, n_particles::Int
)
    (; proposal,) = path
    (; policy,) = sampler
    ψ0 = first(policy)
    n_dims = length(proposal)
    μx, Σx = mean(proposal), Distributions._cov(proposal)
    Σx_diag = Diagonal(Σx)
    μv = Zeros(n_dims)
    Σv = Diagonal(one(Σx_diag))
    Σxv = Diagonal(zero(Σx_diag))
    μz = BatchVectors2(repeat(μx, 1, n_particles), repeat(μv, 1, n_particles))
    Σz = BlockHermitian2by2(Σx_diag, Σxv, Σv)
    Σz_pd = PDMats.PDMat(Σz)

    z    = twist_mvnormal_rand(rng, ψ0, μz, Σz_pd)
    ℓG0  = zero(eltype(z))
    ℓqψ0 = twist_mvnormal_logmarginal(ψ0, μz, Σz_pd)
    ℓψ0  = twist_logdensity(ψ0, z)
    πtp1 = get_target(path, 2)
    ℓMψ  = twist_kernel_logmarginal(sampler, policy[2], πtp1, 2, z)
    ℓGψ  = @. ℓG0 + ℓqψ0 + ℓMψ - ℓψ0
    return z, ℓGψ
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::CSMCKLMC, t::Int, πt, πtm1, ztm1::AbstractMatrix
)
    (; smc, policy, path) = sampler
    (; stepsizes, dampings, sigma_klmc) = smc
    h, γ = stepsizes[t], dampings[t]
    ψ = policy[t]

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), I)

    μ_klmc = klmc_mean(πt, xtm1, vtm1, h, γ)
    K      = BatchMvNormal(μ_klmc, sigma_klmc)
    zt     = twist_mvnormal_rand(rng, ψ, μ_klmc, sigma_klmc)
    xt, vt = zt[1:d, :], zt[(d + 1):end, :]

    ℓπt     = LogDensityProblems.logdensity(πt, xt)
    ℓπtm1   = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    μ_klmc_back = klmc_mean(πtm1, xt, -vt, h, γ)
    L           = BatchMvNormal(μ_klmc_back, sigma_klmc)

    ℓk = logpdf(K, BatchVectors2(xt, vt))
    ℓl = logpdf(L, BatchVectors2(xtm1, vtm1))
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk

    ℓψ  = twist_logdensity(ψ, zt)
    T   = length(path)
    ℓGψ = if t < T
        ψtp1 = policy[t + 1]
        πtp1 = get_target(path, t + 1)
        ℓMψ  = twist_kernel_logmarginal(sampler, ψtp1, πtp1, t + 1, zt)
        @. ℓG + ℓMψ - ℓψ
    elseif t == T
        @. ℓG - ℓψ
    end
    return zt, ℓGψ, (μ_klmc=μ_klmc,)
end
