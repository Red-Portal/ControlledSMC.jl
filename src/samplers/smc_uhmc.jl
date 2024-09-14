
struct SMCUHMC{Stepsize<:Real,Mass<:AbstractMatrix} <: AbstractSMC
    leapfrog_stepsize  :: Stepsize
    diffusion_stepsize :: Stepsize
    mass_matrix        :: Mass
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::SMCUHMC, path::AbstractPath, n_particles::Int
)
    (; mass_matrix,) = sampler
    (; proposal,) = path

    x      = rand(rng, proposal, n_particles)
    n_dims = size(x, 1)
    v      = rand(rng, MvNormal(Zeros(n_dims), mass_matrix), n_particles)
    ℓG     = zeros(n_particles)
    return vcat(x, v), ℓG
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCUHMC, t::Int, πt, πtm1, ztm1::AbstractMatrix
)
    (; leapfrog_stepsize, diffusion_stepsize, mass_matrix) = sampler

    h, δ, M = diffusion_stepsize, leapfrog_stepsize, mass_matrix

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), M)

    vthalf = h * vtm1 + sqrt(1 - h^2) * unwhiten(M, randn(rng, size(vtm1)))
    xt, vt = leapfrog_batch(πt, xtm1, vthalf, δ, M)

    ℓπt     = logdensity_batch(πt, xt)
    ℓπtm1   = logdensity_batch(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = MvNormal.(h * eachcol(vthalf), Ref((1 - h^2) * M))
    K  = MvNormal.(h * eachcol(vtm1), Ref((1 - h^2) * M))
    ℓl = logpdf.(L, eachcol(vtm1))
    ℓk = logpdf.(K, eachcol(vthalf))
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end
