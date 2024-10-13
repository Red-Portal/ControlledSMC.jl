
struct SMCUHMC{Stepsizes<:AbstractVector,Dampings<:AbstractVector,Mass<:AbstractMatrix} <: AbstractSMC
    stepsizes   :: Stepsizes
    dampings    :: Stepsizes
    mass_matrix :: Mass
end

function SMCUHMC(stepsize::Real, damping::Real, n_steps::Int, mass_matrix::AbstractMatrix)
    stepsizes = Fill(stepsize, n_steps)
    dampings  = Fill(damping, n_steps)
    SMCUHMC{typeof(stepsizes), typeof(dampings), typeof(mass_matrix)}(
        stepsizes, dampings,  mass_matrix
    )
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
    (; stepsizes, dampings, mass_matrix) = sampler
    ϵ, α, M = stepsizes[t], dampings[t], mass_matrix
    sqrt1mα = sqrt(1 - α)

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), M)

    vthalf = sqrt1mα * vtm1 + sqrt(α) * unwhiten(M, randn(rng, size(vtm1)))
    xt, vt = leapfrog(πt, xtm1, vthalf, ϵ, M)

    ℓπt     = LogDensityProblems.logdensity(πt, xt)
    ℓπtm1   = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = MvNormal.(sqrt1mα * eachcol(vthalf), Ref(α * M))
    K  = MvNormal.(sqrt1mα * eachcol(vtm1), Ref(α * M))
    ℓl = logpdf.(L, eachcol(vtm1))
    ℓk = logpdf.(K, eachcol(vthalf))
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end
