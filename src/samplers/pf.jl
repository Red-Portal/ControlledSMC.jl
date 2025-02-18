
abstract type AbstractParticleFilter <: AbstractSMC end

function rand_transition(::Random.AbstractRNG, ::AbstractParticleFilter, ::Int, ::AbstractMatrix) end

function logpdf_transition(::AbstractParticleFilter, ::Int, ::AbstractMatrix, ::AbstractMatrix) end

function potential(::AbstractParticleFilter, ::Int, ::AbstractMatrix) end

function mutate_with_potential(
    rng::Random.AbstractRNG,
    sampler::AbstractParticleFilter,
    t::Int,
    x::AbstractMatrix
)
    x′ = rand_transition(rng, sampler, t, x)
    ℓG = potential(sampler, t, x′)
    x′, ℓG, NamedTuple()
end

