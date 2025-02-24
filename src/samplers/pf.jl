
abstract type AbstractParticleFilter <: AbstractSMC end

function transition(::AbstractParticleFilter, ::Int, ::AbstractVector) end

function potential(::AbstractParticleFilter, ::Int, ::AbstractMatrix) end

function kalman_update(::Any, ::Any) end

function mutate_with_potential(
    rng::Random.AbstractRNG,
    sampler::AbstractParticleFilter,
    t::Int,
    x::AbstractMatrix
)
    x′ = mapslices(x, dims=1) do xn
        Mn = transition(sampler, t, xn)
        rand(rng, Mn)
    end
    ℓG = potential(sampler, t, x′)
    x′, ℓG, NamedTuple()
end

