
function rand_initial_particles(
    rng        ::Random.AbstractRNG,
    sampler    ::AbstractSMC,
    n_particles::Int,
)
    rand(rng, sampler.proposal, n_particles)
end

function potential_init(::AbstractSMC, x::AbstractMatrix, logtarget)
    n_particles = size(x, 2)
    zeros(n_particles)
end

function sample(
    rng        ::Random.AbstractRNG,
    sampler    ::AbstractSMC,
    n_particles::Int,
    threshold  ::Real,
    logtarget,
)
    T      = length(sampler)
    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    x  = rand_initial_particles(rng, sampler, n_particles)
    ℓG = potential_init(sampler, x, logtarget)
    ℓZ = 0.0
    ℓw = fill(-log(n_particles), n_particles)

    ℓw, ℓZ, ess = reweight(ℓw, ℓG, ℓZ)
    x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

    states[1] = (particles=x, ancestors=ancestors, logG=ℓG[ancestors])
    info[1]   = (iteration=1, ess=n_particles, logZ=ℓZ)

    for t in 2:T
        x_next = mutate(rng, sampler, t, x, logtarget)
        ℓG     = potential(sampler, t, x_next, x, logtarget)
        x      = x_next

        ℓw, ℓZ, ess                 = reweight(ℓw, ℓG, ℓZ)
        x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

        states[t] = (particles=x, ancestors=ancestors, logG=ℓG)
        info[t]   = (iteration=t, ess=ess, logZ=ℓZ, resampled=resampled)
    end
    x, states, info
end
