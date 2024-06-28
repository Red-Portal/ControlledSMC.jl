
function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::AbstractSMC,
               ::Any,
    n_particles::Int,
)
    rand(rng, sampler.proposal, n_particles), zeros(n_particles)
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

    x, ℓG = rand_initial_with_potential(rng, sampler, logtarget, n_particles)
    ℓZ    = 0.0
    ℓw    = fill(-log(n_particles), n_particles)

    ℓw, ℓZ, ess = reweight(ℓw, ℓG, ℓZ)
    x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

    states[1] = (particles=x, ancestors=ancestors, logG=ℓG[ancestors])
    info[1]   = (iteration=1, ess=n_particles, logZ=ℓZ)

    for t in 2:T
        x, ℓG, aux                  = mutate_with_potential(rng, sampler, t, x, logtarget)
        ℓw, ℓZ, ess                 = reweight(ℓw, ℓG, ℓZ)
        x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

        states[t] = merge((particles=x, logG=ℓG[ancestors]), aux)
        info[t]   = (iteration=t, ess=ess, logZ=ℓZ, resampled=resampled)
    end
    x, states, info
end
