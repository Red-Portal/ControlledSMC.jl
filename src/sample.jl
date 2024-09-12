
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
               ::AbstractSMC,
    path       ::AbstractPath,
    n_particles::Int,
)
    x  = rand(rng, path.proposal, n_particles)
    ℓG = zeros(eltype(eltype(x)), n_particles)
    x, ℓG
end

function sample(
    rng          ::Random.AbstractRNG,
    sampler      ::AbstractSMC,
    path         ::AbstractPath,
    n_particles  ::Int,
    threshold    ::Real;
    show_progress::Bool = true,
)
    n_iters = length(path)
    states  = Array{NamedTuple}(undef, n_iters)
    info    = Array{NamedTuple}(undef, n_iters)
    prog    = ProgressMeter.Progress(
        n_iters; barlen=31, showspeed=true, enabled=show_progress
    )

    x, ℓG = rand_initial_with_potential(rng, sampler, path, n_particles)
    ℓZ    = zero(eltype(x))
    ℓw    = fill(-log(convert(eltype(x), n_particles)), n_particles)

    ℓw, ℓZ, ess = reweigh(ℓw, 1, ℓG, ℓZ)
    x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

    states[1] = (particles=x, ancestors=ancestors, log_potential=ℓG[ancestors])
    info[1]   = (iteration=1, ess=n_particles, log_normalizer=ℓZ)

    for t in 2:n_iters
        x, ℓG, aux                  = mutate_with_potential(rng, sampler, path, t, x)
        ℓw, ℓZ, ess                 = reweigh(ℓw, t, ℓG, ℓZ)
        x, ℓw, ancestors, resampled = resample(rng, x, ℓw, ess, threshold)

        states[t] = merge((particles=x, log_potential=ℓG[ancestors]), aux)
        info[t]   = (iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled)

        pm_next!(prog, info[t])
    end
    x, states, info
end
