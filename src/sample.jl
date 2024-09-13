
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
    n_particles  ::Int;
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

    ℓw, ℓZ, ess = reweigh(ℓw, ℓG, ℓZ)
    x, ℓw, ancestors, resampled = resample(rng, x, ℓw)

    states[1] = (particles=x, ancestors=ancestors, log_potential=ℓG[ancestors])
    info[1]   = (iteration=1, ess=n_particles, log_normalizer=ℓZ)

    target_prev = step(path, 1, x, ℓw)
    for t in 2:n_iters
        target                      = step(path, t, x, ℓw)
        x, ℓG, aux                  = mutate_with_potential(rng, sampler, t, target, target_prev, x)
        ℓw, ℓZ, ess                 = reweigh(ℓw, ℓG, ℓZ)
        x, ℓw, ancestors, resampled = resample(rng, x, ℓw)

        states[t] = merge((particles=x, log_potential=ℓG[ancestors]), aux)
        info[t]   = (iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled)

        pm_next!(prog, info[t])
        target_prev = target
    end
    x, states, info
end

function sample(
    sampler      ::AbstractSMC,
    path         ::AbstractPath,
    n_particles  ::Int;
    show_progress::Bool = true,
)
    sample(Random.default_rng(), sampler, path, n_particles; show_progress)
end
