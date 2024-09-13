
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, ::AbstractSMC, path::AbstractPath, n_particles::Int
)
    x  = rand(rng, path.proposal, n_particles)
    ℓG = zeros(eltype(eltype(x)), n_particles)
    return x, ℓG
end

function sample(
    rng::Random.AbstractRNG,
    sampler::AbstractSMC,
    path::AbstractPath,
    n_particles::Int,
    threshold::Real;
    show_progress::Bool=true,
)
    @assert 0 ≤ threshold ≤ 1

    n_iters = length(path)
    states  = NamedTuple[]
    stats   = NamedTuple[]
    prog    = ProgressMeter.Progress(n_iters; barlen=31, showspeed=true, enabled=show_progress)

    x, ℓG = rand_initial_with_potential(rng, sampler, path, n_particles)
    ℓZ    = zero(eltype(x))
    ℓw    = ℓG

    ℓw_norm, ess         = normalize_weights(ℓw)
    ancestors, resampled = resample(rng, ℓw_norm, ess, threshold)

    if resampled
        ℓZ = update_log_normalizer(ℓZ, ℓw)

        x  = x[:, ancestors]
        ℓw = zeros(eltype(x), n_particles)
        ℓG = ℓG[ancestors]
    end

    state = (particles=x, ancestors=ancestors, log_potential=ℓG)
    stat  = (iteration=1, ess=n_particles, log_normalizer=ℓZ)
    push!(states, state)
    push!(stats, stat)

    target_prev = step(path, 1, x, ℓw)
    for t in 2:n_iters
        target               = step(path, t, x, ℓw)
        x, ℓG, aux           = mutate_with_potential(rng, sampler, t, target, target_prev, x)
        ℓw                   = ℓw + ℓG
        ℓw_norm, ess         = normalize_weights(ℓw)
        ancestors, resampled = resample(rng, ℓw_norm, ess, threshold)

        if resampled || t == n_iters
            ℓZ = update_log_normalizer(ℓZ, ℓw)
        end

        if resampled
            x  = x[:, ancestors]
            ℓw = zeros(eltype(x), n_particles)
            ℓG = ℓG[ancestors]
        end

        target_prev = target

        state = merge((particles=x, log_potential=ℓG), aux)
        stat  = (iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled)
        push!(states, state)
        push!(stats, stat)

        pm_next!(prog, last(stats))
    end
    ℓw_norm, _ = normalize_weights(ℓw)
    return x, ℓw_norm, states, stats
end

function sample(
    sampler::AbstractSMC,
    path::AbstractPath,
    n_particles::Int,
    threshold::Real;
    show_progress::Bool=true,
)
    return sample(
        Random.default_rng(), sampler, path, n_particles, threshold; show_progress
    )
end
