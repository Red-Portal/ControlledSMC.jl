
function rand_initial_with_potential(
    rng::Random.AbstractRNG, ::AbstractSMC, path::AbstractPath, n_particles::Int
)
    x  = rand(rng, path.proposal, n_particles)
    ℓG = zeros(eltype(eltype(x)), n_particles)
    return x, ℓG
end

function log_potential_moments(ℓw::AbstractVector, ℓG::AbstractVector)
    ℓ∑w = logsumexp(ℓw)
    ℓG1 = logsumexp(ℓw + ℓG) - ℓ∑w
    ℓG2 = logsumexp(ℓw + 2 * ℓG) - ℓ∑w
    return (log_potential_moments=(ℓG1, ℓG2),)
end

function adapt_sampler(
    ::Random.AbstractRNG, sampler, ::Int, ::Any, ::Any, ::AbstractMatrix, ::AbstractVector
)
    return sampler
end

function sample(
    rng::Random.AbstractRNG,
    sampler::AbstractSMC,
    n_particles::Int,
    resample_threshold::Real;
    show_progress::Bool=true,
    save_particle_history::Bool=false,
)
    @assert 0 ≤ resample_threshold ≤ 1

    path    = sampler.path
    n_iters = length(path)
    states  = NamedTuple[]
    infos   = NamedTuple[]
    prog    = ProgressMeter.Progress(n_iters; showspeed=true, enabled=show_progress)

    x, ℓG = rand_initial_with_potential(rng, sampler, path, n_particles)
    ℓZ    = zero(eltype(x))
    ℓw    = ℓG

    ℓw_norm, ess         = normalize_weights(ℓw)
    ancestors, resampled = resample(rng, ℓw_norm, ess, resample_threshold)

    if resampled
        ℓZ = update_log_normalizer(ℓZ, ℓw)
        x  = x[:, ancestors]
        ℓw = zeros(eltype(x), n_particles)
        ℓG = ℓG[ancestors]
    end

    state = (particles=x, ancestors=ancestors, log_potential=ℓG)
    info  = (iteration=0, ess=n_particles, log_normalizer=ℓZ, resampled=resampled)

    if save_particle_history
        state = merge(state, (particles=x,))
    end

    push!(states, state)
    push!(infos,  info)
    pm_next!(prog, info)

    target_prev = get_target(path, 0)
    for t in 1:n_iters
        target        = get_target(path, t)
        sampler, info = adapt_sampler(rng, sampler, t, target, target_prev, x, ℓw)
        x, ℓG, aux    = mutate_with_potential(rng, sampler, t, target, target_prev, x)

        ℓG = @. ifelse(isfinite(ℓG), ℓG, -Inf)

        info′ = log_potential_moments(ℓw, ℓG)
        info = merge(info′, info)

        ℓw                   = ℓw + ℓG
        ℓw_norm, ess         = normalize_weights(ℓw)
        ancestors, resampled = resample(rng, ℓw_norm, ess, resample_threshold)

        if !isfinite(ess)
            throw(ErrorException("The ESS is NaN. Something is broken. Most likely all particles degenerated."))
        end

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
        info  = merge((iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled), info)

        if save_particle_history
            state = merge(state, (particles=x,))
        end

        push!(states, state)
        push!(infos, info)
        pm_next!(prog, info)
    end
    ℓw_norm, _ = normalize_weights(ℓw)
    return x, ℓw_norm, sampler, states, infos
end

function sample(
    sampler::AbstractSMC,
    n_particles::Int,
    resample_threshold::Real;
    kwargs...
)
    return sample(Random.default_rng(), sampler, n_particles, resample_threshold; kwargs...)
end
