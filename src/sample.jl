
function log_potential_moments(ℓw::AbstractVector, ℓG::AbstractVector)
    ℓ∑w = logsumexp(ℓw)
    ℓG1 = logsumexp(ℓw + ℓG) - ℓ∑w
    ℓG2 = logsumexp(ℓw + 2 * ℓG) - ℓ∑w
    return (log_potential_moments=(ℓG1, ℓG2),)
end

function adapt_sampler(
    ::Random.AbstractRNG, sampler, ::Int, ::AbstractMatrix, ::AbstractVector
)
    return sampler, NamedTuple()
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

    n_iters = length(sampler)
    states  = NamedTuple[]
    infos   = NamedTuple[]
    prog    = ProgressMeter.Progress(n_iters + 1; showspeed=true, enabled=show_progress)

    x, ℓG = rand_initial_with_potential(rng, sampler, n_particles)
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
    push!(infos, info)
    pm_next!(prog, info)

    for t in 1:n_iters
        sampler, info = adapt_sampler(rng, sampler, t, x, ℓw)
        x, ℓG, aux    = mutate_with_potential(rng, sampler, t, x)

        ℓG = @. ifelse(isfinite(ℓG), ℓG, -Inf)

        info′ = log_potential_moments(ℓw, ℓG)
        info = merge(info′, info)

        ℓw                   = ℓw + ℓG
        ℓw_norm, ess         = normalize_weights(ℓw)
        ancestors, resampled = resample(rng, ℓw_norm, ess, resample_threshold)

        if !isfinite(ess)
            throw(
                ErrorException(
                    "The ESS is NaN. Something is broken. Most likely all particles degenerated.",
                ),
            )
        end

        if resampled || t == n_iters
            ℓZ = update_log_normalizer(ℓZ, ℓw)
        end

        if resampled
            x  = x[:, ancestors]
            ℓw = zeros(eltype(x), n_particles)
            ℓG = ℓG[ancestors]
        end

        state = merge((particles=x, log_potential=ℓG), aux)
        info = merge((iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled), info)

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

function sample(sampler::AbstractSMC, n_particles::Int, resample_threshold::Real; kwargs...)
    return sample(Random.default_rng(), sampler, n_particles, resample_threshold; kwargs...)
end
