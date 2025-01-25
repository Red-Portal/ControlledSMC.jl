
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
    ::AbstractRNG, sampler, ::Int, ::Any, ::Any, ::AbstractMatrix, ::AbstractVector
)
    return sampler
end

function incremental_elbo(ℓw, ℓG)
    ∑ℓw     = logsumexp(ℓw)
    ℓw_norm = ℓw .- ∑ℓw
    dot(exp.(ℓw_norm), ℓG)
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

    ∑elbo = 0.0
    
    n_iters = length(path)
    states  = NamedTuple[]
    stats   = NamedTuple[]
    prog    = ProgressMeter.Progress(n_iters; showspeed=true, enabled=show_progress)

    x, ℓG = rand_initial_with_potential(rng, sampler, path, n_particles)
    ℓZ    = zero(eltype(x))
    ℓw    = ℓG

    ℓw_norm, ess         = normalize_weights(ℓw)
    ancestors, resampled = resample(rng, ℓw_norm, ess, threshold)

    ignore_derivatives() do
        if resampled
            ℓZ = update_log_normalizer(ℓZ, ℓw)
            x  = x[:, ancestors]
            ℓw = zeros(eltype(x), n_particles)
            ℓG = ℓG[ancestors]
        end
    end

    state = (particles=x, ancestors=ancestors, log_potential=ℓG)
    stat  = (iteration=1, ess=n_particles, log_normalizer=ℓZ, resampled=resampled)

    ignore_derivatives() do
        push!(states, state)
        push!(stats, stat)
        pm_next!(prog, last(stats))
    end

    target_prev = get_target(path, 1)
    for t in 2:n_iters
        target        = get_target(path, t)
        sampler, stat = adapt_sampler(rng, sampler, t, target, target_prev, x, ℓw)
        x, ℓG, aux    = mutate_with_potential(rng, sampler, t, target, target_prev, x)

        ℓG = @. ifelse(isfinite(ℓG), ℓG, -Inf)

        stat′ = log_potential_moments(ℓw, ℓG)
        stat = merge(stat′, stat)

        ℓw           = ℓw + ℓG
        ∑elbo        = incremental_elbo(ℓw, ℓG)
        ℓw_norm, ess = normalize_weights(ℓw)

        ess_nodiff = ignore_derivatives(ess) 
        ancestors, resampled = resample(rng, ℓw_norm, ess_nodiff, threshold)

        if !isfinite(ess)
            throw(ErrorException("The ESS is NaN. Something is broken. Most likely all particles degenerated."))
        end

        if resampled || t == n_iters
            ℓZ = update_log_normalizer(ℓZ, ℓw)
        end

        if resampled
            ℓw_norm_rsp     = ℓw_norm[ancestors]
            score_grad_term = ℓw_norm_rsp - ignore_derivatives(ℓw_norm_rsp)

            x  = x[:, ancestors]
            ℓw = score_grad_term
            ℓG = ℓG[ancestors]
        end

        target_prev = target
        ignore_derivatives() do
            state = merge((particles=x, log_potential=ℓG), aux)
            stat = merge((iteration=t, ess=ess, log_normalizer=ℓZ, resampled=resampled), stat)
            push!(states, state)
            push!(stats, stat)
            pm_next!(prog, last(stats))
        end
    end
    ℓw_norm, _ = normalize_weights(ℓw)
    return x, ℓw_norm, sampler, states, stats, ∑elbo
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
