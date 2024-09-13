
function systematic_sampling(
    rng::Random.AbstractRNG, weights::AbstractVector, n_resample::Int=length(weights)
)
    N  = length(weights)
    Δs = 1 / n_resample
    u  = rand(rng, Uniform(0, Δs))
    s  = 1

    resample_idx = zeros(Int, n_resample)
    stratas      = cumsum(weights)
    @inbounds for i in 1:n_resample
        while (u > stratas[s] && s < N)
            s += 1
        end
        resample_idx[i] = s
        u += Δs
    end
    return resample_idx
end

function resample(
    rng::Random.AbstractRNG, ℓw_norm::AbstractVector, ess::Real, threshold::Real
)
    N = length(ℓw_norm)
    if ess < N * threshold
        ancestors = systematic_sampling(rng, exp.(ℓw_norm))
        resampled = true
        ancestors, resampled
    else
        resampled = false
        ancestor  = collect(1:N)
        ancestor, resampled
    end
end

function update_log_normalizer(ℓZ::Real, ℓw::AbstractVector)
    N = length(ℓw)
    return ℓZ + logsumexp(ℓw) - log(N)
end

function normalize_weights(ℓw::AbstractVector)
    ℓw_norm = ℓw .- logsumexp(ℓw)
    return ℓw_norm, exp(-logsumexp(2 * ℓw_norm))
end

function reweigh(ℓw::AbstractVector, ℓG::AbstractVector, ℓZ::Real)
    N       = length(ℓw)
    ℓw      = ℓw + ℓG
    ℓZ′     = ℓZ + logsumexp(ℓG) - log(N)
    ℓw_norm = ℓw .- logsumexp(ℓw)
    ess     = exp(-logsumexp(2 * ℓw_norm))
    return ℓw_norm, ℓZ′, ess
end
