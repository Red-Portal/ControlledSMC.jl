
function systematic_sampling(
    rng       ::Random.AbstractRNG,
    weights   ::AbstractVector,
    n_resample::Int = length(weights)
)
    N  = length(weights)
    Δs = 1/n_resample
    u  = rand(rng, Uniform(0, Δs))
    s  = 1

    resample_idx = zeros(Int, n_resample)
    stratas      = cumsum(weights)
    @inbounds for i = 1:n_resample
        while(u > stratas[s] && s < N)
            s += 1
        end
        resample_idx[i] = s
        u += Δs
    end
    resample_idx
end

function resample(
    rng      ::Random.AbstractRNG,
    x        ::AbstractMatrix,
    ℓw       ::AbstractVector,
    ess      ::Real,
    threshold::Real,
)
    n_particles = size(x, 1)
    # Marginal likelihood update rule currently seems biased for
    # adaptive resampling.
    if true #ess < n_particles*threshold
        idx       = systematic_sampling(rng, exp.(ℓw))
        resampled = true
        x         = x[:,idx]
        ℓw[:]    .= -log(n_particles)
        x, ℓw, idx, resampled
    else
        resampled = false
        ancestor  = collect(1:n_particles)
        x, ℓw, ancestor, resampled
    end
end

function reweigh(
    ℓw::AbstractVector,
    t,
    ℓG::AbstractVector,
    ℓZ::Real
)
    N       = length(ℓw)
    ℓw      = ℓw + ℓG
    ℓZ′     = ℓZ + logsumexp(ℓG) - log(N)
    ℓw_norm = ℓw .- logsumexp(ℓw)
    ess     = exp(-logsumexp(2*ℓw_norm))
    ℓw_norm, ℓZ′, ess
end
