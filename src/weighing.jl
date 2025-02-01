
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

function ssp_sampling(
    rng::Random.AbstractRNG, weights::AbstractVector, n_resample::Int=length(weights)
)
#=
    SSP resampling.

    SSP stands for Srinivasan Sampling Process. This resampling scheme is
    discussed in Gerber et al (2019). Basically, it has similar properties as
    systematic resampling (number of off-springs is either k or k + 1, with
    k <= N W^n < k +1), and in addition is consistent. See that paper for more
    details.

    Reference
    =========
    Gerber M., Chopin N. and Whiteley N. (2019). Negative association, ordering
    and convergence of resampling methods. Ann. Statist. 47 (2019), no. 4, 2236–2260.
=##
    n       = length(weights)
    m       = n_resample
    mw      = m * weights
    n_child = floor.(Int, mw)
    xi      = mw - n_child
    u       = rand(rng, n - 1)
    i, j    = 1, 2
    for k in 1:n-1
        δi = min(xi[j], 1 - xi[i])
        δj = min(xi[i], 1 - xi[j])
        ∑δ = δi + δj

        pj = (∑δ > 0) ? δi / ∑δ : 0
        if u[k] < pj
            j, i = i, j
            δi   = δj
        end
        if xi[j] < 1 - xi[i]
            xi[i] += δi
            j      = k + 2
        else
            xi[j]      -= δi
            n_child[i] += 1
            i           = k + 2
        end
    end

    # due to round-off error accumulation, we may be missing one particle
    if sum(n_child) == m - 1
        last_ij = if j == n + 1
            i
        else
            j
        end
        if xi[last_ij] > 0.99
            n_child[last_ij] += 1
        end
    end
    if sum(n_child) != m
        throw(RuntimeError("ssp resampling: wrong size for output"))
    end
    return inverse_rle(1:n, n_child)
end

function resample(
    rng::Random.AbstractRNG, ℓw_norm::AbstractVector, ess::Real, threshold::Real
)
    N = length(ℓw_norm)
    if ess < N * threshold
        ancestors = ssp_sampling(rng, exp.(ℓw_norm))
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
