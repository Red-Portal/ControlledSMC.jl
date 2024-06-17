
function systematic_sampling(rng, weights::AbstractVector, n_resample=length(weights))
    N  = length(weights)
    Δs = 1/n_resample
    u  = rand(rng, Uniform(0.0, Δs))
    s  = 1

    resample_idx = zeros(Int64, n_resample)
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

function resample(rng, x, w, logw, ess)
    n_particles = size(x, 2)
    if true #ess < n_particles/2
        idx       = systematic_sampling(rng, w)       
        resampled = true
        x[:,:]    = x[:,idx]
        w[:]     .= 1/n_particles
        logw[:]  .= -log(n_particles)
        x, w, logw, idx, resampled
    else
        resampled = false
        ancestor  = collect(1:n_particles)
        x, w, logw, ancestor, resampled
    end
end

function reweight(logw, G, logZ)
    N    = length(logw)
    logw = logw + G
    logZ = logZ + logsumexp(G) - log(N)
    logw = logw .- logsumexp(logw)
    w    = exp.(logw)
    ess  = 1 / sum(w.^2)
    w, logw, logZ, ess
end

function euler_fwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x + h/2*Γ*∇logπt
end

function euler_bwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x - h/2*Γ*∇logπt
end
