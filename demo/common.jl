
abstract type AbstractSMC end

abstract type AbstractBackwardKernel end

struct AnnealingPath{S <: AbstractVector{<:Real}}
    schedule
end

function AnnealingPath(schedule::AbstractVector)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2
    AnnealingPath{typeof(schedule)}(schedule)
end

Base.length(path::AnnealingPath) = length(path.schedule)

function anneal(path::AnnealingPath, t::Int, x, y)
    γt = path.schedule[t]
    (1 - γt)*x + γt*y
end

function annealed_logtarget(
    path     ::AnnealingPath,
    t        ::Int,
    x        ::AbstractVector,
    proposal,
    logtarget
)
    anneal(path, t, logpdf(proposal, x), logtarget(x))
end

struct DetailedBalance <: AbstractBackwardKernel end

struct ForwardKernel <: AbstractBackwardKernel end

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

function resample(rng, x, ℓw, ess, threshold)
    n_particles = size(x, 2)
    # Marginal likelihood update rule currently seems biased for
    # adaptive resampling.
    if true #ess < n_particles*threshold
        idx       = systematic_sampling(rng, exp.(ℓw))
        resampled = true
        x[:,:]    = x[:,idx]
        ℓw[:]    .= -log(n_particles)
        x, ℓw, idx, resampled
    else
        resampled = false
        ancestor  = collect(1:n_particles)
        x, ℓw, ancestor, resampled
    end
end

function reweight(ℓw, ℓG, ℓZ)
    N   = length(ℓw)
    ℓw  = ℓw + ℓG
    ℓZ  = ℓZ + logsumexp(ℓG) - log(N)
    ℓw  = ℓw .- logsumexp(ℓw)
    ess = exp(-logsumexp(2*ℓw))
    ℓw, ℓZ, ess
end

function euler_fwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x + h/2*Γ*∇logπt
end

function euler_bwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x - h/2*Γ*∇logπt
end

function leapfrog(logtarget, x, ν, h, M)
    ν = ν + h/2*ForwardDiff.gradient(logtarget, x)
    x = x - h*(M\ν)
    ν = ν + h/2*ForwardDiff.gradient(logtarget, x)
    x, ν
end
