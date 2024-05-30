
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using SimpleUnPack

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

function resample(rng, x, w, logw, q, ess)
    n_particles = size(x, 2)
    if ess < n_particles/2
        idx       = systematic_sampling(rng, w)       
        x         = x[:,idx]
        q         = q[:,idx]
        logw      = fill(-log(n_particles), n_particles)
        resampled = true
        ancestor  = idx
        x, q, logw, ancestor, resampled
    else
        resampled = false
        ancestor  = collect(1:n_particles)
        x, q, logw, ancestor, resampled
    end
end

struct QuadTwisting{
    M1 <: AbstractMatrix,
    M2 <: AbstractMatrix,
    V <: AbstractVector,
    S <: Real
}
    K::M1
    A::M2
    b::V
    c::S
end

function logpdf(twisting::QuadTwisting, x)
    @unpack A, b, c = twisting
    dot(x, A*x) + dot(x, b) + c
end

function randn_twisted(
    rng::Random.AbstractRNG, twisting::QuadTwisting, μ, Σ, h::Real
)
    @unpack K, b = twisting
    rand(rng, MvNormal(K*(Σ\μ - h*b), h*K))
end

function mutate_ula_twisted(
    rng, logtarget_annealed, x, h, Γ, γ, twisting
)
    ∇logπt = ForwardDiff.gradient(x_ -> logtarget_annealed(x_, γ), x)
    q′      = x + h/2*Γ*∇logπt
    randn_twisted(rng, twisting, q′, Γ, h), q′ 
end

function reweight(logw, Δlogw, logZ)
    logw = logw + Δlogw
    logw = logw .- logsumexp(logw)
    w    = exp.(logw)
    logZ = logZ + dot(Δlogw, w)
    ess  = 1 / sum(w.^2)
    w, logw, logZ, ess
end

function smc_ula(
    rng,
    logtarget,
    proposal::MvNormal,
    h, Γchol,
    n_particles::Int,
    schedule   ::AbstractVector,
    policy     ::AbstractVector{<:QuadTwisting}
)
    @assert first(schedule) == 0 && last(schedule) == 1
    T      = length(schedule)
    logπ   = logtarget
    Γ      = Hermitian(Γchol*Γchol')
    π0     = proposal

    logtarget_annealed(x_, γ) = (1 - γ)*logpdf(π0, x_) + γ*logπ(x_)
    qΓ⁻¹q  = zeros(n_particles)
    Δlogws = zeros(n_particles)
    logws  = zeros(n_particles)
    logZ   = 0.0

    ψ0    = first(policy)
    xs    = rand(rng, π0, n_particles)
    qs    = similar(xs)

    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    γ1 = schedule[1]
    for i in 1:size(xs,2)
        xi, qi  = mutate_ula_twisted(rng, logtarget_annealed, xs[:,i], h, Γ, Γchol, γ1)
        qs[:,i] = qi
        xs[:,i] = xi

        Δlogw     = logtarget_annealed(xi, γ1) - logpdf(π0, xi)
        Δlogws[i] = Δlogw
        qΓ⁻¹q[i]  = sum(abs2, Γchol\qi)
    end

    #ws, logws, logZ, ess               = reweight(logws, Δlogws, logZ)
    #xs, qs, logws, ancestor, resampled = resample(rng, xs, ws, logws, qs, ess)

    states[1] = (particles=xs)
    info[1]   = (iteration=1, ess=n_particles, logZ=logZ, resampled=false)

    for t in 2:T
        γprev = schedule[t-1]
        γcurr = schedule[t]
        ψ     = policy[t]

        for i in 1:size(xs,2)
            xi, qi  = mutate_ula_twisted(rng, logtarget_annealed, xs[:,i], h, Γ, γcurr, ψ)
            qs[:,i] = qi
            xs[:,i] = xi

            Δlogw     = logtarget_annealed(xi, γcurr) - logtarget_annealed(xi, γprev)
            Δlogws[i] = Δlogw
            qΓ⁻¹q[i]  = sum(abs2, Γchol\qi)
        end

        ws, logws, logZ, ess               = reweight(logws, Δlogws, logZ)
        xs, qs, logws, ancestor, resampled = resample(rng, xs, ws, logws, qs, ess)

        states[t] = (particles=xs, ancestor=ancestor, fwdeuler=qs, eulerquad=qΓ⁻¹q)
        info[t]   = (iteration=t, ess=ess, logZ=logZ, resampled=resampled)
    end
    xs, info
end

# function optimize_policy(states)
#     for state in reverse(states)
        
#     end
# end

function main()
    rng          = Random.default_rng()
    d            = 10
    μ            = randn(d)
    println(μ)
    logtarget(x) = logpdf(MvNormal(μ, 0.01*I), x) - 100
    proposal     = MvNormal(zeros(d), I)
    h            = 5e-2
    n_particles  = 512
    n_iters      = 8
    schedule     = range(0, 1; length=n_iters)

    policy = vcat(
        QuadTwisting(cov(proposal), Zeros(d,d), Zeros(d), 0.0),
        fill(QuadTwisting(I, Zeros(d,d), Zeros(d), 0.0), length(schedule)-1)
    )
    logZs = @showprogress map(1:32) do _
        _, stats    = smc_ula(
            rng, logtarget, proposal,
            h, Diagonal(ones(d)),
            n_particles, schedule, policy
        )
        last(stats).logZ
    end

    #println(mean(xs, dims=2)[:,1])
    violin(ones(length(logZs)), logZs)
    dotplot!(ones(length(logZs)), logZs)
end
