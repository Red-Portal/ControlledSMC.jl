
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using SimpleUnPack
using FillArrays

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

function reweight(logw, Δlogw, logZ)
    logw = logw + Δlogw
    logw = logw .- logsumexp(logw)
    w    = exp.(logw)
    logZ = logZ + dot(Δlogw, w)
    ess  = 1 / sum(w.^2)
    w, logw, logZ, ess
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

function Distributions.logpdf(twisting::QuadTwisting, x)
    @unpack A, b, c = twisting
    dot(x, A*x) + dot(x, b) + c
end

struct MvNormalTwisted{TW <: QuadTwisting, MVN <: MvNormal}
    twisting::TW 
    mvnormal::MVN
end

function Distributions.rand(
    rng        ::Random.AbstractRNG,
    mvtwist    ::MvNormalTwisted,
    n_particles::Int
)
    @unpack K, b, c = mvtwist.twisting
    mvnormal = mvtwist.mvnormal
    μ, Σ = mean(mvnormal), cov(mvnormal)
    rand(rng, MvNormal(K*(Σ\μ - b), K), n_particles)
end

function logmarginal(mvtwist::MvNormalTwisted)
    @unpack K, b, c = mvtwist.twisting
    mvnormal = mvtwist.mvnormal
    μ, Σ  = mean(mvnormal), cov(mvnormal)
    Σinvμ = Σ\μ
    z     = Σinvμ - b
    (
        -logdet(Σ)
        + logdet(K)
        + dot(z, K*z)
        - dot(μ, Σinvμ)
        - c
     )/2
end

function logmarginal_ula_twisted(twisting, q, Γ, h, logπincr)
    @unpack K, A, b, c = twisting
    Γinvq = Γ\q 
    z     = Γinvq - h*b
    (
        -logdet(Γ)
        + logdet(K)
        + dot(z, K*z)/h
        - dot(q, Γinvq)/h
        - c
        + logπincr
     )/2
end

function euler_fwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x + h/2*Γ*∇logπt
end

function mutate_ula_twisted(
    rng, q, h, Γ, Γchol
)
    q + h*(Γchol*randn(rng, length(q)))
    #randn_twisted(rng, twisting, q′, Γ, h), q′ 
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
    @assert length(schedule) > 2

    T      = length(schedule)
    logπ   = logtarget
    Γ      = Hermitian(Γchol*Γchol')
    π0     = proposal

    logtarget_annealed(x_, γ) = (1 - γ)*logpdf(proposal, x_) + γ*logπ(x_)
    qΓ⁻¹q  = zeros(n_particles)
    Δlogws = zeros(n_particles)
    logws  = zeros(n_particles)
    logZ   = 0.0

    ψ0 = first(policy)
    ψ1 = policy[2]
    π0 = MvNormalTwisted(ψ0, proposal)
    xs = rand(rng, π0, n_particles)
    qs = similar(xs)

    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    γcurr = schedule[1]
    γnext = schedule[2]
    for i in 1:size(xs,2)
        x    = xs[:,i]
        G0   = logtarget_annealed(x, γcurr) - logpdf(proposal, x)
        q′   = euler_fwd(Base.Fix2(logtarget_annealed, γcurr), x, h, Γ )

        ℓμψ  = logmarginal(π0)
        Δℓπ  = logtarget_annealed(x, γnext) - logtarget_annealed(x, γcurr)
        ℓM1ψ = logmarginal_ula_twisted(ψ1, q′, Γ, h, Δℓπ)

        Δlogws[i] = ℓμψ + G0 + ℓM1ψ
        qs[:,i]   = q′
        qΓ⁻¹q[i]  = sum(abs2, Γchol\q′)
    end

    ws, logws, logZ, ess               = reweight(logws, Δlogws, logZ)
    xs, qs, logws, ancestor, resampled = resample(rng, xs, ws, logws, qs, ess)

    states[1] = (particles=xs, ancestor=ancestor, fwdeuler=qs, eulerquad=qΓ⁻¹q)
    info[1]   = (iteration=1, ess=ess, logZ=logZ, resampled=resampled)

    for t in 2:T
        γprev = schedule[t-1]
        γcurr = schedule[t]
        ψ     = policy[t]

        for i in 1:size(xs,2)
            x     = xs[:,i]
            q     = euler_fwd(Base.Fix2(logtarget_annealed, γcurr), x, h, Γ)
            x′    = mutate_ula_twisted(rng, q, h, Γ, Γchol)
            Δlogw = logtarget_annealed(x, γcurr) - logtarget_annealed(x, γprev)

            qs[:,i]   = q
            xs[:,i]   = x′
            Δlogws[i] = Δlogw
            qΓ⁻¹q[i]  = sum(abs2, Γchol\q)
        end

        ws, logws, logZ, ess               = reweight(logws, Δlogws, logZ)
        xs, qs, logws, ancestor, resampled = resample(rng, xs, ws, logws, qs, ess)

        states[t] = (particles=xs, ancestor=ancestor, fwdeuler=qs, eulerquad=qΓ⁻¹q)
        info[t]   = (iteration=t, ess=ess, logZ=logZ, resampled=resampled)
    end
    xs, info
end

function optimize_policy(states)
     for state in reverse(states)
         
     end
end

function main()
    rng          = Random.default_rng()
    d            = 10
    μ            = randn(d)
    println(μ)
    logtarget(x) = logpdf(MvNormal(μ, 0.01*I), x)

    μ0           = Zeros(d)
    Σ0           = 2*Eye(d)
    proposal     = MvNormal(μ0, Σ0)
    h            = 3e-2
    n_particles  = 512
    n_iters      = 32
    schedule     = range(0, 1; length=n_iters)

    policy = vcat(
        QuadTwisting(cov(proposal), Zeros(d,d), Zeros(d), 0.0),
        fill(QuadTwisting(Eye(d), Zeros(d,d), Zeros(d), 0.0), length(schedule)-1)
    )
    logZs = @showprogress map(1:256) do _
        _, stats    = smc_ula(
            rng, logtarget, proposal, h, Eye(d), n_particles, schedule, policy
        )
        last(stats).logZ
    end

    #println(mean(xs, dims=2)[:,1])
    violin(ones(length(logZs)), logZs)
    dotplot!(ones(length(logZs)), logZs)
end
