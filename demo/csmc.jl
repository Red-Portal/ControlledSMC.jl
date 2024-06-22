
using Accessors
using Distributions
using FillArrays
using ForwardDiff
using LinearAlgebra
using LogExpFunctions
using PDMats
using Plots, StatsPlots
using ProgressMeter
using Random
using Random123
using SimpleUnPack
using Statistics

include("common.jl")
include("sample.jl")
include("smc.jl")

struct CSMCULA{S <: SMCULA, P <: AbstractVector} <: AbstractSMC
    smc   ::S
    policy::P
end

Base.length(sampler::CSMCULA) = length(sampler.smc)

function CSMCULA(
    Γ       ::Diagonal,
    h0      ::F,
    hT      ::F,
    backward::AbstractBackwardKernel,
    proposal::MvNormal,
    path    ::AnnealingPath,
) where {F<:Real}
    Σ0 = Distributions._cov(proposal)
    d  = size(Γ, 1)
    T  = length(path)
    policy = map(1:T) do t
        A  = Diagonal(zeros(d))
        b  = zeros(d)
        c  = 0.0 
        K  = if t == 1
            Σ0
        else
            Γ
        end
        (A=A, b=b, c=c, K=K)
    end
    smc = SMCULA(Γ, h0, hT, backward, proposal, path)
    CSMCULA{typeof(smc), typeof(policy)}(smc, policy)
end

function rand_initial_particles(
    rng        ::Random.AbstractRNG,
    sampler    ::CSMCULA,
    n_particles::Int,
)
    (; policy, smc) = sampler 
    proposal = smc.proposal
    (; K, b) = first(policy)
    μ  = mean(proposal)
    Σ  = Distributions._cov(proposal)
    qψ = MvNormal(K*(Σ\μ - b), K)
    rand(rng, qψ, n_particles)
end

function twisted_kernel_marginal(sampler::CSMCULA, t, logtarget, x)
    (; smc, policy)               = sampler
    (; path, h0, hT, Γ, proposal) = smc
    @unpack A, K, b, c = policy[t+1]
    logπt(x)   = annealed_logtarget(path,  t,  x, proposal, logtarget)
    logπtp1(x) = annealed_logtarget(path, t+1, x, proposal, logtarget)
    h = anneal(path, t+1, h0, hT)

    ℓπt_xt   = map(xi -> logπt(xi),   eachcol(x))
    ℓπtp1_xt = map(xi -> logπtp1(xi), eachcol(x))

    ℓdetΓ = logdet(Γ)
    ℓdetK = logdet(K)
    q     = mapslices(xi -> euler_fwd(logπt, xi, h, Γ), x, dims=1)
    z     = Γ\q .- h*b
    Δlogπ = ℓπtp1_xt - ℓπt_xt

    ((-ℓdetΓ + ℓdetK) .+ quad(K, z)/h - invquad(Γ, q)/h)/2 .- c + Δlogπ
end

function potential_init(
    sampler ::CSMCULA,
    x       ::AbstractMatrix,
    logtarget
)
    (; smc, policy) = sampler
    ψ0 = first(policy)

    @unpack A, K, b, c = ψ0

    μ    = mean(smc.proposal)
    Σ    = Distributions._cov(smc.proposal)
    ℓG0  = 0.0
    z    = Σ\μ - b
    ℓqψ0 = (-logdet(Σ) + logdet(K) + quad(K, z) - quad(Σ, μ))/2 - c
    ℓψ0  = -sum(x.*(A*x), dims=1)[1,:] - (b'*x)[1,:] .- c
    ℓMψ  = twisted_kernel_marginal(sampler, 1, logtarget, x)

    ℓG0 + ℓqψ0 .+ (ℓMψ - ℓψ0)
end

function mutate(
    rng       ::Random.AbstractRNG,
    sampler   ::CSMCULA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget
)
    (; path, h0, hT, Γ, proposal) = sampler.smc
    (; b, K) = sampler.policy[t]
    logπt(x) = annealed_logtarget(path, t, x, proposal, logtarget)
    ht = anneal(path, t, h0, hT)
    q  = mapslices(xi -> euler_fwd(logπt, xi, ht, Γ), x, dims=1)

    K*(Γ\q .- ht*b) + sqrt(ht)*unwhiten(K, randn(rng, size(q)))
end

function potential(
    sampler   ::CSMCULA,
    t         ::Int,
    x_curr    ::AbstractMatrix,
    x_prev    ::AbstractMatrix,
    logtarget,
)
    (; smc, policy)    = sampler
    (; A, b, c)        = policy[t]
    (; path, proposal) = smc

    ℓG = potential(smc, t, x_curr, x_prev, logtarget)

    logπt(x)   = annealed_logtarget(path,   t, x, proposal, logtarget)
    logπtm1(x) = annealed_logtarget(path, t-1, x, proposal, logtarget)
    ℓπt_xtm1   = map(xi -> logπt(xi),   eachcol(x_prev))
    ℓπtm1_xtm1 = map(xi -> logπtm1(xi), eachcol(x_prev))
    Δℓπ        = ℓπt_xtm1 - ℓπtm1_xtm1

    ℓψ = -sum(x_curr.*(A*x_curr), dims=1)[1,:] - (b'*x_curr)[1,:] .- c + Δℓπ
    T  = length(smc)

    if t < T
        ℓMψ = twisted_kernel_marginal(sampler, t, logtarget, x_curr)
        ℓG + ℓMψ - ℓψ
    elseif t == T
        ℓG - ℓψ
    end
end

function optimize_policy(sampler, states)
    policy   = sampler.policy
    proposal = sampler.smc.proposal

    Σ0 = Distributions._cov(proposal)

    particles, ancestors, logG = first(states)

    ψ0 = first(policy)
    d  = size(particles, 1)
    n  = size(particles, 2)

    @unpack A, b, c = ψ0

    #particles = particles[:,ancestors]
    #logG      = logG

    y  = logG #+ logM
    Xt = vcat(particles.^2, particles, ones(1, n))
    X  = Array(Xt')
    β  = cholesky(X'*X + 1e-7*I)\(X'*y)

    ΔA = Diagonal(abs.(β[1:d]))
    Δb = β[d+1:2*d]
    Δc = β[end]

    A += ΔA
    b += Δb
    c += Δc

    # println(diag(A))
    # println(b)
    # println(c)
    # throw()

    K = inv(inv(Σ0) + 2*A)

    @set sampler.policy[1] = (A=A, b=b, c=c, K=K)
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 4
    μ            = randn(d)/2
    logtarget(x) = logpdf(MvNormal(μ, 0.01*I + 0.001*ones(d,d)), x)
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #h0    = 5e-2
    #hT    = 5e-3
    h0    = hT = 5e-3
    Γ     = Diagonal(ones(d))

    n_iters  = 16
    schedule = range(0, 1; length=n_iters).^2

    hline([0.0]) |> display

    smc = SMCULA(
        Γ, h0, hT, ForwardKernel(), proposal, AnnealingPath(schedule), 
    )
    csmc = CSMCULA(
        Γ, h0, hT, ForwardKernel(), proposal, AnnealingPath(schedule), 
    )

    n_particles = 128
    res = @showprogress map(1:64) do _
        xs, _, stats_smc = sample(rng, smc, n_particles, 0.5, logtarget)

        xs, states, stats_csmc0 = sample(rng, csmc, n_particles, 0.5, logtarget)
        #csmc                    = optimize_policy(csmc, states)
        #xs, _, stats_csmc1      = sample(rng, csmc, n_particles, 0.5, logtarget)

        (last(stats_smc).logZ, last(stats_csmc0).logZ,)
    end
    logZ = [first(r) for r in res]
    violin!( fill(1, length(logZ)), logZ,   fillcolor=:blue, alpha=0.2) |> display
    dotplot!(fill(1, length(logZ)), logZ, markercolor=:blue)            |> display

    logZ = [last(r) for r in res]
    violin!( fill(2, length(logZ)), logZ,   fillcolor=:red, alpha=0.2) |> display
    dotplot!(fill(2, length(logZ)), logZ, markercolor=:red)            |> display
end
