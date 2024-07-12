
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using PDMats
using SimpleUnPack
using FillArrays
using Random123

include("common.jl")
include("sample.jl")
include("mcmc.jl")
include("smc_uha.jl")
include("twist.jl")

struct CSMCUHA{
    S <: SMCUHA, P <: AbstractVector
} <: AbstractSMC
    smc   ::S
    policy::P
end

Base.length(sampler::CSMCUHA) = length(sampler.smc.path)

function CSMCULA(
    M       ::Diagonal,
    δ0      ::F,
    δT      ::F,
    h       ::F,
    backward::AbstractBackwardKernel,
    proposal::MvNormal,
    path    ::AnnealingPath,
) where {F<:Real}
    d      = size(M, 1)
    policy = [(a=zeros(F, d), b=zeros(F, d), c=zero(F)) for _ in 1:length(path)]
    smc    = CSMCULA(M, δ0, δT, h, backward, proposal, path)
    CSMCULA{typeof(smc), typeof(policy)}(smc, policy)
end

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::CSMCUHA,
               ::Any,
    n_particles::Int,
)
    (; smc, policy) = sampler
    (; proposal, M) = smc
    
    ψ0       = first(policy)
    proposal = proposal
    μ, Σ     = mean(proposal), Distributions._cov(proposal)
    d        = length(μ)
    p        = rand_twist_mvnormal(rng, ψ0, Zeros(d, 1, n_particles), M)

    q = rand(rng, proposal, n_particles)
    d = size(q, 1)
    p = rand(rng, MvNormal(Zeros(d), M), n_particles)
    x = vcat(q, p)
    ℓG = zeros(n_particles)
    x, ℓG
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::CSMCUHA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget,
)
    (; smc, policy) = sampler
    (; path, δ0, δT, h, M, proposal) = smc
    logπt(x_)   = annealed_logtarget(path, t,   x_, proposal, logtarget)
    logπtm1(x_) = annealed_logtarget(path, t-1, x_, proposal, logtarget)
    δt = anneal(path, t, δ0, δT)

    d      = size(x, 1) ÷ 2
    q, p   = x[1:d,:], x[d+1:end,:]
    p_dist = MvNormal(Zeros(d), M)

    phalf = h*p + sqrt(1 - h^2)*unwhiten(M, randn(rng, size(p)))
    q′, p′ = leapfrog(logπt, q, phalf, δt, M)
    x′     = vcat(q′, p′)

    ℓπt     = logπt.(eachcol(q′)) 
    ℓπtm1   = logπtm1.(eachcol(q)) 
    ℓauxt   = logpdf.(Ref(p_dist), eachcol(p′))
    ℓauxtm1 = logpdf.(Ref(p_dist), eachcol(p))
    B       = MvNormal.(h*eachcol(phalf), Ref((1 - h^2)*M))
    F       = MvNormal.(h*eachcol(p),     Ref((1 - h^2)*M))
    ℓB      = logpdf.(B, eachcol(p))
    ℓF      = logpdf.(F, eachcol(phalf))
    ℓG      = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓB - ℓF
    x′, ℓG, NamedTuple()
end

function underdamped_langevin(rng, logtarget, h, δ, q, M, n_samples)
    d      = length(q)
    p_dist = MvNormal(Zeros(d), M)
    p      = rand(rng, p_dist)

    qs = zeros(d, n_samples)
    for i in 1:n_samples
        phalf = h*p + sqrt(1 - h^2)*randn(rng, d)
        q, p  = leapfrog(logtarget, q, phalf, δ, M)
        qs[:,i] = q
    end
    qs
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 20
    μ            = Fill(10, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)

    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #δ0    = 5e-2
    #δT    = 5e-3
    δ0    = δT = 1.0
    h     = 0.5
    M     = Eye(d)

    #qs = underdamped_langevin(rng, logtarget, h, δ0, randn(rng, d), M, 1000)
    #return Plots.plot(qs[1,:], qs[2,:], marker=:circle)

    n_iters  = 32
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    sampler = CSMCUHA(
        M,
        δ0,
        δT,
        h,
        ForwardKernel(),
        proposal,
        AnnealingPath(schedule)
    )

    particles = [32, 64, 128, 256, 512, 1024]
    for (idx, n_particles) in enumerate(particles)
        res = @showprogress map(1:64) do _
            xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
            (mean(xs, dims=2)[:,1], last(stats).logZ)
        end

        logZ = [last(r) for r in res]

        violin!( fill(2*idx-1, length(logZ)), logZ, fillcolor  =:blue, alpha=0.2, label="N=$(n_particles)") |> display
        dotplot!(fill(2*idx-1, length(logZ)), logZ, markercolor=:blue, label=nothing) |> display
    end

    # sampler = CSMCUHA(
    #     Γ,
    #     h0,
    #     hT,
    #     DetailedBalance(),
    #     proposal,
    #     AnnealingPath(schedule)
    # )
    # for (idx, n_particles) in enumerate(particles)
    #     res = @showprogress map(1:64) do _
    #         xs, _, stats    = sample(rng, sampler, n_particles, 0.5, logtarget)
    #         (mean(xs, dims=2)[:,1], last(stats).logZ)
    #     end
    #     logZ = [last(r) for r in res]
    #     violin!( fill(2*idx, length(logZ)), logZ, fillcolor  =:red, alpha=0.2, label="N=$(n_particles)") |> display
    #     dotplot!(fill(2*idx, length(logZ)), logZ, markercolor=:red, label=nothing) |> display
    # end
end
