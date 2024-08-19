

using Accessors
using Statistics
using Distributions
using Random
using ForwardDiff
using LogExpFunctions
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using Optim
using PDMats
using SimpleUnPack
using FillArrays
using Random123

include("common.jl")
include("sample.jl")
include("mcmc.jl")
include("smc_klmc.jl")
include("twist.jl")

struct CSMCKLMC{
    S <: SMCKLMC, P <: AbstractVector
} <: AbstractSMC
    smc   ::S
    policy::P
end

Base.length(sampler::CSMCKLMC) = length(sampler.smc.path)

function CSMCKLMC(
    h       ::F,
    gamma   ::F,
    sigma2  ::F,
    backward::AbstractBackwardKernel,
    proposal::MvNormal,
    path    ::AnnealingPath,
) where {F<:Real}
    d      = length(proposal)
    policy = [(a=zeros(F, 2*d), b=zeros(F, 2*d), c=zero(F)) for _ in 1:length(path)]
    smc    = SMCKLMC(h, gamma, sigma2, backward, proposal, path)
    CSMCKLMC{typeof(smc), typeof(policy)}(smc, policy)
end

function twist_kernel_logmarginal(smc::SMCKLMC, twist, t, logtarget, z)
    (; path, proposal, h, Σ_fwd, γ) = smc
    d = length(proposal)

    logπt(x_) = annealed_logtarget(path, t, x_, proposal, logtarget)

    x, v  = z[1:d,:], z[d+1:end,:]
    μ_fwd = klmc_fwd(logπt, x, v, h, γ)

    twist_mvnormal_logmarginal(twist, μ_fwd, Σ_fwd)
end

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::CSMCKLMC,
    logtarget ,
    n_particles::Int,
)
    (; smc, policy) = sampler
    (; proposal, ) = smc
    
    ψ0 = first(policy)
    d  = length(proposal)
    μ  = mean(proposal)
    Σ  = cov(proposal)

    μ_z = vcat(repeat(μ, 1, n_particles), zeros(d, n_particles))
    Σ_z = vcat(diag(Σ), ones(d))
    z   = rand_twist_mvnormal(rng, ψ0, μ_z, Σ_z)

    ℓG0  = Zeros(n_particles)
    ℓqψ0 = twist_mvnormal_logmarginal(ψ0, μ_z, Σ_z)
    ℓψ0  = twist_logdensity(ψ0, z)
    ℓMψ  = twist_kernel_logmarginal(smc, policy[2], 2, logtarget, z)
    ℓGψ  = @. ℓG0 + ℓqψ0 + ℓMψ - ℓψ0

    z, ℓGψ
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::CSMCKLMC,
    t         ::Int,
    z         ::AbstractMatrix,
    logtarget,
)
    (; policy, smc) = sampler
    (; path, h, γ, Σ_fwd, proposal) = smc
    logπt(x_)   = annealed_logtarget(path, t,   x_, proposal, logtarget)
    logπtm1(x_) = annealed_logtarget(path, t-1, x_, proposal, logtarget)
    d           = length(proposal)
    n           = size(z, 2)
    v_dist      = MvNormal(Zeros(d), I)

    ψ      = policy[t]
    x, v   = z[1:d,:], z[d+1:end,:]

    μ_fwd = klmc_fwd(logπt, x, v, h, γ)
    z′     = rand_twist_mvnormal(rng, ψ, μ_fwd, Σ_fwd)
    x′, v′ = z′[1:d,:], z′[d+1:end,:]

    ℓπt     = logπt.(eachcol(x′)) 
    ℓπtm1   = logπtm1.(eachcol(x)) 
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(v′))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(v))

    μ_bwd = klmc_fwd(logπtm1, x′, -v′, h, γ)

    ℓF = map(i -> logpdf(MvNormal(μ_fwd[:,i], Σ_fwd), z′[:,i]), 1:n) 
    ℓB = map(i -> logpdf(MvNormal(μ_bwd[:,i], Σ_fwd), z[:,i]), 1:n)
    ℓG = ℓπt - ℓπtm1 + ℓB - ℓF + ℓauxt - ℓauxtm1 

    ℓψ   = twist_logdensity(ψ, z′)
    T    = length(smc)
    ℓGψ  = if t < T
        ℓMψ = twist_kernel_logmarginal(smc, policy[t+1], t+1, logtarget, z′)
        @. ℓG + ℓMψ - ℓψ
    elseif t == T
        @. ℓG - ℓψ
    end
    z′, ℓGψ, (μ_fwd=μ_fwd,)
end

function optimize_policy(sampler::CSMCKLMC, states)
    smc = sampler.smc
    (; proposal, Σ_fwd) = smc

    policy_prev = sampler.policy
    proposal    = sampler.smc.proposal
    T           = length(sampler)
    policy_next = deepcopy(policy_prev)

    twist_recur = last(policy_prev)
    for t in T:-1:1
        (; particles, logG) = states[t]

        V = if t == T
            logG
        else
            twist_prev      = policy_prev[t+1]
            a_prev, b_prev  = twist_prev.a, twist_prev.b

            μ_fwd_tp1 = states[t+1].μ_fwd
            K         = inv(Diagonal(2*a_prev) + inv(PDMats.PDMat(Σ_fwd)))
            μ_twisted = K*(Σ_fwd\μ_fwd_tp1 .- b_prev)
            Σ_twisted = K
            logM      = twist_mvnormal_logmarginal(twist_recur, μ_twisted, Σ_twisted) 
            logG + logM
        end

        Δa, Δb, Δc, rmse = fit_quadratic(particles, -V)

        @info("", t, rmse)

        a_next = policy_prev[t].a + Δa
        b_next = policy_prev[t].b + Δb
        c_next = policy_prev[t].c + Δc

        policy_next[t] = (a=a_next, b=b_next, c=c_next)
        twist_recur    = (a=Δa, b=Δb, c=Δc)
    end
    @set sampler.policy = policy_next
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 5
    μ            = Fill(10, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)

    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    h  = 8.0
    γ  = 5.0
    σ2 = 2*γ
    M  = Eye(d)

    n_episodes = 2 #3

    #qs = underdamped_langevin(rng, logtarget, h, δ0, randn(rng, d), M, 1000)
    #return Plots.plot(qs[1,:], qs[2,:], marker=:circle)

    n_iters  = 32
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    n_particles = 256
    res = @showprogress map(1:8) do _

        smc = SMCKLMC(
            h, γ, σ2,
            ForwardKernel(),
            proposal,
            AnnealingPath(schedule)
        )

        xs, _, stats_smc = sample(rng, smc, n_particles, 0.5, logtarget)

        csmc = CSMCKLMC(
            h, γ, σ2,
            ForwardKernel(),
            proposal,
            AnnealingPath(schedule)
        )
        xs, states, stats_csmc_init = sample(rng, csmc, n_particles, 0.5, logtarget)
        #Plots.plot!([stats.logZ for stats in stats_csmc_init]) |> display

        stats_csmc                  = stats_csmc_init
        for _ in 1:n_episodes
            csmc                   = optimize_policy(csmc, states)
            xs, states, stats_csmc = sample(rng, csmc, n_particles, 0.5, logtarget)
            #Plots.plot!([stats.logZ for stats in stats_csmc]) |> display
        end
        #throw()

        (
            last(stats_smc).logZ,
            last(stats_csmc_init).logZ,
            last(stats_csmc).logZ,
        )
    end

    logZ = [first(r) for r in res]
    violin!( fill(1, length(logZ)), logZ, fillcolor=:blue, alpha=0.2, label="SMC N=$(n_particles)") |> display
    dotplot!(fill(1, length(logZ)), logZ, markercolor=:blue, label=nothing) |> display

    logZ = [r[2] for r in res]
    violin!( fill(2, length(logZ)), logZ, fillcolor=:red, alpha=0.2, label="CSMC N=$(n_particles) J=0") |> display
    dotplot!(fill(2, length(logZ)), logZ, markercolor=:red, label=nothing) |> display

    logZ = [last(r) for r in res]
    violin!( fill(3, length(logZ)), logZ, fillcolor=:red, alpha=0.2, label="CSMC N=$(n_particles) J=$(n_episodes)") |> display
    dotplot!(fill(3, length(logZ)), logZ, markercolor=:red, label=nothing) |> display
end
