
using Accessors
using Base.Iterators
using Distributions
using FillArrays
using ForwardDiff
using LinearAlgebra
using LogExpFunctions
using Optim
using PDMats
using Plots, StatsPlots
using ProgressMeter
using Random
using Random123
using SimpleUnPack
using Statistics

include("common.jl")
include("mcmc.jl")
include("sample.jl")
include("smc_ula.jl")
include("twist.jl")

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
    d      = size(Γ, 1)
    policy = [(a=zeros(F, d), b=zeros(F, d), c=zero(F)) for _ in 1:length(path)]
    smc    = SMCULA(Γ, h0, hT, backward, proposal, path)
    CSMCULA{typeof(smc), typeof(policy)}(smc, policy)
end

function twist_kernel_logmarginal(smc::SMCULA, twist, t, logtarget, x)
    (; path, h0, hT, Γ, proposal) = smc
    logπt(x) = annealed_logtarget(path, t, x, proposal, logtarget)
    ht    = anneal(path, t, h0, hT)
    q     = mapslices(xi -> euler_fwd(logπt, xi, ht, Γ), x, dims=1)
    twist_mvnormal_logmarginal(twist, q, ht*diag(Γ))
end

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::CSMCULA,
    logtarget,
    n_particles::Int,
)
    (; policy, smc) = sampler 
    ψ0       = first(policy)
    proposal = smc.proposal
    μ, Σ     = mean(proposal), Distributions._cov(proposal)
    x        = rand_twist_mvnormal(rng, ψ0, repeat(μ, 1, n_particles), diag(Σ))

    ℓG0  = 0.0
    ℓqψ0 = twist_mvnormal_logmarginal(ψ0, μ, diag(Σ))
    ℓψ0  = twist_logdensity(ψ0, x)
    ℓMψ  = twist_kernel_logmarginal(smc, policy[2], 2, logtarget, x)
    ℓGψ  = @. ℓG0 + ℓqψ0 + ℓMψ - ℓψ0

    x, ℓGψ
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::CSMCULA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget
)
    (; policy, smc) = sampler
    (; path, h0, hT, Γ, proposal) = smc
    logπt(x) = annealed_logtarget(path, t, x, proposal, logtarget)
    ht = anneal(path, t, h0, hT)

    ψ = policy[t]
    q = mapslices(xi -> euler_fwd(logπt, xi, ht, Γ), x, dims=1)
    x′ = rand_twist_mvnormal(rng, ψ, q, ht*diag(Γ))

    ℓG   = potential(smc, t, x′, x, logtarget)
    ℓψ   = twist_logdensity(ψ, x′)
    T    = length(smc)
    ℓGψ  = if t < T
        ψtp1 = policy[t+1]
        ℓMψ  = twist_kernel_logmarginal(smc, ψtp1, t+1, logtarget, x′)
        @. ℓG + ℓMψ - ℓψ
    elseif t == T
        @. ℓG - ℓψ
    end
    x′, ℓGψ, (q=q,)
end

function optimize_policy(sampler::CSMCULA, states)
    smc = sampler.smc
    (; path, h0, hT, Γ, proposal) = smc

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
            twist_prev     = policy_prev[t+1]
            a_prev, b_prev = twist_prev.a, twist_prev.b

            htp1       = anneal(path, t+1, h0, hT)
            qtp1       = states[t+1].q
            γ          = diag(Γ)
            K          = Diagonal(@. 1/(2*htp1*a_prev + 1/γ))
            μ_twisted  = K*(Γ\qtp1 .- htp1*b_prev)
            Σ_twisted  = htp1*K
            σ_twisted  = diag(Σ_twisted)
            logM       = twist_mvnormal_logmarginal(twist_recur, μ_twisted, σ_twisted)
            logG + logM
        end

        Δa, Δb, Δc, rmse = fit_quadratic(particles, -V)

        @info("", t, rmse = sum(
            abs2,
            twist_logdensity((a=Δa, b=Δb, c=Δc), particles) - V
        ))

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

    d            = 10
    μ            = Fill(10, d)
    Σ            = I
    logtarget(x) = logpdf(MvNormal(μ, Σ), x)
    n_episodes   = 5
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #h0    = 0.5
    #hT    = 0.05
    h0    = hT = 0.5
    Γ     = Diagonal(ones(d))

    n_iters  = 16
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    n_particles = 256
    res = @showprogress map(1:16) do _
        smc = SMCULA(
            Γ, h0, hT, PastForwardKernel(), proposal, AnnealingPath(schedule), 
        )
        csmc = CSMCULA(
            Γ, h0, hT, PastForwardKernel(), proposal, AnnealingPath(schedule), 
        )

        xs, _, stats_smc = sample(rng, smc, n_particles, 0.5, logtarget)

        xs, states, stats_csmc_init = sample(rng, csmc, n_particles, 0.5, logtarget)
        stats_csmc                  = stats_csmc_init
        for _ in 1:n_episodes
            csmc                   = optimize_policy(csmc, states)
            xs, states, stats_csmc = sample(rng, csmc, n_particles, 0.5, logtarget)
        end

        (
            last(stats_smc).logZ,
            last(stats_csmc_init).logZ,
            last(stats_csmc).logZ,
        )
        #(nothing, last(stats_csmc).logZ,)
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

function visualize()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 2
    μ            = Fill(10, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)
    n_episodes   = 3
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #h0    = 5e-2
    #hT    = 5e-3
    h0    = hT = 0.1
    Γ     = Diagonal(ones(d))
    n_iters  = 32
    schedule = range(0, 1; length=n_iters)

    csmc = CSMCULA(
        Γ, h0, hT, ForwardKernel(), proposal, AnnealingPath(schedule), 
    )

    Plots.plot() |> display

    n_particles = 256
    xs, states, stats_csmc_init = sample(rng, csmc, n_particles, 0.5, logtarget)
    stats_csmc                  = stats_csmc_init
    for (j, c)  in zip(1:3, [:Blues, :YlGn, :OrRd])
        csmc                   = optimize_policy(csmc, states)
        xs, states, stats_csmc = sample(rng, csmc, n_particles, 0.5, logtarget)

        bs = [twist.b[[1,2]] for twist in csmc.policy]
        Plots.plot!(
            [b[1] for b in bs],
            [b[2] for b in bs],
            linez=length(bs):-1:1,
            c     = c,
            #mark  = :circ,
            label = "episode $(j)"
        ) |> display
    end
end
