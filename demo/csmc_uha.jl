
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
include("smc_uha.jl")
include("twist.jl")

struct CSMCUHA{
    S <: SMCUHA, P <: AbstractVector
} <: AbstractSMC
    smc   ::S
    policy::P
end

Base.length(sampler::CSMCUHA) = length(sampler.smc.path)

function CSMCUHA(
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
    smc    = SMCUHA(M, δ0, δT, h, backward, proposal, path)
    CSMCUHA{typeof(smc), typeof(policy)}(smc, policy)
end

function twist_kernel_logmarginal(smc::SMCUHA, twist, t, ν)
    (; h, M,) = smc
    twist_mvnormal_logmarginal(twist, h*ν, (1 - h^2)*diag(M))
end

function rand_initial_with_potential(
    rng        ::Random.AbstractRNG,
    sampler    ::CSMCUHA,
               ::Any,
    n_particles::Int,
)
    (; smc, policy) = sampler
    (; proposal, M) = smc
    
    ψ0 = first(policy)
    d  = length(proposal)
    p  = rand_twist_mvnormal(rng, ψ0, Zeros(d, n_particles), diag(M))
    q  = rand(rng, proposal, n_particles)
    x  = vcat(q, p)

    ℓG0  = Zeros(n_particles)
    ℓqψ0 = twist_mvnormal_logmarginal(ψ0, Zeros(d), diag(M))
    ℓψ0  = twist_logdensity(ψ0, p)
    ℓMψ  = twist_kernel_logmarginal(smc, policy[2], 2, p)
    ℓGψ  = @. ℓG0 + ℓqψ0 + ℓMψ - ℓψ0

    x, ℓGψ
end

function mutate_with_potential(
    rng       ::Random.AbstractRNG,
    sampler   ::CSMCUHA,
    t         ::Int,
    x         ::AbstractMatrix,
    logtarget,
)
    (; policy, smc) = sampler
    (; path, δ0, δT, h, M, proposal) = smc
    logπt(x_)   = annealed_logtarget(path, t,   x_, proposal, logtarget)
    logπtm1(x_) = annealed_logtarget(path, t-1, x_, proposal, logtarget)
    δt = anneal(path, t, δ0, δT)

    ψ      = policy[t]
    d      = length(proposal)
    q, p   = x[1:d,:], x[d+1:end,:]
    p_dist = MvNormal(Zeros(d), M)

    phalf = rand_twist_mvnormal(rng, ψ, h*p, (1 - h^2)*diag(M))
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

    ℓψ   = twist_logdensity(ψ, phalf)
    T    = length(smc)
    ℓGψ  = if t < T
        ψtp1 = policy[t+1]
        ℓMψ  = twist_kernel_logmarginal(smc, ψtp1, t+1, p′)
        @. ℓG + ℓMψ - ℓψ
    elseif t == T
        @. ℓG - ℓψ
    end
    x′, ℓGψ, (phalf=phalf,)
end

function optimize_policy(sampler::CSMCUHA, states)
    smc = sampler.smc
    (; h, M, proposal) = smc

    policy_prev = sampler.policy
    proposal    = sampler.smc.proposal
    T           = length(sampler)
    policy_next = deepcopy(policy_prev)

    twist_recur = last(policy_prev)
    for t in T:-1:1
        (; particles, logG) = states[t]
        d = length(proposal)
        p = particles[d+1:end,:]

        ptilde = if t == 1
            p
        else
            states[t].phalf
        end

        V = if t == T
            logG
        else
            twist_prev     = policy_prev[t+1]
            a_prev, b_prev = twist_prev.a, twist_prev.b

            Σ         = (1 - h^2)*M
            Σii       = diag(Σ)
            K         = Diagonal(@. 1/(2*a_prev + 1/Σii))
            μ_twisted = K*(Σ\(h*p) .- b_prev)
            σ_twisted = diag(K)
            logM      = twist_mvnormal_logmarginal(twist_recur, μ_twisted, σ_twisted)
            logG + logM
        end

        Δa, Δb, Δc, rmse = fit_quadratic(ptilde, -V)

        @info("", t, rmse = sum(
            abs2,
            twist_logdensity((a=Δa, b=Δb, c=Δc), ptilde) - V
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

    d            = 2
    μ            = Fill(10, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)

    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    #δ0    = 5e-2
    #δT    = 5e-3
    δ0         = δT = 0.1
    h          = 0.5
    M          = Eye(d)
    n_episodes = 2

    #qs = underdamped_langevin(rng, logtarget, h, δ0, randn(rng, d), M, 1000)
    #return Plots.plot(qs[1,:], qs[2,:], marker=:circle)

    n_iters  = 32
    schedule = range(0, 1; length=n_iters)

    hline([0.0], label="True logZ") |> display

    n_particles = 256
    res = @showprogress map(1:8) do _
        smc = SMCUHA(
            M,
            δ0,
            δT,
            h,
            ForwardKernel(),
            proposal,
            AnnealingPath(schedule)
        )
        csmc = CSMCUHA(
            M,
            δ0,
            δT,
            h,
            ForwardKernel(),
            proposal,
            AnnealingPath(schedule)
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

    d            = 10
    μ            = Fill(5, d)
    logtarget(x) = logpdf(MvNormal(μ, I), x)
    n_episodes   = 10
    
    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)

    δ0         = δT = 0.1
    h          = 0.5
    M          = Eye(d)
    n_episodes = 5
    n_iters    = 32
    schedule   = range(0, 1; length=n_iters)

    csmc = CSMCUHA(
        M,
        δ0,
        δT,
        h,
        ForwardKernel(),
        proposal,
        AnnealingPath(schedule)
    )

    Plots.plot() |> display

    n_particles = 256
    xs, states, stats_csmc_init = sample(rng, csmc, n_particles, 0.5, logtarget)
    stats_csmc                  = stats_csmc_init
    for _ in 1:n_episodes
        csmc                   = optimize_policy(csmc, states)
        xs, states, stats_csmc = sample(rng, csmc, n_particles, 0.5, logtarget)

        bs = [twist.b[[1,2]] for twist in csmc.policy]
        Plots.plot!([b[1] for b in bs], [b[2] for b in bs]) |> display
    end
end
