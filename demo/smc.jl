
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
using Random123

include("common.jl")

function mutate_ula(rng, logπt, x, h, Γ, Γchol)
    q = euler_fwd(logπt, x, h, Γ)
    q + sqrt(2*h)*Γchol*randn(rng, length(q))
end

function potential(logtarget_annealed, x_prev, x_curr, γprev, γcurr, h, Γ)
    logtarget_annealed(x_prev, γcurr) - logtarget_annealed(x_prev, γprev)
end

# function potential(logtarget_annealed, x_prev, x_curr, γprev, γcurr, h, Γ)
#      ℓπ_prev = Base.Fix2(logtarget_annealed, γprev)
#      ℓπ_curr = Base.Fix2(logtarget_annealed, γcurr)
#      L_fwd   = euler_fwd(ℓπ_curr, x_prev, h, Γ)
#      L_bwd   = euler_fwd(ℓπ_prev, x_curr, h, Γ)
#      (ℓπ_curr(x_curr) + logpdf(MvNormal(L_bwd, 2*h*Γ), x_prev)) -
#          (ℓπ_prev(x_prev) + logpdf(MvNormal(L_fwd, 2*h*Γ), x_curr))
# end

function smc_ula(
    rng,
    logtarget,
    proposal::MvNormal,
    h, Γchol,
    n_particles::Int,
    schedule   ::AbstractVector,
)
    @assert first(schedule) == 0 && last(schedule) == 1
    @assert length(schedule) > 2

    T      = length(schedule)
    logπ   = logtarget
    Γ      = Hermitian(Γchol*Γchol')
    π0     = proposal

    logtarget_annealed(x_, γ) = (1 - γ)*logpdf(proposal, x_) + γ*logπ(x_)

    xs    = rand(rng, π0, n_particles)
    logws = fill(-log(n_particles), n_particles)
    Gs    = zeros(n_particles)
    logZ  = 0.0

    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    states[1] = (particles=xs,)
    info[1]   = (iteration=1, ess=n_particles, logZ=logZ)

    for t in 2:T
        γprev = schedule[t-1]
        γcurr = schedule[t]

        @inbounds for i in 1:size(xs,2)
            x_prev = xs[:,i]
            x_curr = mutate_ula(rng, Base.Fix2(logtarget_annealed, γcurr), x_prev, h, Γ, Γchol)
            G      = potential(logtarget_annealed, x_prev, x_curr, γprev, γcurr, h, Γ)

            xs[:,i] = x_curr
            Gs[i]   = G
        end

        ws, logws, logZ, ess               = reweight(logws, Gs, logZ)
        xs, ws, logws, ancestor, resampled = resample(rng, xs, ws, logws, ess)

        states[t] = (particles=xs, ancestor=ancestor)
        info[t]   = (iteration=t, ess=ess, logZ=logZ, resampled=resampled)
    end
    xs, info
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    d            = 4
    μ            = randn(d)/2
    logtarget(x) = logpdf(MvNormal(μ, 0.01*I), x)

    μ0           = Zeros(d)
    Σ0           = Eye(d)
    proposal     = MvNormal(μ0, Σ0)
    h            = .3e-3
    #n_particles  = 512
    n_iters      = 16
    schedule     = range(0, 1; length=n_iters).^2

    for (idx, n_particles) in enumerate([64, 256, 1024, 4096, 16384])
    res = @showprogress map(1:64) do _
        xs, stats    = smc_ula(
            rng, logtarget, proposal, h, Eye(d), n_particles, schedule
        )
        (mean(xs, dims=2)[:,1], last(stats).logZ)
    end

    logZ = [last(r) for r in res]
    x    = hcat([first(r) for r in res]...)

    #scatter(x[1,:], x[2,:]) |> display
    #scatter!([μ[1]], [μ[2]]) |> display

    violin!(fill(idx, length(logZ)), logZ)  |> display
    dotplot!(fill(idx, length(logZ)), logZ) |> display
    end
end
