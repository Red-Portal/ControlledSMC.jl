
using Distributions
using FillArrays
using ForwardDiff
using LinearAlgebra
using PDMats
using Plots
using ProgressMeter
using Random

struct ULA
    d
    h
end

struct KLMC
    d
    h
    γ
end

function kernel_mean(k::ULA, ∇U, x)
    (; h) = k
    x + h/2*-∇U(x)
end

function kernel_mean(k::KLMC, ∇U, z)
    (; d, h, γ) = k

    x, v = z[1:d], z[d+1:end]

    b = ∇U(x)
    η = exp(-γ*h)

    μ_x = x + (1 - η)/γ*v - 1/(γ*γ)*(h*γ + η - 1)*b
    μ_v = η*v - (1 - η)/γ*b
    vcat(μ_x, μ_v)
end

function kernel_cov(k::ULA)
    (; d, h) = k
    PDMats.ScalMat(d, 2*h)
end

function kernel_cov(k::KLMC)
    (; d, h, γ) = k
    σ2   = 2*γ
    η    = exp(-γ*h)
    σ2xx = σ2/(2*γ^2)*(2*h - (3 - 4*η + η^2)/γ)
    σ2xv = σ2/(2*γ^2)*(1 - η)^2
    σ2vv = σ2/(2*γ)*(1 - η^2)
    Σ    = zeros(2*d, 2*d)

    @inbounds for i in 1:d
        Σ[i,i] = σ2xx
    end
    @inbounds for i in d+1:2*d
        Σ[i,i] = σ2vv
    end
    @inbounds for i in 1:d
        Σ[i+d,i] = σ2xv
        Σ[i,i+d] = σ2xv
    end
    PDMats.PDMat(Σ)
end

function rand_reflection_coupling(
    rng::Random.AbstractRNG,
    μ1 ::AbstractVector,
    μ2 ::AbstractVector,
    Σ  ::PDMats.AbstractPDMat
)
    # Maximal reflection coupling for normal random walk proposals
    #   x′ ~ N(μ1, Σ) (marginally)
    #   y′ ~ N(μ2, Σ) (marginally)
    #
    # Official Rcpp implementation by Pierre Jacob:
    #     https://github.com/pierrejacob/unbiasedmcmc/blob/ ...
    #         a1eea04eef08710463ff9ff2228f5e58565fe78b/src/mvnorm.cpp#L185

    q_x = MvNormal(μ1, Σ)
    q_y = MvNormal(μ2, Σ)
    u   = rand(rng)

    x′ = rand(rng, q_x)

    if log(u) + logpdf(q_x, x′) < logpdf(q_y, x′)
        return x′, x′, true
    else
        ξ  = x′ - μ1
        η  = -ξ
        y′ = μ2 + η
        return x′, y′, false
    end
end

function rand_lagged_coupling(rng, k, π0, ∇U, L; tmax::Int=10^6)
    @assert tmax > 2*L

    x = rand(rng, π0)
    y = rand(rng, π0)

    Σ = kernel_cov(k)
    t = 1

    for _ in 1:L
        μ  = kernel_mean(k, ∇U, x)
        x  = rand(rng, MvNormal(μ, Σ))
        t += 1
    end

    is_coupled = false
    for _ in 1:tmax-L
        μ1, μ2 = kernel_mean(k, ∇U, x), kernel_mean(k, ∇U, y)

        x, y, is_coupled = rand_reflection_coupling(rng, μ1, μ2, Σ)

        if is_coupled
            return t
        end

        t += 1
    end
    error("couldn't couple within $(tmax) iterations")
end

function simulate_mixing_time(rng, π, π0, k, L, ϵ; visualize=false)
    ∇U(x) = ForwardDiff.gradient(x′ -> -logpdf(π, x′), x)

    ts = 1:1000

    samples = @showprogress map(1:128) do _
        τ = rand_lagged_coupling(rng, k, π0, ∇U, L)
        @. max(0, ceil((τ - L - ts)/L))
    end
    tv_bound = mean(samples)

    if visualize
        Plots.plot(mean(samples), ylims=(0, Inf)) |> display
        Plots.hline!([1.0]) |> display
    end
    findfirst(tv_bound .< ϵ)
end

function main()
    rng = Random.default_rng()

    ϵ = 0.5 
    L = 10^4
    d = 5 
    π = MvTDist(20.0, Fill(10.0, d), Matrix{Float64}(I, d, d) |> PDMats.PDMat)


    h_range = [0.7, 0.8, 0.9, 1.0]
    t_mixes = map(h_range) do h
        k  = ULA(d, h)
        π0 = MvNormal(Zeros(d), 1.0)

        simulate_mixing_time(rng, π, π0, k, L, ϵ; visualize=false)
    end
    Plots.plot(h_range, t_mixes) |> display

    # h  = 100.0
    # γ  = 50.0

    hγ_range = [(10, 10), (20, 20), (30, 30), (40, 40), (50,50), (50,50)]
    t_mixes = map(hγ_range) do (h, γ)
        k  = KLMC(d, h, γ)
        π0 = MvNormal(Zeros(2*d), 1.0)

        simulate_mixing_time(rng, π, π0, k, L, ϵ; visualize=false)
    end
    Plots.plot([last(hγ) for hγ in hγ_range], t_mixes) |> display
end
