
using ADTypes
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD, StanLogDensityProblems
using PermutationTests
using PosteriorDB
using Random
using ReverseDiff
using Test

using ControlledSMC

struct Dist{D}
    dist::D
end

function LogDensityProblems.capabilities(::Type{<:Dist})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::Dist) = length(prob.dist)

function LogDensityProblems.logdensity(prob::Dist, x)
    return logpdf(prob.dist, x)
end

@testset "LogDensityProblems" begin
    # Problem Setup
    d       = 3
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    # Sampler Setup
    n_iters     = 32
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters)
    path        = GeometricAnnealingPath(schedule, proposal, prob_ad)
    n_particles = 2^10

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(0.5, 0.5, TimeCorrectForwardKernel(), Eye(d), path),
        ),
        ("SMCULA + ForwardKernel", SMCULA(0.5, 0.5, ForwardKernel(), Eye(d), path)),
        ("SMCULA + DetailedBalance", SMCULA(0.5, 0.5, DetailedBalance(), Eye(d), path)),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, Eye(d))),
    ]
        ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
    end
end

@testset "StanLogDensityProblems" begin
    # Problem Setup
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, ".stan/"; force=true)
    d    = LogDensityProblems.dimension(prob)

    # Sampler Setup
    n_particles = 2^10
    n_iters     = 32
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters)
    path        = GeometricAnnealingPath(schedule, proposal, prob)

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(1e-4, 1e-4, TimeCorrectForwardKernel(), Eye(d), path),
        ),
        ("SMCULA + ForwardKernel", SMCULA(1e-4, 1e-4, ForwardKernel(), Eye(d), path)),
        ("SMCULA + DetailedBalance", SMCULA(1e-4, 1e-4, DetailedBalance(), Eye(d), path)),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, Eye(d))),
    ]
        ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
    end
end

@testset "unbiasedness" begin
    pvalue_threshold = 0.01
    n_test_samples   = 128

    # Problem Setup
    d       = 5
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)
    Z_true  = 1.0

    # Sampler Setup
    n_iters     = 64
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters)
    path        = GeometricAnnealingPath(schedule, proposal, prob_ad)
    n_particles = 256

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(0.5, 0.5, TimeCorrectForwardKernel(), Eye(d), path),
        ),
        ("SMCULA + ForwardKernel", SMCULA(0.5, 0.5, ForwardKernel(), Eye(d), path)),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, Eye(d))),
    ]
        ℓZ  = map(1:n_test_samples) do _
            xs, _, _, stats = ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
            last(stats).log_normalizer
        end
        Z   = exp.(ℓZ)
        res = tTest1S(Z; refmean=Z_true, verbose=false)

        @test res.p > pvalue_threshold
    end
end
