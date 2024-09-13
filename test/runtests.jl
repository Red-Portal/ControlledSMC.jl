
using ADTypes
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD, StanLogDensityProblems
using PosteriorDB
using Random
using ReverseDiff
using Test

using ControlledSMC

@testset "LogDensityProblems" begin
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

    d       = 3
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)

    n_iters  = 32
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters)
    path     = GeometricAnnealingPath(schedule, proposal, prob_ad)

    path = GeometricAnnealingPath(schedule, proposal, prob_ad)
    h0   = 0.5
    hT   = 0.5
    Γ    = Eye(d)

    n_particles = 2^10

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(h0, hT, TimeCorrectForwardKernel(), Γ, path),
        )
        ("SMCULA + ForwardKernel", SMCULA(h0, hT, ForwardKernel(), Γ, path))
        ("SMCULA + DetailedBalance", SMCULA(h0, hT, DetailedBalance(), Γ, path))
    ]
        ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
    end
end

@testset "StanLogDensityProblems" begin
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, ".stan/"; force=true)
    d    = LogDensityProblems.dimension(prob)

    n_particles = 2^10

    n_iters  = 32
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters)
    path     = GeometricAnnealingPath(schedule, proposal, prob)
    h0       = 0.0001
    hT       = 0.0001
    Γ        = Eye(d)

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(h0, hT, TimeCorrectForwardKernel(), Γ, path),
        )
        ("SMCULA + ForwardKernel", SMCULA(h0, hT, ForwardKernel(), Γ, path))
        ("SMCULA + DetailedBalance", SMCULA(h0, hT, DetailedBalance(), Γ, path))
    ]
        ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
    end
end
