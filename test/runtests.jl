
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
using PDMats

using ControlledSMC
using ControlledSMC: *, +

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

include("batchmvnormal.jl")
include("linalg.jl")
include("interface.jl")
include("adaptation.jl")
include("unbiasedness.jl")
