
module ControlledSMC

using ADTypes
using DifferentiationInterface
using Distributions
using FillArrays
using LogDensityProblems
using LogExpFunctions
using PDMats
using ProgressMeter
using Random
using ReverseDiff
using Statistics

abstract type AbstractSMC end

abstract type AbstractPath end

abstract type AbstractBackwardKernel end

include("weighing.jl")
include("mcmc.jl")

include("annealing.jl")

export GeometricAnnealing

include("backwardkernel.jl")

export
    DetailedBalance,
    ForwardKernel,
    TimeCorrectForwardKernel

include("sample.jl")

export sample

include("smc_ula.jl")

export SMCULA

end
