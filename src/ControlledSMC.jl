
module ControlledSMC

using ADTypes
using Accessors
using DifferentiationInterface
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using Optim
using PDMats
using ProgressMeter
using Random
using ReverseDiff
using Statistics
using UnicodePlots

abstract type AbstractSMC end

abstract type AbstractPath end

abstract type AbstractBackwardKernel end

include("weighing.jl")
include("mcmc.jl")
include("utils.jl")
include("linalg.jl")
include("bivariatenormal.jl")

# Target Paths
include("paths/annealing.jl")

include("paths/geometric_annealing.jl")
export GeometricAnnealingPath

include("paths/adaptive_geometric_annealing.jl")
export AdaptiveGeometricAnnealing

# Backward Kernels
include("backwardkernel.jl")

export DetailedBalance, ForwardKernel, TimeCorrectForwardKernel

include("sample.jl")

export sample

include("samplers/smc_ula.jl")
export SMCULA

include("samplers/smc_uhmc.jl")
export SMCUHMC

include("samplers/smc_klmc.jl")
export SMCKLMC

abstract type AbstractControlledSMC <: AbstractSMC end

include("twist.jl")
include("adp.jl")
export optimize_policy

include("samplers/csmc_ula.jl")
export CSMCULA

end
