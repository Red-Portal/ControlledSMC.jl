
module ControlledSMC

using ADTypes
using Accessors
using DifferentiationInterface
using Distributions
using FillArrays
using DataInterpolations
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

include("linalg/types.jl")
include("linalg/blas1.jl")
include("linalg/blas2.jl")
include("linalg/blas3.jl")
include("linalg/pdmat.jl")

include("weighing.jl")
include("mcmc.jl")
include("utils.jl")
include("batchmvnormal.jl")
include("logdensity.jl")

include("adaptation.jl")
export
    NoAdaptation,
    ForwardKLMin,
    BackwardKLMin,
    PartialForwardKLMin,
    PartialBackwardKLMin,
    AnnealedFlowTransport

# Target Paths
include("paths/annealing.jl")

include("paths/geometric_annealing.jl")
export GeometricAnnealingPath

include("paths/adaptive_geometric_annealing.jl")
export AdaptiveGeometricAnnealing

# Vanilla SMC Samplers
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

# Optimized Annlealed SMC
include("paths/update_schedule.jl")

# Controlled SMC
abstract type AbstractControlledSMC <: AbstractSMC end

include("control/twist.jl")
include("control/adp.jl")
export optimize_policy

include("samplers/csmc_ula.jl")
export CSMCULA

include("samplers/csmc_klmc.jl")
export CSMCKLMC

end
