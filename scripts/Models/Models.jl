
#module Models

using Base.Iterators
using DelimitedFiles
using Distributions
using FillArrays
using KernelFunctions
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using PDMats
using Statistics
using StatsBase
using StatsFuns

include("dist.jl")
include("funnel.jl")
include("logistic.jl")
include("lgcp.jl")
include("brownian.jl")

#end
