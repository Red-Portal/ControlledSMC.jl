
using ADTypes, Zygote, ForwardDiff, ReverseDiff
using Accessors
using ControlledSMC
using Distributions
using FillArrays
using LinearAlgebra
using Random
using Random123
using StatsFuns

struct LGSSM{Rho, SigmaX, SigmaY, Data} <: ControlledSMC.AbstractParticleFilter
    rho::Rho
    sigmax::SigmaX
    sigmay::SigmaY
    data::Vector{Data}
end

Base.length(pf::LGSSM) = length(pf.data)

function ControlledSMC.rand_initial_with_potential(
    rng::Random.AbstractRNG, model::LGSSM, n_particles::Int
)
    σx, ρ = model.sigmax, model.rho
    σx/sqrt(1 - ρ^2)*randn(rng, 1, n_particles), zeros(n_particles)
end

function ControlledSMC.potential(model::LGSSM, t::Int, x::AbstractMatrix)
    yt   = model.data[t]
    σy   = model.sigmay
    d, n = size(x)
    logpdf(ControlledSMC.BatchMvNormal(x, σy^2*Eye(d)), repeat(yt, 1, n))
end

function ControlledSMC.rand_transition(
    rng::Random.AbstractRNG,
    model::LGSSM,
    ::Int,
    x::AbstractMatrix
)
    ρ, σx = model.rho, model.sigmax
    rand(rng, ControlledSMC.BatchMvNormal(ρ*x, σx^2*Eye(1)) )
end

function ControlledSMC.logpdf_transition(model::LGSSM, ::Int, x::AbstractMatrix, x′::AbstractMatrix)
    ρ, σx  = model.rho, model.sigmax
    logpdf(ControlledSMC.BatchMvNormal(ρ*x, σx^2*Eye(1)), x′)
end

function create_data(rng, d, ρ, σx, σy, T)
    x0 = rand(rng, MvNormal(Zeros(d), σx/sqrt(1 - ρ^2)))
    x  = x0
    map(1:T) do t
        x′ = rand(rng, MvNormal(ρ*x, σx))
        y′ = rand(rng, MvNormal(x, σy))
        x = x′
        y′
    end
end

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 0)
    
    d  = 1
    ρ  = 0.9
    σx = 1.
    σy = 0.2
    T  = 100
    y  = create_data(rng, d, ρ, σx, σy, T)

    #hs      = fill(0.01, length(y))
    #fkmodel = LGSSM(ρ, σx, σy, y)
    #adtype  = AutoForwardDiff()
    #sampler = PFULA(adtype, fkmodel, hs)

    adaptor = BackwardKLMin(n_subsample=128, regularization=0.1)
    fkmodel = LGSSM(ρ, σx, σy, y)
    adtype  = AutoReverseDiff()
    sampler = PFULA(adtype, fkmodel, adaptor)

    n = 1000
    _, _, _, _, stats = ControlledSMC.sample(rng, sampler, n, 0.5; show_progress=true)
    stats
end
