
using ControlledSMC
using Distributions
using FillArrays
using LinearAlgebra
using LogDensityProblems, StanLogDensityProblems
using Plots
using PosteriorDB
using Random, Random123

function main()
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)
    set_counter!(rng, 1)

    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, ".stan/"; force=true)
    d    = LogDensityProblems.dimension(prob)

    n_iters = 32
    proposal = MvNormal(Zeros(d), I)
    schedule = range(0, 1; length=n_iters) .^ 2
    path = GeometricAnnealingPath(schedule, proposal, prob)
    n_particles = 512

    #h0       = 0.0001
    #hT       = 0.0001
    #Γ        = Eye(d)
    #sampler = SMCULA(h0, hT, TimeCorrectForwardKernel(), Γ, path)

    γ = 500.0
    h = 0.0001
    smc = SMCKLMC(d, γ*h, γ)
    csmc = CSMCKLMC(smc, path)
    _, _, states, stats = ControlledSMC.sample(
        rng, csmc, path, n_particles, 1.0; show_progress=true
    )

    for _ in 1:2
        csmc = optimize_policy(csmc, states)
        _, _, states, stats = ControlledSMC.sample(
            rng, csmc, path, n_particles, 1.0; show_progress=true
        )
    end
    return

    h       = 0.9
    δ       = 0.01
    M       = Eye(d)
    sampler = SMCUHMC(δ, h, M)

    xs, _, _, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    display(
        Plots.plot(
            [stat.ess for stat in stats]; ylims=[0, Inf], ylabel="ESS", xlabel="Iterations"
        ),
    )
    return stats
end
