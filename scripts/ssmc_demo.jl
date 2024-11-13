
using Accessors
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
    post = PosteriorDB.posterior(pdb, "dogs-dogs_hierarchical")
    prob = StanProblem(post, ".stan/"; force=true)
    d    = LogDensityProblems.dimension(prob)

    n_iters     = 32
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters).^2
    path        = GeometricAnnealingPath(schedule, proposal, prob)
    n_particles = 512

    h = 0.0001
    Γ = Eye(d)
    
    sampler = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ)
    _, _, states, stats = ControlledSMC.sample(
        rng, sampler, path, n_particles, 0.5; show_progress=true
    )
    #Plots.plot([stat.ess for stat in stats]) |> display
    #Plots.plot(schedule) |> display
    Plots.plot() |> display

    for i in 1:3
        schedule, local_barrier, _ = ControlledSMC.update_schedule(schedule, stats, n_iters)

        path          = @set path.schedule = schedule
        sampler       = SMCULA(h, n_iters, TimeCorrectForwardKernel(), Γ)
        _, _, states, stats = ControlledSMC.sample(
            rng, sampler, path, n_particles, 0.5; show_progress=true
        )

        #println(sum([stat.resampled for stat in stats]))
        Plots.plot!(schedule[2:end], local_barrier[2:end], xscale=:log10) |> display
        #Plots.plot!(schedule) |> display
        #Plots.plot!([stat.ess for stat in stats]) |> display
    end
end
