
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
            SMCULA(0.5, n_iters, TimeCorrectForwardKernel(), Eye(d)),
        ),
        ("SMCULA + ForwardKernel", SMCULA(0.5, n_iters, ForwardKernel(), Eye(d))),
        ("SMCULA + DetailedBalance", SMCULA(0.5, n_iters, DetailedBalance(), Eye(d))),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, n_iters, Eye(d))),
        ("SMCKLMC", SMCKLMC(d, 5.0, 5.0, n_iters)),
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
            SMCULA(1e-4, n_iters, TimeCorrectForwardKernel(), Eye(d)),
        ),
        ("SMCULA + ForwardKernel", SMCULA(1e-4, n_iters, ForwardKernel(), Eye(d))),
        ("SMCULA + DetailedBalance", SMCULA(1e-4, n_iters, DetailedBalance(), Eye(d))),
        ("SMCUHMC", SMCUHMC(0.01, 0.5, n_iters, Eye(d))),
        ("SMCKLMC", SMCKLMC(d, 0.0001, 1000.0, n_iters)),
    ]
        ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
    end
end
