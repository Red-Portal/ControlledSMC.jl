
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
            SMCULA(path, 0.5; backward=TimeCorrectForwardKernel(), precond=Eye(d)),
        ),
        (
            "SMCULA + ForwardKernel",
            SMCULA(path, 0.5; backward=ForwardKernel(), precond=Eye(d)),
        ),
        (
            "SMCULA + DetailedBalance",
            SMCULA(path, 0.5; backward=DetailedBalance(), precond=Eye(d)),
        ),
        ("Adaptive SMCULA", SMCULA(path, BackwardKLMin(; n_subsample=32))),
        ("SMCUHMC", SMCUHMC(path, 1.0, 0.5; mass_matrix=Eye(d))),
        ("SMCUHMC", SMCUHMC(path, BackwardKLMin(; n_subsample=32); mass_matrix=Eye(d))),
        ("SMCMALA", SMCMALA(path, AcceptanceRateControl(n_subsample=32, target_acceptance_rate=0.5, regularization=0.1); precond=Eye(d))),
        #("SMCKLMC", SMCKLMC(d, 5.0, 5.0, n_iters)),
    ]
        ControlledSMC.sample(sampler, n_particles, 0.5; show_progress=false)
    end
end

@testset "StanLogDensityProblems" begin
    # Problem Setup
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, "dogs-dogs")
    prob = StanProblem(post, ".stan/"; force=true, nan_on_error=true)
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
            SMCULA(path, 1e-4; backward=TimeCorrectForwardKernel(), precond=Eye(d)),
        ),
        (
            "SMCULA + ForwardKernel",
            SMCULA(path, 1e-4; backward=ForwardKernel(), precond=Eye(d)),
        ),
        (
            "SMCULA + DetailedBalance",
            SMCULA(path, 1e-4; backward=DetailedBalance(), precond=Eye(d)),
        ),
        ("Adaptive SMCULA", SMCULA(path, BackwardKLMin(; n_subsample=32))),
        ("SMCUHMC", SMCUHMC(path, 0.01, 0.5; mass_matrix=Eye(d))),
        ("SMCUHMC", SMCUHMC(path, BackwardKLMin(n_subsample=32, regularization=10.0); mass_matrix=Eye(d))),
        ("SMCMALA", SMCMALA(path, AcceptanceRateControl(n_subsample=32, target_acceptance_rate=0.5, regularization=0.1); precond=Eye(d))),
        #("SMCKLMC", SMCKLMC(d, 0.0001, 1000.0, n_iters)),
    ]
        ControlledSMC.sample(sampler, n_particles, 0.5; show_progress=false)
    end
end
