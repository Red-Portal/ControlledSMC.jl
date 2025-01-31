
function run_unbiasedness_test(sampler, path, n_particles, n_test_samples, Z_true)
    ℓZ  = map(1:n_test_samples) do _
        _, _, _, stats = ControlledSMC.sample(sampler, path, n_particles, 0.5; show_progress=false)
        last(stats).log_normalizer
    end
    Z   = exp.(ℓZ)
    res = tTest1S(Z; refmean=Z_true, verbose=false)
    return res.p
end

@testset "unbiasedness" begin
    pvalue_threshold = 0.1
    n_test_samples   = 128

    # Problem Setup
    d       = 5
    μ       = Fill(10, d)
    prob    = Dist(MvNormal(μ, I))
    prob_ad = ADgradient(AutoReverseDiff(), prob)
    Z_true  = 1.0

    # Sampler Setup
    n_iters     = 64
    proposal    = MvNormal(Zeros(d), I)
    schedule    = range(0, 1; length=n_iters)
    path        = GeometricAnnealingPath(schedule, proposal, prob_ad)
    n_particles = 256

    @testset "$(name)" for (name, sampler) in [
        (
            "SMCULA + TimeCorrectForwardKernel",
            SMCULA(0.5, n_iters, TimeCorrectForwardKernel(), Eye(d), path),
        ),
        ("SMCULA + ForwardKernel", SMCULA(0.5, n_iters, ForwardKernel(), Eye(d))),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, n_iters, Eye(d))),
        ("SMCKLMC", SMCKLMC(d, 5.0, 10.0, n_iters)),
    ]
        pvalue = run_unbiasedness_test(sampler, path, n_particles, n_test_samples, Z_true)

        # Bonferroni-correct the pvalue since we may have a second try
        @test if 2 * pvalue > pvalue_threshold
            true
        else
            pvalue2 = run_unbiasedness_test(sampler, path, 4 * n_particles, Z_true)
            pvalue2 > pvalue_threshold
        end
    end
end
