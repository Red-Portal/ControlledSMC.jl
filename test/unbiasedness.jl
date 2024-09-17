
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
            SMCULA(0.5, 0.5, TimeCorrectForwardKernel(), Eye(d), path),
        ),
        ("SMCULA + ForwardKernel", SMCULA(0.5, 0.5, ForwardKernel(), Eye(d), path)),
        ("SMCUHMC", SMCUHMC(1.0, 0.5, Eye(d))),
        ("SMCKLMC", SMCKLMC(5.0, 10.0)),
    ]
        pvalue = run_unbiasedness_test(sampler, path, n_particles, n_test_samples, Z_true)
        @test if pvalue > pvalue_threshold
            true
        else
            # Bonferroni-corrected second try with larger sample size
            pvalue2 = run_test(sampler, path, 4 * n_particles, Z_true)
            2 * pvalue2 > pvalue_threshold
        end
    end
end
