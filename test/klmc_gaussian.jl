
@testset "KLMC Gaussian" begin
    h = 0.1
    γ = 0.3
    η = exp(-γ * h)

    σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
    σ2xv = 1 / γ * (1 - 2 * η + η^2)
    σ2vv = 1 - η^2

    Σ_klmc_struct = ControlledSMC.klmc_cov(h, γ)

    @testset begin
        "Cholesky"
        Σ_true = Hermitian([σ2xx σ2xv; σ2xv σ2vv])
        L_true    = cholesky(Σ_true).L
        Linv_true = inv(L_true)

        (; lxx, lxv, lvv, linvxx, linvxv, linvvv) = Σ_klmc_struct
        L_klmc = [lxx 0; lxv lvv]
        Linv_klmc = [linvxx 0; linvxv linvvv]

        @test L_klmc ≈ L_true rtol = 0.001
        @test Linv_klmc ≈ Linv_true rtol = 0.001
    end

    @testset begin
        "logpdf"
        d = 3
        μx = randn(d, 4)
        μv = randn(d, 4)
        μ  = vcat(μx, μv)

        Σ_true                           = zeros(2 * d, 2 * d)
        Σ_true[1:d, 1:d]                 = σ2xx * Eye(d)
        Σ_true[1:d, (d + 1):end]         = σ2xv * Eye(d)
        Σ_true[(d + 1):end, 1:d]         = σ2xv * Eye(d)
        Σ_true[(d + 1):end, (d + 1):end] = σ2vv * Eye(d)
        ks_ref                           = MvNormal.(eachcol(μ), Ref(Σ_true))

        k = ControlledSMC.KLMCKernelBatch(μx, μv, Σ_klmc_struct)

        x                 = randn(d, 4)
        v                 = randn(d, 4)
        z                 = vcat(x, v)
        logdensities_true = logpdf.(ks_ref, eachcol(z))
        logdensities      = ControlledSMC.klmc_logpdf_batch(k, x, v)

        @test logdensities_true ≈ logdensities rtol = 0.001
    end
end
