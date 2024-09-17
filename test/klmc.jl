
@testset "KLMC Cov" begin
    h = 0.1
    γ = 0.3
    η = exp(-γ * h)

    σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
    σ2xv = 1 / γ * (1 - 2 * η + η^2)
    σ2vv = 1 - η^2

    Σ_klmc_struct = ControlledSMC.klmc_cov(h, γ)

    Σ_true = Hermitian([σ2xx σ2xv; σ2xv σ2vv])
    L_true    = cholesky(Σ_true).L
    Linv_true = inv(L_true)

    lxx, lxv, lvv, linvxx, linvxv, linvvv = Σ_klmc_struct
    L_klmc = [lxx 0; lxv lvv]
    Linv_klmc = [linvxx 0; linvxv linvvv]

    @test L_klmc ≈ L_true rtol = 0.001
    @test Linv_klmc ≈ Linv_true rtol = 0.001
end
