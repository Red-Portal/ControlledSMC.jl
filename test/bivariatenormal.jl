
@testset "BivariateMvNormal" begin
    @testset "scalar entries" begin
        d = 3
        n = 4
        h = 0.1
        γ = 0.3
        η = exp(-γ * h)

        σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
        σ2xv = 1 / γ * (1 - 2 * η + η^2)
        σ2vv = 1 - η^2

        Σ_struct    = ControlledSMC.klmc_cov(h, γ)
        L_struct    = cholesky(Σ_struct)
        Linv_struct = inv(L_struct)

        μx, μv = randn(d, n), randn(d, n)
        p      = ControlledSMC.BivariateMvNormal(μx, μv, L_struct, Linv_struct)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = σ2xx * Eye(d)
        Σ[1:d, (d + 1):end]         = σ2xv * Eye(d)
        Σ[(d + 1):end, 1:d]         = σ2xv * Eye(d)
        Σ[(d + 1):end, (d + 1):end] = σ2vv * Eye(d)
        p_true                      = MvNormal.(eachcol(vcat(μx, μv)), Ref(Σ))

        @testset "logpdf" begin
            x, v = σ2xx * randn(d, n), σ2vv * randn(d, n)
            z    = vcat(x, v)

            logdensities_true = logpdf.(p_true, eachcol(z))
            logdensities      = ControlledSMC.bivariate_logpdf(p, x, v)

            @test logdensities_true ≈ logdensities rtol = 0.0001
        end
    end

    @testset "diagonal entries" begin
        d = 3
        n = 4

        Σ11  = 1.0 * Diagonal(1:3)
        Σ21  = 0.1 * Diagonal(1:3)
        Σ22  = 2.0 * Diagonal(1:3)
        Σ    = ControlledSMC.BlockHermitian2by2(Σ11, Σ21, Σ22)
        L    = cholesky(Σ)
        Linv = inv(L)

        μ1, μ2 = randn(d, n), randn(d, n)
        p      = ControlledSMC.BivariateMvNormal(μ1, μ2, L, Linv)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = Σ11
        Σ[1:d, (d + 1):end]         = Σ21
        Σ[(d + 1):end, 1:d]         = Σ21
        Σ[(d + 1):end, (d + 1):end] = Σ22
        p_true                      = MvNormal.(eachcol(vcat(μ1, μ2)), Ref(Σ))

        @testset "logpdf" begin
            x1, x2 = Σ11 * randn(d, n), Σ22 * randn(d, n)
            z      = vcat(x1, x2)

            logdensities_true = logpdf.(p_true, eachcol(z))
            logdensities      = ControlledSMC.bivariate_logpdf(p, x1, x2)

            @test logdensities_true ≈ logdensities rtol = 0.0001
        end
    end
end
