
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

        Σ_st = ControlledSMC.klmc_cov(d, h, γ)
        Σ_pd = PDMats.PDMat(Σ_st)

        μxs, μvs = randn(d, n), randn(d, n)
        μs       = vcat(μxs, μvs)
        μs_st    = ControlledSMC.BatchVectors2(μxs, μvs)
        p        = ControlledSMC.BatchMvNormal(μs_st, Σ_pd)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = σ2xx * Eye(d)
        Σ[1:d, (d + 1):end]         = σ2xv * Eye(d)
        Σ[(d + 1):end, 1:d]         = σ2xv * Eye(d)
        Σ[(d + 1):end, (d + 1):end] = σ2vv * Eye(d)
        p_true                      = MvNormal.(eachcol(μs), Ref(Σ))

        @testset "logpdf" begin
            xs, vs = σ2xx * randn(d, n), σ2vv * randn(d, n)
            zs_st  = ControlledSMC.BatchVectors2(xs, vs)
            zs     = vcat(xs, vs)

            logdensities_true = logpdf.(p_true, eachcol(zs))
            logdensities      = logpdf(p, zs_st)

            @test logdensities_true ≈ logdensities rtol = 0.0001
        end
    end

    @testset "diagonal entries" begin
        d = 3
        n = 4

        Σ11  = 1.0 * Diagonal(1:3)
        Σ21  = 0.1 * Diagonal(1:3)
        Σ22  = 2.0 * Diagonal(1:3)
        Σ_st = ControlledSMC.BlockHermitian2by2(Σ11, Σ21, Σ22)
        Σ_pd = PDMats.PDMat(Σ_st)

        μ1s, μ2s = randn(d, n), randn(d, n)
        μs       = vcat(μ1s, μ2s)
        μs_st    = ControlledSMC.BatchVectors2(μ1s, μ2s)
        p        = ControlledSMC.BatchMvNormal(μs_st, Σ_pd)

        Σ      = Matrix(Σ_st)
        p_true = MvNormal.(eachcol(μs), Ref(Σ))

        @testset "logpdf" begin
            x1, x2 = Σ11 * randn(d, n), Σ22 * randn(d, n)
            z      = vcat(x1, x2)
            z_st   = ControlledSMC.BatchVectors2(x1, x2)

            logdensities_true = logpdf.(p_true, eachcol(z))
            logdensities      = logpdf(p, z_st)

            @test logdensities_true ≈ logdensities rtol = 0.0001
        end
    end
end
