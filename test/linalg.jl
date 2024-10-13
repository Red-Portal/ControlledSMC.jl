
@testset "Linear Algebra Routines" begin
    d = 3

    Σ11  = 1.0 * Diagonal(1:d)
    Σ21  = 0.1 * Diagonal(1:d)
    Σ22  = 2.0 * Diagonal(1:d)
    Σ_st = ControlledSMC.BlockHermitian2by2(Σ11, Σ21, Σ22)
    Σ_pd = PDMats.PDMat(Σ_st)
    Σ    = Matrix(Σ_st)

    L_true    = cholesky(Σ).L
    Linv_true = inv(L_true)

    @testset "cholesky" begin
        L_st = cholesky(Σ_st)
        L = Matrix(L_st)
        @test L ≈ L_true rtol = 0.01
    end

    @testset "cholesky inv" begin
        L_st = cholesky(Σ_st)
        Linv_st = inv(L_st)
        Linv = Matrix(Linv_st)
        @test Linv ≈ Linv_true rtol = 0.01
    end

    @testset "logdet" begin
        @test logdet(Σ) ≈ logdet(Σ_pd) rtol = 0.001
    end

    @testset "logdet" begin
        @test logdet(Σ) ≈ logdet(Σ_pd) rtol = 0.001
    end

    n    = 4
    x1   = randn(d, n)
    x2   = randn(d, n)
    x_st = ControlledSMC.BatchVectors2(x1, x2)
    x    = Matrix(x_st)

    @testset "system solve" begin
        Σinvx_true = Σ \ x
        Σinvx_st   = Σ_pd \ x_st
        Σinvx      = Matrix(Σinvx_st)

        @test Σinvx_true ≈ Σinvx rtol = 0.001
    end

    @testset "quad" begin
        @test quad(Σ, x) ≈ quad(Σ_pd, x_st)
    end

    @testset "invquad" begin
        @test invquad(Σ, x) ≈ invquad(Σ_pd, x_st)
    end

    @testset "posterior covariance" begin
        A1   = 0.2 * Diagonal(d:-1:1)
        A2   = 0.3 * Diagonal(1:d)
        A_st = ControlledSMC.BlockDiagonal2by2(A1, A2)
        A    = Matrix(A_st)

        K_true = inv(inv(Σ) + 2 * A)

        K_st = ControlledSMC.control_cov(A_st, Σ_pd)
        K = Matrix(K_st.Σ)
        @test K ≈ K_true rtol = 0.001
    end
end
