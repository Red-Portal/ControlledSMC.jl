
@testset "Linear Algebra Routines" begin
    @testset "scalar entries" begin
        σ11, σ21, σ22 = 0.5, 0.2, 0.7
        Σ         = [σ11  σ21; σ21 σ22]
        Σ_struct  = ControlledSMC.BlockHermitian2by2(σ11, σ21, σ22)
        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L_struct = cholesky(Σ_struct)
            L        = [L_struct.L11 0; L_struct.L21 L_struct.L22]

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L_struct    = cholesky(Σ_struct)
            Linv_struct = inv(L_struct)
            Linv        = [Linv_struct.L11 0; Linv_struct.L21 Linv_struct.L22]

            @test Linv ≈ Linv_true rtol = 0.01
        end
    end

    @testset "diagonal matrix entries" begin
        d = 3

        Σ11      = 1.0 * Diagonal(1:d)
        Σ21      = 0.1 * Diagonal(1:d)
        Σ22      = 2.0 * Diagonal(1:d)
        Σ_struct = ControlledSMC.BlockHermitian2by2(Σ11, Σ21, Σ22)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = Σ11
        Σ[1:d, (d + 1):end]         = Σ21
        Σ[(d + 1):end, 1:d]         = Σ21
        Σ[(d + 1):end, (d + 1):end] = Σ22

        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L_struct                    = cholesky(Σ_struct)
            L                           = zeros(2 * d, 2 * d)
            L[1:d, 1:d]                 = L_struct.L11
            L[(d + 1):end, 1:d]         = L_struct.L21
            L[(d + 1):end, (d + 1):end] = L_struct.L22

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L_struct    = cholesky(Σ_struct)
            Linv_struct = inv(L_struct)

            Linv                           = zeros(2 * d, 2 * d)
            Linv[1:d, 1:d]                 = Linv_struct.L11
            Linv[(d + 1):end, 1:d]         = Linv_struct.L21
            Linv[(d + 1):end, (d + 1):end] = Linv_struct.L22

            @test Linv ≈ Linv_true rtol = 0.01
        end

    end

    @testset "algebraic operations" begin
        d = 3

        Σ11      = 1.0 * Diagonal(1:d)
        Σ21      = 0.1 * Diagonal(1:d)
        Σ22      = 2.0 * Diagonal(1:d)
        Σ_struct = ControlledSMC.BlockHermitian2by2(Σ11, Σ21, Σ22)

        A1       = 0.2*Diagonal(d:-1:1)
        A2       = 0.3*Diagonal(1:d)
        A_struct = ControlledSMC.BlockDiagonal2by2(A1, A2)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = Σ11
        Σ[1:d, (d + 1):end]         = Σ21
        Σ[(d + 1):end, 1:d]         = Σ21
        Σ[(d + 1):end, (d + 1):end] = Σ22

        A                  = Diagonal(zeros(2 * d))
        A[1:d,1:d]         = A1
        A[d+1:end,d+1:end] = A2

        Σ_post_true = inv(inv(Σ) + 2*A)
        

        L_true    = cholesky(0.5*I + sqrt(A)*Σ*sqrt(A)).L
        Linv_true = inv(L_true)
        C_sqrt_tr = Linv_true*sqrt(A)*Σ
        C         = C_sqrt_tr'*C_sqrt_tr

        # (Σ^{-1} + 2A)^{-1} = Σ - Σ √A (1/2 I + √A Σ √A)^{-1} √A Σ 
        # B ≜ 1/2 I + √A Σ √A
        # C ≜ L_{B}^{-1} √A Σ 
        half_eye    = ControlledSMC.BlockDiagonal2by2(
            Diagonal(fill(0.5, d)), Diagonal(fill(0.5, d))
        )
        A_sqrt        = sqrt(A_struct)
        AsqrtΣ        = A_sqrt*Σ_struct
        AsqrtΣAsqrt   = ControlledSMC.quad(Σ_struct, A_sqrt)
        B             = half_eye + AsqrtΣAsqrt
        B_chol        = cholesky(B)
        B_cholinv     = inv(B_chol)
        C_sqrt_tr     = B_cholinv*AsqrtΣ

        Σ_post_struct = Σ_struct - ControlledSMC.transpose_square(C_sqrt_tr)

        Σ_post = zeros(2 * d, 2 * d)
        Σ_post[1:d, 1:d]                 = Σ_post_struct.Σ11
        Σ_post[1:d, (d + 1):end]         = Σ_post_struct.Σ21
        Σ_post[(d + 1):end, 1:d]         = Σ_post_struct.Σ21
        Σ_post[(d + 1):end, (d + 1):end] = Σ_post_struct.Σ22

        @test Σ_post ≈ Σ_post_true rtol = 0.01
    end
end
