
@testset "Linear Algebra Routines" begin
    @testset "scalar entries" begin
        σ11, σ12, σ22 = 0.5, 0.2, 0.7
        Σ         = [σ11  σ12; σ12 σ22]
        Σ_struct  = ControlledSMC.BlockHermitian2by2(σ11, σ12, σ22)
        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L_struct = ControlledSMC.cholesky2by2(Σ_struct)
            L        = [L_struct.L11 0; L_struct.L12 L_struct.L22]

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L_struct    = ControlledSMC.cholesky2by2(Σ_struct)
            Linv_struct = ControlledSMC.inv2by2(L_struct)
            Linv        = [Linv_struct.L11 0; Linv_struct.L12 Linv_struct.L22]

            @test Linv ≈ Linv_true rtol = 0.01
        end
    end

    @testset "diagonal matrix entries" begin
        d = 3

        Σ11      = 1.0 * Diagonal(1:d)
        Σ12      = 0.1 * Diagonal(1:d)
        Σ22      = 2.0 * Diagonal(1:d)
        Σ_struct = ControlledSMC.BlockHermitian2by2(Σ11, Σ12, Σ22)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = Σ11
        Σ[1:d, (d + 1):end]         = Σ12
        Σ[(d + 1):end, 1:d]         = Σ12
        Σ[(d + 1):end, (d + 1):end] = Σ22

        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L_struct                    = ControlledSMC.cholesky2by2(Σ_struct)
            L                           = zeros(2 * d, 2 * d)
            L[1:d, 1:d]                 = L_struct.L11
            L[(d + 1):end, 1:d]         = L_struct.L12
            L[(d + 1):end, (d + 1):end] = L_struct.L22

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L_struct    = ControlledSMC.cholesky2by2(Σ_struct)
            Linv_struct = ControlledSMC.inv2by2(L_struct)

            Linv                           = zeros(2 * d, 2 * d)
            Linv[1:d, 1:d]                 = Linv_struct.L11
            Linv[(d + 1):end, 1:d]         = Linv_struct.L12
            Linv[(d + 1):end, (d + 1):end] = Linv_struct.L22

            @test Linv ≈ Linv_true rtol = 0.01
        end
    end
end
