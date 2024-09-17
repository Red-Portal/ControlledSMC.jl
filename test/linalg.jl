
@testset "Linear Algebra Routines" begin
    @testset "scalar entries" begin
        σ11, σ12, σ22 = 0.5, 0.2, 0.7
        Σ = [σ11  σ12; σ12 σ22]

        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L11, L12, L22 = ControlledSMC.cholesky2by2(σ11, σ12, σ22)
            L = [L11 0; L12 L22]

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L11, L12, L22          = ControlledSMC.cholesky2by2(σ11, σ12, σ22)
            Linv11, Linv12, Linv22 = ControlledSMC.inv2by2(L11, L12, L22)
            Linv                   = [Linv11 0; Linv12 Linv22]

            @test Linv ≈ Linv_true rtol = 0.01
        end
    end

    @testset "diagonal matrix entries" begin
        d = 3

        Σ11 = Diagonal(1:d)
        Σ12 = 0.1 * Diagonal((1:d))
        Σ22 = 2 * Diagonal(1:d)

        Σ                           = zeros(2 * d, 2 * d)
        Σ[1:d, 1:d]                 = Σ11
        Σ[1:d, (d + 1):end]         = Σ12
        Σ[(d + 1):end, 1:d]         = Σ12
        Σ[(d + 1):end, (d + 1):end] = Σ22

        L_true    = cholesky(Σ).L
        Linv_true = inv(L_true)

        @testset "cholesky" begin
            L11, L12, L22               = ControlledSMC.cholesky2by2(Σ11, Σ12, Σ22)
            L                           = zeros(2 * d, 2 * d)
            L[1:d, 1:d]                 = L11
            L[(d + 1):end, 1:d]         = L12
            L[(d + 1):end, (d + 1):end] = L22

            @test L ≈ L_true rtol = 0.01
        end

        @testset "cholesky inv" begin
            L11, L12, L22          = ControlledSMC.cholesky2by2(Σ11, Σ12, Σ22)
            Linv11, Linv12, Linv22 = ControlledSMC.inv2by2(L11, L12, L22)

            Linv                           = zeros(2 * d, 2 * d)
            Linv[1:d, 1:d]                 = Linv11
            Linv[(d + 1):end, 1:d]         = Linv12
            Linv[(d + 1):end, (d + 1):end] = Linv22

            @test Linv ≈ Linv_true rtol = 0.01
        end
    end
end
