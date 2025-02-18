
@testset "minimize" begin
    x_opt = 2
    x_inf = 3
    f(x) =
        if x > x_inf
            Inf
        else
            (x - x_opt)^2
        end

    c  = 0.5
    r  = 1.5
    x0 = -3
    ϵ  = 1e-2

    x_sol, _ = ControlledSMC.minimize(f, x0, c, r, ϵ)

    @test abs(x_sol - x_opt) < 1e-2
end
