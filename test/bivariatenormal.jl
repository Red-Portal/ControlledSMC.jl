
@testset "BivariateMvNormal" begin
    d = 3
    n = 4
    h = 0.1
    γ = 0.3
    η = exp(-γ * h)

    σ2xx = 2 / γ * (h - 2 / γ * (1 - η) + 1 / (2 * γ) * (1 - η^2))
    σ2xv = 1 / γ * (1 - 2 * η + η^2)
    σ2vv = 1 - η^2

    Σ_klmc_struct = ControlledSMC.klmc_cov(h, γ)

    μx, μv = randn(d, n), randn(d, n)
    p      = ControlledSMC.BivariateMvNormal(μx, μv, Σ_klmc_struct...)

    Σ                           = zeros(2 * d, 2 * d)
    Σ[1:d, 1:d]                 = σ2xx * Eye(d)
    Σ[1:d, (d + 1):end]         = σ2xv * Eye(d)
    Σ[(d + 1):end, 1:d]         = σ2xv * Eye(d)
    Σ[(d + 1):end, (d + 1):end] = σ2vv * Eye(d)
    p_true = MvNormal.(eachcol(vcat(μx, μv)), Ref(Σ))

    @testset "logpdf" begin 
        x, v = σ2xx*randn(d, n), σ2vv*randn(d, n)
        z    = vcat(x, v)

        logdensities_true = logpdf.(p_true, eachcol(z))
        logdensities      = ControlledSMC.bivariate_logpdf(p, x, v)

        @test logdensities_true ≈ logdensities rtol = 0.001
    end
end
