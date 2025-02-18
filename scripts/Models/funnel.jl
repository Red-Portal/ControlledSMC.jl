
struct Funnel
    d :: Int
end

function LogDensityProblems.logdensity(prob::Funnel, θ)
    d    = prob.d
    y    = θ[1]
    x    = θ[2:end]
    ℓp_y = logpdf(Normal(0, 3), y)
    ℓp_x = logpdf(MvNormal(Zeros(d-1), exp(y/2)), x)
    ℓp_x + ℓp_y
end

function LogDensityProblems.capabilities(::Type{<:Funnel})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::Funnel) = prob.d

