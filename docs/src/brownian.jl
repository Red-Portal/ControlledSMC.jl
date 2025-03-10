
using LogDensityProblems
using Distributions
using Statistics
using StatsFuns
using LogExpFunctions

struct BrownianMotion{Y, Idx}
    y       :: Y
    obs_idx :: Idx
end

function LogDensityProblems.logdensity(prob::BrownianMotion, θ)
    (; y, obs_idx) = prob
    x     = θ[1:30]
    α_inn = softplus(θ[31])
    α_obs = softplus(θ[32])

    ℓjac_α_inn = loglogistic(θ[31])
    ℓjac_α_obs = loglogistic(θ[32])

    ℓp_α_inn = logpdf(LogNormal(0, 2), α_inn)
    ℓp_α_obs = logpdf(LogNormal(0, 2), α_obs)
    ℓp_x1    = logpdf(Normal(0, α_inn), x[1])
    ℓp_x     = logpdf(MvNormal(x[1:end-1], α_inn), x[2:end])
    ℓp_y     = logpdf(MvNormal(x[obs_idx], α_obs), y)

    ℓp_y + ℓp_x1 + ℓp_x + ℓp_α_inn + ℓp_α_obs + ℓjac_α_inn + ℓjac_α_obs
end

function LogDensityProblems.capabilities(::Type{<:BrownianMotion})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::BrownianMotion) = 32

function BrownianMotion()
    y = [
        0.21592641,
        0.118771404,
        -0.07945447,
        0.037677474,
        -0.27885845,
        -0.1484156,
        -0.3250906,
        -0.22957903,
        -0.44110894,
        -0.09830782,       
        #
        -0.8786016,
        -0.83736074,
        -0.7384849,
        -0.8939254,
        -0.7774566,
        -0.70238715,
        -0.87771565,
        -0.51853573,
        -0.6948214,
        -0.6202789,
    ]
    obs_idx = vcat(1:10, 21:30)
    BrownianMotion(y, obs_idx)
end
