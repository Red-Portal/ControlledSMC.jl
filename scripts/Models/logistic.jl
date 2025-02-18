
struct LogisticRegression{XT, YT}
    X::XT
    y::YT
end

function LogDensityProblems.logdensity(prob::LogisticRegression, θ)
    (; X, y) = prob

    ℓp_θ   = mapreduce(normlogpdf, +, θ)

    logits = X*θ
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))

    ℓp_θ + ℓp_y
end

function LogDensityProblems.capabilities(::Type{<:LogisticRegression})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::LogisticRegression) = size(prob.X, 2)

function preprocess_features(X::AbstractMatrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .<= eps(Float64)] .= 1.0
    X = (X .- μ) ./ σ
    hcat(X, ones(size(X, 1), 1))
end

function LogisticRegressionSonar()
    data   = readdlm(joinpath(@__DIR__, "datasets", "sonar.csv"), ',', Any, '\n')
    X      = convert(Matrix{Float64}, data[:, 1:end-1])
    y      = data[:, end] .== "R"
    X_proc = preprocess_features(X)
    LogisticRegression(X_proc, y)
end
