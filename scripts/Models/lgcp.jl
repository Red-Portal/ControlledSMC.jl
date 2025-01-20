
struct LogGaussianCoxProcess{Area <: Real, Counts <: AbstractVector{<:Int}, GPMean, GPCovChol}
    area        :: Area
    counts      :: Counts
    gp_mean     :: GPMean
    gp_cov_chol :: GPCovChol
end

function LogDensityProblems.logdensity(prob::LogGaussianCoxProcess, f_white)
    (; area, counts, gp_mean, gp_cov_chol) = prob
    f    = gp_cov_chol*f_white + gp_mean
    ℓp_f = logpdf(MvNormal(Zeros(length(f)), Ones(length(f))), f)
    ℓp_y = sum(@. f*counts - area*exp(f))
    ℓp_f + ℓp_y + sum(log.(diag(gp_cov_chol)))
end

function LogDensityProblems.capabilities(::Type{<:LogGaussianCoxProcess})
    return LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(prob::LogGaussianCoxProcess) = length(prob.gp_mean)

function LogGaussianCoxProcess()
    n_grid_points      = 20
    grid_length        = 1.0
    area               = grid_length/n_grid_points
    cell_boundaries_1d = range(0, grid_length; length=n_grid_points+1)

    data = readdlm(joinpath(@__DIR__, "datasets", "pines.csv"), ',', Float64, '\n', header=true)
    
    pine_coords   = data[1][:,2:3]
    pine_coords_x = pine_coords[:,1]
    pine_coords_y = pine_coords[:,2]

    hist            = fit(Histogram, (pine_coords_x, pine_coords_y), (cell_boundaries_1d, cell_boundaries_1d))
    counts_2d       = hist.weights
    grid_1d_unit    = Float64.(0:n_grid_points-1)
    grid_2d_unit    = Iterators.product(grid_1d_unit, grid_1d_unit) |> collect
    coordinates     = reshape(grid_2d_unit, :)
    coordinates_mat = hcat(
        [point[1] for point in coordinates],
        [point[2] for point in coordinates]
    )
    counts_1d       = reshape(counts_2d, :)

    σ2     = 1.91
    β      = 1/33
    μ0     = log(126) - σ2/2
    kernel = σ2 * KernelFunctions.compose(
        ExponentialKernel(), ScaleTransform(1/(n_grid_points*β))
    )

    K      = kernelmatrix(kernel, coordinates_mat, obsdim=1)
    K_chol = cholesky(K).L

    gp_mean = Fill(μ0, n_grid_points^2)

    #gp_prior = MvNormal(Fill(μ0, n_grid_points^2), PDMats.PDMat(K))

    LogGaussianCoxProcess{typeof(area), typeof(counts_1d), typeof(gp_mean), typeof(K_chol)}(
        area, counts_1d, gp_mean, K_chol
    )
end
