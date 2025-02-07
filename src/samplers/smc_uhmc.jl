
struct SMCUHMC{
    Path<:AbstractPath,
    Stepsizes<:AbstractVector,
    RefreshRates<:AbstractVector,
    Mass<:Union{<:AbstractMatrix,<:UniformScaling},
    Adaptor<:Union{<:AbstractAdaptor,Nothing},
    RefreshRateGrid<:Union{<:AbstractVector,Nothing},
} <: AbstractSMC
    path              :: Path
    stepsizes         :: Stepsizes
    refresh_rates     :: RefreshRates
    mass_matrix       :: Mass
    adaptor           :: Adaptor
    refresh_rate_grid :: RefreshRateGrid
end

function SMCUHMC(
    path::AbstractPath,
    adaptor::AbstractAdaptor;
    mass_matrix::Union{<:AbstractMatrix,<:UniformScaling}=I,
    refresh_rate_grid::AbstractVector=[0.1, 0.9],
)
    stepsizes     = zeros(Float64, length(path))
    refresh_rates = zeros(Float64, length(path))
    return SMCUHMC{
        typeof(path),
        typeof(stepsizes),
        typeof(refresh_rates),
        typeof(mass_matrix),
        typeof(adaptor),
        typeof(refresh_rate_grid),
    }(
        path, stepsizes, refresh_rates, mass_matrix, adaptor, refresh_rate_grid
    )
end

function SMCUHMC(
    path::AbstractPath,
    stepsize::Union{<:AbstractVector,<:Real},
    refresh_rate::Union{<:AbstractVector,<:Real};
    mass_matrix::Union{<:AbstractMatrix,<:UniformScaling}=I,
)
    if stepsize isa Real
        stepsize = fill(stepsize, length(path))
    end
    if refresh_rate isa Real
        refresh_rate = fill(refresh_rate, length(path))
    end
    @assert length(stepsize) == length(path)
    @assert length(refresh_rate) == length(path)
    @assert all(@. 0 < stepsize)
    @assert all(@. 0 < refresh_rate ≤ 1)
    return SMCUHMC{
        typeof(path),
        typeof(stepsize),
        typeof(refresh_rate),
        typeof(mass_matrix),
        Nothing,
        Nothing,
    }(
        path, stepsize, refresh_rate, mass_matrix, nothing, nothing
    )
end

function rand_initial_with_potential(
    rng::Random.AbstractRNG, sampler::SMCUHMC, path::AbstractPath, n_particles::Int
)
    (; mass_matrix,) = sampler
    (; proposal,) = path

    x      = rand(rng, proposal, n_particles)
    n_dims = size(x, 1)
    v      = rand(rng, MvNormal(Zeros(n_dims), mass_matrix), n_particles)
    ℓG     = zeros(n_particles)
    return vcat(x, v), ℓG
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCUHMC, t::Int, πt, πtm1, ztm1::AbstractMatrix
)
    (; stepsizes, refresh_rates, mass_matrix) = sampler
    d = size(ztm1, 1) ÷ 2
    ϵ, ρ = stepsizes[t], refresh_rates[t]
    M = (mass_matrix isa UniformScaling) ? mass_matrix(d) : mass_matrix

    xtm1, vtm1 = view(ztm1, 1:d, :), view(ztm1, (d + 1):(2 * d), :)
    v_dist     = MvNormal(Zeros(d), M)

    vthalf = sqrt(1 - ρ^2) * vtm1 + ρ * unwhiten(M, randn(rng, size(vtm1)))
    xt, vt = leapfrog(πt, xtm1, vthalf, ϵ, M)

    ℓπt     = logdensity_safe(πt, xt)
    ℓπtm1   = logdensity_safe(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = BatchMvNormal(sqrt(1 - ρ^2) * vthalf, ρ^2 * M)
    K  = BatchMvNormal(sqrt(1 - ρ^2) * vtm1, ρ^2 * M)
    ℓl = logpdf(L, vtm1)
    ℓk = logpdf(K, vthalf)
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end

function coordinate_descent_uhmc(
    obj,
    ρ,
    ℓh,
    ρ_grid;
    n_max_iters     = 32,
    ℓh_change_coeff = 0.02,
    ℓh_change_ratio = 1.5,
    abstol          = 1e-2,
)
    n_obj_evals     = 0
    coordesc_iters  = 0
    coordesc_abstol = 1e-2
    n_iters         = 1

    while true
        obj_ℓh = ℓh′ -> obj([ℓh′, ρ])
        ℓh_upper, n_upper_bound_evals = find_golden_section_search_interval(
            obj_ℓh, ℓh, ℓh_change_coeff, ℓh_change_ratio
        )
        ℓh_lower, n_lower_bound_evals = find_golden_section_search_interval(
            obj_ℓh, ℓh_upper, -ℓh_change_coeff, ℓh_change_ratio
        )
        n_obj_evals += n_lower_bound_evals + n_upper_bound_evals

        ℓh′, n_gss_iters = golden_section_search(obj_ℓh, ℓh_lower, ℓh_upper; abstol)
        n_obj_evals += 2 * n_gss_iters

        ρ′ = argmin(ρ′ -> obj([ℓh′, ρ′]), ρ_grid)
        n_obj_evals += length(ρ_grid)

        if max(abs(ℓh′ - ℓh), abs(ρ′ - ρ)) < coordesc_abstol ||
            coordesc_iters == n_max_iters
            ρ  = ρ′
            ℓh = ℓh′
            break
        end

        n_iters += 1

        ρ  = ρ′
        ℓh = ℓh′
    end
    return ρ, ℓh, (n_obj_evals=n_obj_evals, n_iters=n_iters)
end

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCUHMC,
    t::Int,
    πt,
    πtm1,
    ztm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if isnothing(sampler.adaptor)
        return sampler, NamedTuple()
    end

    # Subsample particles to reduce adaptation overhead
    w_norm    = exp.(ℓwtm1 .- logsumexp(ℓwtm1))
    n_sub     = sampler.adaptor.n_subsample
    sub_idx   = systematic_sampling(rng, w_norm, n_sub)
    ztm1_sub  = ztm1[:, sub_idx]
    ℓdPdQ_sub = ℓwtm1[sub_idx]
    ℓwtm1_sub = fill(-log(n_sub), n_sub)

    ρ_grid = sampler.refresh_rate_grid

    function obj(ℓh_, ρ_)
        rng_fixed    = copy(rng)
        sampler′     = @set sampler.stepsizes[t] = exp(ℓh_)
        sampler′     = @set sampler′.refresh_rates[t] = ρ_
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, ztm1_sub)
        τ            = sampler.adaptor.regularization
        reg          = if t == 1
            τ * abs2(ℓh_)
        else
            ℓh_prev = log(sampler.stepsizes[t - 1])
            τ * abs2(ℓh_prev - ℓh_)
        end
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) + reg
    end

    r = 2.0
    c = 0.01
    ϵ = 1e-2
    δ = -1
    ℓh_guess = -7.5
    ρ_guess = 0.1
    n_evals_total = 0

    ℓh, ρ = if t == 1
        ℓh, n_evals = find_feasible_point(
            ℓh_ -> obj(ℓh_, ρ_guess), ℓh_guess, δ, log(eps(eltype(ztm1)))
        )
        n_evals_total += n_evals
        ℓh, ρ_guess
    else
        log(sampler.stepsizes[t - 1]), sampler.refresh_rates[t - 1]
    end

    n_max_iters = 10
    i           = 1
    while true
        ℓh′, n_evals = minimize(ℓh_ -> obj(ℓh_, ρ), ℓh, c, r, ϵ)
        n_evals_total += n_evals

        ρ′ = argmin(ρ_ -> obj(ℓh′, ρ_), ρ_grid)
        n_evals_total += length(ρ_grid)

        if max(abs(ℓh′ - ℓh), abs(ρ′ - ρ)) < ϵ || i == n_max_iters
            ρ  = ρ′
            ℓh = ℓh′
            break
        end

        i  += 1
        ρ  = ρ′
        ℓh = ℓh′
    end

    # Consume rng states so that the actual mutation is less biased.
    rand(rng, size(ztm1))

    h     = exp(ℓh)
    stats = (coordinate_descent_iterations=i, n_objective_evals=n_evals_total, uhmc_stepsize=h, uhmc_refresh_rate=ρ)

    sampler = @set sampler.stepsizes[t] = h
    sampler = @set sampler.refresh_rates[t] = ρ
    return sampler, stats
end
