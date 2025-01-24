
struct SMCUHMC{
    Stepsizes<:AbstractVector,
    Dampings<:AbstractVector,
    Mass<:AbstractMatrix,
    Adaptor<:AbstractAdaptor,
    DampingGrid<:AbstractVector,
} <: AbstractSMC
    stepsizes    :: Stepsizes
    dampings     :: Dampings
    mass_matrix  :: Mass
    adaptor      :: Adaptor
    damping_grid :: DampingGrid
end

function SMCUHMC(
    stepsize::Real,
    damping::Real,
    n_steps::Int,
    mass_matrix::AbstractMatrix,
    adaptor::AbstractAdaptor;
    damping_grid::AbstractVector=[0.1, 0.9],
)
    stepsizes = Fill(stepsize, n_steps)
    dampings  = Fill(damping, n_steps)
    return SMCUHMC{
        typeof(stepsizes),
        typeof(dampings),
        typeof(mass_matrix),
        typeof(adaptor),
        typeof(damping_grid),
    }(
        stepsizes, dampings, mass_matrix, adaptor, damping_grid
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
    (; stepsizes, dampings, mass_matrix) = sampler
    ϵ, α, M = stepsizes[t], dampings[t], mass_matrix
    sqrt1mα = sqrt(1 - α)

    d          = size(ztm1, 1) ÷ 2
    xtm1, vtm1 = ztm1[1:d, :], ztm1[(d + 1):end, :]
    v_dist     = MvNormal(Zeros(d), M)

    vthalf = sqrt1mα * vtm1 + sqrt(α) * unwhiten(M, randn(rng, size(vtm1)))
    xt, vt = leapfrog(πt, xtm1, vthalf, ϵ, M)

    ℓπt     = logdensity_safe(πt, xt)
    ℓπtm1   = logdensity_safe(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = BatchMvNormal(sqrt1mα * vthalf, α * M)
    K  = BatchMvNormal(sqrt1mα * vtm1, α * M)
    ℓl = logpdf(L, vtm1)
    ℓk = logpdf(K, vthalf)
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
end

function coordinate_descent_uhmc(
    obj, ρ, ℓh, ρ_grid;
    n_max_iters          = 32,
    ℓh_change_coeff      = 0.02,
    ℓh_change_ratio      = 1.5,
    abstol               = 1e-2,
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

        ℓh′, n_gss_iters = golden_section_search(
            obj_ℓh, ℓh_lower, ℓh_upper; abstol
        )
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
    return ρ, ℓh, (n_obj_evals=n_obj_evals, n_iters=n_iters,)
end

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCUHMC,
    t::Int,
    πt,
    πtm1,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if sampler.adaptor isa NoAdaptation
        return sampler, NamedTuple()
    end

    # Subsample particles to reduce adaptation overhead
    w_norm    = exp.(ℓwtm1 .- logsumexp(ℓwtm1))
    n_sub     = sampler.adaptor.n_subsample
    sub_idx   = systematic_sampling(rng, w_norm, n_sub)
    xtm1_sub  = xtm1[:, sub_idx]
    ℓdPdQ_sub = ℓwtm1[sub_idx]
    ℓwtm1_sub = fill(-log(n_sub), n_sub)

    ρ_grid = sampler.damping_grid

    if t == 2
        function obj_init(params)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(params[1])
            sampler′ = @set sampler′.dampings[t] = params[2]
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) +
                sampler.adaptor.regularization * abs2(params[1])
        end

        ℓh_lower_guess = -7.5
        ρ_guess        = 0.01

        obj_init_ℓh = ℓh′ -> obj_init([ℓh′, ρ_guess])

        ## Find any point that is not degenerate
        ℓh_decrease_stepsize = 1
        ℓh_lower, n_feasible_evals = find_feasible_point(
            obj_init_ℓh,
            ℓh_lower_guess,
            ℓh_decrease_stepsize,
            log(eps(Float64)),
        )

        ρ, ℓh, stats = coordinate_descent_uhmc(obj_init, ρ_guess, ℓh_lower, ρ_grid)

        # Consume rngs so that the actual mutation is less biased.
        rand(rng, size(xtm1))

        h     = exp(ℓh)
        stats = (
            feasible_search_objective_evaluations    = n_feasible_evals,
            coordinate_descent_iterations            = stats.n_iters,
            coordinate_descent_objective_evaluations = stats.n_obj_evals,
            uhmc_stepsize                            = h,
            uhmc_damping                             = ρ,
        )

        sampler = @set sampler.stepsizes[2] = h
        sampler = @set sampler.dampings[2] = ρ
        return sampler, stats
    else
        ℓh_prev = log(sampler.stepsizes[t - 1])
        ρ_prev  = sampler.dampings[t - 1]

        function obj(params)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(params[1])
            sampler′ = @set sampler′.dampings[t] = params[2]
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) +
                   sampler.adaptor.regularization * abs2(ℓh_prev - params[1])
        end

        ρ, ℓh, stats = coordinate_descent_uhmc(obj, ρ_prev, ℓh_prev, ρ_grid)

        # Consume rngs so that the actual mutation is less biased.
        rand(rng, size(xtm1))

        h       = exp(ℓh)
        sampler = @set sampler.stepsizes[t] = h
        sampler = @set sampler.dampings[t] = ρ
        stats   = (
            coordinate_descent_iterations            = stats.n_iters,
            coordinate_descent_objective_evaluations = stats.n_obj_evals,
            uhmc_stepsize                            = h,
            uhmc_damping                             = ρ,
        )
        return sampler, stats
    end
end
