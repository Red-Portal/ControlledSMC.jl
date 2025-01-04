
struct SMCUHMC{
    Stepsizes<:AbstractVector,
    Dampings<:AbstractVector,
    Mass<:AbstractMatrix,
    Adaptor<:AbstractAdaptor,
} <: AbstractSMC
    stepsizes   :: Stepsizes
    dampings    :: Dampings
    mass_matrix :: Mass
    adaptor     :: Adaptor
end

function SMCUHMC(
    stepsize::Real,
    damping::Real,
    n_steps::Int,
    mass_matrix::AbstractMatrix,
    adaptor::AbstractAdaptor,
)
    stepsizes = Fill(stepsize, n_steps)
    dampings  = Fill(damping, n_steps)
    return SMCUHMC{typeof(stepsizes),typeof(dampings),typeof(mass_matrix),typeof(adaptor)}(
        stepsizes, dampings, mass_matrix, adaptor
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

    ℓπt     = LogDensityProblems.logdensity(πt, xt)
    ℓπtm1   = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓauxt   = logpdf.(Ref(v_dist), eachcol(vt))
    ℓauxtm1 = logpdf.(Ref(v_dist), eachcol(vtm1))

    L  = MvNormal.(sqrt1mα * eachcol(vthalf), Ref(α * M))
    K  = MvNormal.(sqrt1mα * eachcol(vtm1), Ref(α * M))
    ℓl = logpdf.(L, eachcol(vtm1))
    ℓk = logpdf.(K, eachcol(vthalf))
    ℓG = ℓπt + ℓauxt - ℓπtm1 - ℓauxtm1 + ℓl - ℓk
    return vcat(xt, vt), ℓG, NamedTuple()
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
    n_particles = size(xtm1, 2)
    idx_sub     = StatsBase.sample(rng, 1:n_particles, sampler.adaptor.n_subsample; replace=false)
    xtm1_sub    = xtm1[:, idx_sub]
    ℓwtm1_sub   = ℓwtm1[idx_sub]

    ρ_grid = [0.01, 0.99] #[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    if t == 2
        function obj_init(params)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(params[1])
            sampler′ = @set sampler′.dampings[t] = params[2]
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub) +
                   0.01 * abs2(params[1])
        end

        ℓh_lower_guess = -10.0
        ρ_guess        = 0.5

        ## Find any point that is not degenerate
        ℓh_decrease_stepsize = log(0.5)
        ℓh_lower, n_feasible_evals = find_feasible_point(
            ℓh′ -> obj_init([ℓh′, ρ_guess]),
            ℓh_lower_guess,
            ℓh_decrease_stepsize,
            log(eps(Float64)),
        )

        ## Find remaining endpoint of an interval containing a (possibly local) minima
        ℓh_upper_increase_ratio = (1 + √5) / 2
        n_interval_max_iters = ceil(Int, log(ℓh_upper_increase_ratio, 40))
        ℓh_upper, _, n_interval_evals = find_golden_section_search_interval(
            ℓh′ -> obj_init([ℓh′, ρ_guess]),
            ℓh_lower,
            ℓh_upper_increase_ratio,
            1;
            n_max_iters=n_interval_max_iters,
        )

        ## Properly optimize objective with the initial guess damping
        gss_abstol = 1e-2
        ℓh, n_gss_init_iters = golden_section_search(
            ℓh′ -> obj_init([ℓh′, ρ_guess]), ℓh_lower, ℓh_upper; abstol=gss_abstol
        )

        ## Properly optimize objective with coordinate descent
        ℓh_lower_coordesc, ℓh_upper_coordesc = ℓh - 1, ℓh + 1
        ρ = ρ_guess
        n_obj_evals = 0
        coordesc_iters = 0
        n_coordesc_max_iters = 32
        coordesc_abstol = 1e-2

        while true
            ℓh′, n_gss_iters = golden_section_search(
                ℓh′ -> obj_init([ℓh′, ρ]),
                ℓh_lower_coordesc,
                ℓh_upper_coordesc;
                abstol=gss_abstol,
            )
            n_obj_evals += 2 * n_gss_iters

            ρ′ = argmin(ρ′ -> obj_init([ℓh′, ρ′]), ρ_grid)
            n_obj_evals += length(ρ_grid)

            coordesc_iters += 1
            if max(abs(ℓh′ - ℓh), abs(ρ′ - ρ)) < coordesc_abstol ||
                coordesc_iters == n_coordesc_max_iters
                ρ  = ρ′
                ℓh = ℓh′
                break
            end

            ρ  = ρ′
            ℓh = ℓh′
        end

        h = exp(ℓh)

        stats = (
            feasible_search_objective_evaluations       = n_feasible_evals,
            gss_interval_search_objective_evaluations   = n_interval_evals,
            initialization_search_objective_evaluations = 2 * n_gss_init_iters,
            coordinate_descent_iterations               = coordesc_iters,
            coordinate_descent_objective_evaluations    = n_obj_evals,
            uhmc_stepsize                               = h,
            uhmc_damping                                = ρ,
        )

        sampler = @set sampler.stepsizes[2] = h
        sampler = @set sampler.dampings[2] = ρ
        return sampler, stats
    else
        ℓh_prev            = log(sampler.stepsizes[t - 1])
        ℓh_lower, ℓh_upper = ℓh_prev - 1, ℓh_prev + 1
        ρ_prev             = sampler.dampings[t - 1]

        function obj(params)
            rng_fixed = copy(rng)
            sampler′ = @set sampler.stepsizes[t] = exp(params[1])
            sampler′ = @set sampler′.dampings[t] = params[2]
            _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
            return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub) +
                   0.01 * abs2(ℓh_prev - params[1])
        end

        # Coordinate descent
        ℓh_lower_coordesc, ℓh_upper_coordesc = ℓh_prev - 1, ℓh_prev + 1
        ℓh = ℓh_prev
        ρ = ρ_prev
        n_obj_evals = 0
        coordesc_iters = 0
        n_coordesc_max_iters = 16
        coordesc_abstol = 1e-2
        gss_abstol = 1e-2

        while true
            ℓh′, n_gss_iters = golden_section_search(
                ℓh′ -> obj([ℓh′, ρ]),
                ℓh_lower_coordesc,
                ℓh_upper_coordesc;
                abstol=gss_abstol,
            )
            n_obj_evals += 2 * n_gss_iters

            ρ′ = argmin(ρ′ -> obj([ℓh′, ρ′]), ρ_grid)
            n_obj_evals += length(ρ_grid)

            coordesc_iters += 1
            if max(abs(ℓh′ - ℓh), abs(ρ′ - ρ)) < coordesc_abstol ||
                coordesc_iters == n_coordesc_max_iters
                ρ  = ρ′
                ℓh = ℓh′
                break
            end

            ρ  = ρ′
            ℓh = ℓh′
        end

        h = exp(ℓh)

        sampler = @set sampler.stepsizes[t] = h
        sampler = @set sampler.dampings[t] = ρ
        stats   = (uhmc_stepsize                            = exp(ℓh), uhmc_damping                             = ρ, coordinate_descent_iterations            = coordesc_iters, coordinate_descent_objective_evaluations = n_obj_evals)
        return sampler, stats
    end
end
