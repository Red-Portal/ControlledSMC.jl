
struct SMCUHMC{Stepsizes<:AbstractVector,Dampings<:AbstractVector,Mass<:AbstractMatrix} <: AbstractSMC
    stepsizes   :: Stepsizes
    dampings    :: Stepsizes
    mass_matrix :: Mass
end

function SMCUHMC(stepsize::Real, damping::Real, n_steps::Int, mass_matrix::AbstractMatrix)
    stepsizes = Fill(stepsize, n_steps)
    dampings  = Fill(damping, n_steps)
    SMCUHMC{typeof(stepsizes), typeof(dampings), typeof(mass_matrix)}(
        stepsizes, dampings,  mass_matrix
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
    idx_sub     = StatsBase.sample(
        1:n_particles, sampler.adaptor.n_subsample; replace=false
    )
    xtm1_sub  = xtm1[:,idx_sub]
    ℓwtm1_sub = ℓwtm1[idx_sub]

    # Set online optimization algorithm configurations
    ℓh_lower, ℓh_upper = if t == 2
        log(1e-8), log(10)
    else
        ℓh_prev = log(sampler.stepsizes[t - 1])
        ℓh_prev - 1, ℓh_prev + 1
    end
    n_max_iters = (t == 2) ? 64 : 16

    # Solve optimization problem
    ℓh, n_gss_iters = golden_section_search(
        ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
    ) do ℓh′
        rng_fixed = copy(rng)
        sampler′ = @set sampler.stepsizes[t] = exp(ℓh′)

        # If t == 2, also optimize the stepsize at t = 1.
        # For simplicity, we just set h[1] = h[2], which is suboptimal,
        # but shouldn't be too critical.
        if t == 2
            sampler′ = @set sampler.stepsizes[1] = exp(ℓh′)
        end
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, πt, πtm1, xtm1_sub)
        adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓG_sub)
    end

    h       = exp(ℓh)
    sampler = @set sampler.stepsizes[t] = exp(ℓh)
    stats   = (
        golden_section_search_iterations = n_gss_iters,
        ula_stepsize                     = h
    )
    return sampler, stats
end
