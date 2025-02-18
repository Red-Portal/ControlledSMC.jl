
struct PFULA{FK<:AbstractParticleFilter,Stepsizes,ADType<:ADTypes.AbstractADType,Adaptor} <:
       AbstractParticleFilter
    fkmodel::FK
    stepsizes::Stepsizes
    adtype::ADType
    adaptor::Adaptor
end

Base.length(sampler::PFULA) = length(sampler.fkmodel)

function rand_initial_with_potential(
    rng::Random.AbstractRNG, model::PFULA, n_particles::Int
)
    return rand_initial_with_potential(rng, model.fkmodel, n_particles)
end

function logpdf_optimal_guide_ad_fwd(
    xt::AbstractMatrix, xtm1::AbstractMatrix, sampler::AbstractParticleFilter, t::Int
)
    ℓP     = logpdf_transition(sampler, t, xtm1, xt)
    ℓG_bpf = potential(sampler, t, xt)
    return sum(ℓG_bpf + ℓP)
end

function PFULA(
    adtype::ADTypes.AbstractADType,
    fkmodel::AbstractParticleFilter,
    adaptor::AbstractAdaptor,
)
    stepsizes = zeros(Float64, length(fkmodel))
    return PFULA{typeof(fkmodel),typeof(stepsizes),typeof(adtype),typeof(adaptor)}(
        fkmodel, stepsizes, adtype, adaptor
    )
end

function PFULA(
    adtype::ADTypes.AbstractADType,
    fkmodel::AbstractParticleFilter,
    stepsizes::AbstractVector,
)
    return PFULA{typeof(fkmodel),typeof(stepsizes),typeof(adtype),Nothing}(
        fkmodel, stepsizes, adtype, nothing
    )
end

function ControlledSMC.mutate_with_potential(
    rng::Random.AbstractRNG, sampler::PFULA, t::Int, xtm1::AbstractMatrix
)
    (; adtype, fkmodel, adtype, stepsizes) = sampler
    ht = stepsizes[t]
    d = size(xtm1, 1)

    ∇ℓM_opt = DI.gradient(
        logpdf_optimal_guide_ad_fwd,
        adtype,
        xtm1,
        DI.Constant(xtm1),
        DI.Constant(fkmodel),
        DI.Constant(t),
    )
    ∇U = -∇ℓM_opt
    M = ControlledSMC.BatchMvNormal(xtm1 - ht * ∇U, 2 * ht * Eye(d))
    xt = rand(rng, M)
    ℓM = logpdf(M, xt)
    ℓG_bpf = potential(fkmodel, t, xt)
    ℓP = logpdf_transition(fkmodel, t, xtm1, xt)
    ℓG_gpf = ℓG_bpf + ℓP - ℓM
    return xt, ℓG_gpf, NamedTuple()
end

function ControlledSMC.adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::PFULA,
    t::Int,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if isnothing(sampler.adaptor)
        return sampler, NamedTuple()
    end
    # Subsample particles to reduce adaptation overhead
    w_norm    = exp.(ℓwtm1 .- logsumexp(ℓwtm1))
    n_sub     = sampler.adaptor.n_subsample
    sub_idx   = ssp_sampling(rng, w_norm, n_sub)
    xtm1_sub  = xtm1[:, sub_idx]
    ℓdPdQ_sub = ℓwtm1[sub_idx]
    ℓwtm1_sub = fill(-log(n_sub), n_sub)

    τ = sampler.adaptor.regularization

    function obj(ℓh′)
        rng_fixed    = copy(rng)
        sampler′     = @set sampler.stepsizes[t] = exp(ℓh′)
        _, ℓG_sub, _ = mutate_with_potential(rng_fixed, sampler′, t, xtm1_sub)
        reg          = if t == 1
            τ * abs2(ℓh′)
        else
            ℓh_prev = log(sampler.stepsizes[t - 1])
            τ * abs2(ℓh′ - ℓh_prev)
        end
        return adaptation_objective(sampler.adaptor, ℓwtm1_sub, ℓdPdQ_sub, ℓG_sub) + reg
    end

    r = 2.0
    c = 0.1
    ϵ = 1e-2
    δ = -1
    ℓh_guess = -10.0
    n_evals_total = 0

    ℓh = if t == 1
        ℓh, n_evals = find_feasible_point(obj, ℓh_guess, δ, log(eps(eltype(xtm1))))
        n_evals_total += n_evals
        ℓh
    else
        log(sampler.stepsizes[t - 1])
    end
    ℓh, n_evals = minimize(obj, ℓh, c, r, ϵ)
    n_evals_total += n_evals

    h = exp(ℓh)

    stats = (ula_stepsize=h, n_objective_evals=n_evals_total)

    sampler = @set sampler.stepsizes[t] = h
    return sampler, stats
end
