
struct SMCMALA{
    Stepsizes<:AbstractVector,
    Backward<:AbstractBackwardKernel,
    Precond<:AbstractMatrix,
    Adaptor<:AbstractAdaptor,
} <: AbstractSMC
    stepsizes :: Stepsizes
    backward  :: Backward
    precond   :: Precond
    adaptor   :: Adaptor
end

function SMCMALA(
    stepsize::Real,
    n_steps::Int,
    backward::DetailedBalance,
    precond::AbstractMatrix,
    adaptor::AbstractAdaptor,
)
    stepsizes = Fill(stepsize, n_steps)
    return SMCMALA{typeof(stepsizes),typeof(backward),typeof(precond),typeof(adaptor)}(
        stepsizes, backward, precond, adaptor
    )
end

function transition_mala(rng::Random.AbstractRNG, h, Γ, πt, x::AbstractMatrix)
    n_particles = size(x, 2)

    μ_fwd  = gradient_flow_euler(πt, x, h, Γ)
    x_prop = μ_fwd + sqrt(2 * h) * unwhiten(Γ, randn(rng, eltype(μ_fwd), size(μ_fwd)))
    
    μ_bwd  = gradient_flow_euler(πt, x_prop, h, Γ)

    ℓπt_prop = LogDensityProblems.logdensity(πt, x_prop)
    ℓπt      = LogDensityProblems.logdensity(πt, x)

    q_fwd = MvNormal.(eachcol(μ_fwd), Ref(2 * h * Γ))
    q_bwd = MvNormal.(eachcol(μ_bwd), Ref(2 * h * Γ))

    ℓq_fwd = logpdf.(q_fwd, eachcol(x_prop))
    ℓq_bwd = logpdf.(q_bwd, eachcol(x))

    ℓα = @. min(ℓπt_prop - ℓπt + ℓq_bwd - ℓq_fwd, 0)
    ℓu = -Random.randexp(rng, length(ℓα))

    x_next = mapreduce(hcat, 1:size(x, 2)) do n
        if ℓα[n] > ℓu[n]
            x_prop[:,n]
        else
            x[:,n]
        end
    end
    α = exp(logsumexp(ℓα) - log(n_particles))
    x_next, α
end

function potential(::SMCMALA, t::Int, πt, πtm1, xtm1::AbstractMatrix)
    ℓπt_xtm1   = LogDensityProblems.logdensity(πt, xtm1)
    ℓπtm1_xtm1 = LogDensityProblems.logdensity(πtm1, xtm1)
    ℓπt_xtm1 - ℓπtm1_xtm1
end

function mutate_with_potential(
    rng::Random.AbstractRNG, sampler::SMCMALA, t::Int, πt, πtm1, xtm1::AbstractMatrix
)
    (; stepsizes, precond) = sampler
    ht, Γ       = stepsizes[t], precond

    xt, _ = transition_mala(rng, ht, Γ, πt, xtm1)
    ℓG    = potential(sampler, t, πt, πtm1, xtm1)
    return xt, ℓG, NamedTuple()
end

using Plots

function adapt_sampler(
    rng::Random.AbstractRNG,
    sampler::SMCMALA,
    t::Int,
    path::AbstractPath,
    xtm1::AbstractMatrix,
    ℓwtm1::AbstractVector,
)
    if sampler.adaptor isa NoAdaptation
        return sampler, NamedTuple()
    end

    if t == length(sampler.stepsizes)
        sampler = @set sampler.stepsizes[t] = sampler.stepsizes[t - 1]
        return sampler, NamedTuple()
    end

    Γ    = sampler.precond
    πtm1 = get_target(path, t - 1)
    πt   = get_target(path, t)
    πtp1 = get_target(path, t + 1)

    ℓh_lower, ℓh_upper = if t == 2
        log(1e-8), log(10)
    else
        ℓh_prev = log(sampler.stepsizes[t - 1])
        ℓh_prev - 1, ℓh_prev + 1
    end

    n_max_iters = (t == 1) ? 64 : 16

    # obj = map(range(ℓh_lower, ℓh_upper; length=32)) do ℓh′
    #     rng_fixed = copy(rng)
    #     h′    = exp(ℓh′)
    #     xt, _ = transition_mala(rng_fixed, h′, Γ, πt, xtm1)
    #     ℓGt   = potential(sampler, t, πt, πtm1, xtm1)
    #     ℓGtp1 = potential(sampler, t+1, πtp1, πt, xt)
    #     ℓwt   = ℓwtm1 + ℓGt 
    #     adaptation_objective(sampler.adaptor, ℓwt, ℓGtp1)
    # end
    # Plots.plot(range(ℓh_lower, ℓh_upper; length=32), obj) |> display

    ℓh, n_gss_iters = golden_section_search(
        ℓh_lower, ℓh_upper; n_max_iters, abstol=1e-2
    ) do ℓh′
        rng_fixed = copy(rng)
        h′    = exp(ℓh′)
        xt, α = transition_mala(rng_fixed, h′, Γ, πt, xtm1)
        ℓGt   = potential(sampler, t, πt, πtm1, xtm1)
        ℓGtp1 = potential(sampler, t+1, πtp1, πt, xt)
        ℓwt   = ℓwtm1 + ℓGt 
        adaptation_objective(sampler.adaptor, ℓwt, ℓGtp1, α)
    end

    # Plots.vline!([ℓh]) |> display
    # println(ℓh)

    # if readline() == "n"
    #     throw()
    # end

    h       = exp(ℓh)
    sampler = @set sampler.stepsizes[t] = exp(ℓh)
    stats   = (
        golden_section_search_iterations = n_gss_iters,
        mala_stepsize                    = h
    )
    return sampler, stats
end
