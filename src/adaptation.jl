
abstract type AbstractAdaptor end

struct NoAdaptation <: AbstractAdaptor end

struct ForwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct BackwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct PartialForwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct PartialBackwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct AnnealedFlowTransport <: AbstractAdaptor
    n_subsample::Int
end

struct LogPotentialVarianceMin <: AbstractAdaptor
    n_subsample::Int
end

struct AcceptanceRate{Acc<:Real} <: AbstractAdaptor
    n_subsample::Int
    target_acceptance_rate::Acc
end

struct CondESSMax <: AbstractAdaptor end

adaptation_objective(::NoAdaptation, ::AbstractVector, ::AbstractVector) = 0.0

function adaptation_objective(
    ::ForwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    N       = length(ℓw)
    ℓ∑w     = logsumexp(ℓw)
    ℓw_norm = @. ℓw - ℓ∑w
    ℓw′     = ℓw_norm + ℓG
    ℓ∑w′    = logsumexp(ℓw′)
    ℓEw′    = ℓ∑w′ - log(N)
    w′_norm = @. exp(ℓw′ - ℓ∑w′)
    return dot(w′_norm, ℓG) - ℓEw′
end

function adaptation_objective(
    ::BackwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    N       = length(ℓw)
    ℓ∑w     = logsumexp(ℓw)
    ℓw_norm = @. ℓw - ℓ∑w
    ℓw′     = ℓw_norm + ℓG
    ℓEw′    = logsumexp(ℓw′) - log(N)
    return -mean(ℓG) + ℓEw′
end

function adaptation_objective(
    ::PartialForwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    ℓ∑w     = logsumexp(ℓw)
    ℓw_norm = @. ℓw - ℓ∑w
    ℓw′     = ℓw_norm + ℓG
    ℓ∑w′    = logsumexp(ℓw′)
    w′_norm = @. exp(ℓw′ - ℓ∑w′)
    return dot(w′_norm, ℓG)
end

function adaptation_objective(
    ::PartialBackwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    return -mean(ℓG)
end

function adaptation_objective(
    ::AnnealedFlowTransport, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    ∑ℓw    = logsumexp(ℓw)
    w_norm = @. exp(ℓw - ∑ℓw)
    return dot(w_norm, -ℓG)
end

function adaptation_objective(::CondESSMax, ℓw::AbstractVector, ℓG::AbstractVector, args...)
    N         = length(ℓw)
    ℓ∑WG      = logsumexp(ℓw + ℓG)
    ℓEWG2     = logsumexp(ℓw + 2 * ℓG) - log(N)
    ℓcond_ess = 2 * ℓ∑WG - ℓEWG2
    return -ℓcond_ess
end

function adaptation_objective(
    ::LogPotentialVarianceMin, ℓw::AbstractVector, ℓG::AbstractVector, args...
)
    return var(ℓG)
end

function adaptation_objective(
    adaptor::AcceptanceRate, ::AbstractVector, ::AbstractVector, acceptance_rate
)
    return (acceptance_rate - adaptor.target_acceptance_rate)^2
end

function find_feasible_point(f, x0::Real, stepsize::Real, lb::Real)
    @assert x0 > lb

    x      = x0
    y      = f(x)
    n_eval = 1
    while x > lb
        if isfinite(y)
            return x, n_eval
        else
            x      -= stepsize
            y      = f(x)
            n_eval += 1
        end
    end
    throw(
        errorexception(
            "could not find a fesible initial stepsize after $(n_eval) steps: x = $(x), f(x) = $(y), lb = $(lb)",
        ),
    )
end

function golden_section_search(f, a::Real, b::Real; n_max_iters::Int=10, abstol::Real=1e-2)
    ϕinv    = (√5 - 1) / 2
    n_iters = 0
    for t in 1:n_max_iters
        c = b - (b - a) * ϕinv
        d = a + (b - a) * ϕinv

        if f(c) < f(d)
            b = d
        else
            a = c
        end

        n_iters = t

        if (b - a) ≤ abstol
            break
        end

        if !isfinite(b + a)
            throw(
                ErrorException(
                    "Golden section search failed at b = $b, a = $a with f((a+b)/2) = $(f((a+b)/2))",
                ),
            )
        end
    end
    return (b + a) / 2, n_iters
end

function approx_exact_linesearch(f, x, t0::Real, dir)
    β  = 0.9 #2/(1 + √5)
    d  = dir
    t  = t0
    y  = f(x)
    y′ = f(x + t * d)
    α  = β

    n_evals = 2

    if y′ ≤ y
        α = 1 / β
    end

    while true
        t = α * t
        y = y′
        y′ = f(x + t * d)

        n_evals += 1

        if y′ ≥ y
            break
        end
    end

    if t ≈ t0 / β
        t = t0
        α = β
        y′, y = y, y′

        while true
            t = α * t
            y = y′
            y′ = f(x + t * d)

            n_evals += 1

            if y′ > y
                break
            end
        end
    end

    return t, n_evals
end
