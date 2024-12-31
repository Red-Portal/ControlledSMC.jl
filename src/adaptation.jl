
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
