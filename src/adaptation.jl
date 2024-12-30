
abstract type AbstractAdaptor end

struct NoAdaptation <: AbstractAdaptor end

struct ForwardKLMin <: AbstractAdaptor end

struct BackwardKLMin <: AbstractAdaptor end

struct PartialForwardKLMin <: AbstractAdaptor end

struct PartialBackwardKLMin <: AbstractAdaptor end

struct AnnealedFlowTransport <: AbstractAdaptor end

struct LogPotentialVarianceMin <: AbstractAdaptor end

struct CondESSMax <: AbstractAdaptor end

adaptation_objective(::NoAdaptation, ::AbstractVector, ::AbstractVector) = 0.0

function adaptation_objective(::ForwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector)
    N        = length(ℓw)
    ℓ∑w      = logsumexp(ℓw)
    ℓw_norm  = @. ℓw - ℓ∑w
    ℓw′      = ℓw_norm + ℓG
    ℓ∑w′     = logsumexp(ℓw′)
    ℓEw′     = ℓ∑w′ - log(N)
    w′_norm  = @. exp(ℓw′ - ℓ∑w′)
    return dot(w′_norm, ℓG) - ℓEw′
end

function adaptation_objective(::BackwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector)
    N        = length(ℓw)
    ℓ∑w      = logsumexp(ℓw)
    ℓw_norm  = @. ℓw - ℓ∑w
    ℓw′      = ℓw_norm + ℓG
    ℓEw′     = logsumexp(ℓw′) - log(N)
    return -mean(ℓG) + ℓEw′
end

function adaptation_objective(
    ::PartialForwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector
)
    ℓ∑w      = logsumexp(ℓw)
    ℓw_norm  = @. ℓw - ℓ∑w
    ℓw′      = ℓw_norm + ℓG
    ℓ∑w′     = logsumexp(ℓw′)
    w′_norm  = @. exp(ℓw′ - ℓ∑w′)
    return dot(w′_norm, ℓG)
end

function adaptation_objective(
    ::PartialBackwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector
)
    return -mean(ℓG)
end

function adaptation_objective(
    ::AnnealedFlowTransport, ℓw::AbstractVector, ℓG::AbstractVector
)
    ∑ℓw    = logsumexp(ℓw)
    w_norm = @. exp(ℓw - ∑ℓw)
    return dot(w_norm, -ℓG)
end

function adaptation_objective(
    ::CondESSMax, ℓw::AbstractVector, ℓG::AbstractVector
)
    N         = length(ℓw)
    ℓ∑WG      = logsumexp(ℓw + ℓG)
    ℓEWG2     = logsumexp(ℓw + 2*ℓG) - log(N)
    ℓcond_ess = 2*ℓ∑WG - ℓEWG2
    -ℓcond_ess
end

function adaptation_objective(
    ::LogPotentialVarianceMin, ℓw::AbstractVector, ℓG::AbstractVector
)
    return var(ℓG)
end


function golden_section_search(
    f, a::Real, b::Real; n_max_iters::Int=10, abstol::Real=1e-2
)
    ϕinv    = (√5 - 1)/2
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
            throw(ErrorException("Golden section search failed at b = $b, a = $a with f((a+b)/2) = $(f((a+b)/2))"))
        end
    end
    (b + a) / 2, n_iters
end
