
abstract type AbstractAdaptor end

struct NoAdaptation <: AbstractAdaptor end

struct PathForwardKLMin <: AbstractAdaptor end

struct PathBackwardKLMin <: AbstractAdaptor end

struct AnnealedFlowTransport <: AbstractAdaptor end

adaptation_objective(::NoAdaptation, ::AbstractVector, ::AbstractVector) = 0.0

function adaptation_objective(::PathForwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector)
    N        = length(ℓw)
    ℓ∑w      = logsumexp(ℓw)
    ℓw_norm  = @. ℓw - ℓ∑w
    ℓw′      = ℓw_norm + ℓG
    ℓ∑w′     = logsumexp(ℓw′)
    ℓEw′     = ℓ∑w′ - log(N)
    w′_norm  = @. exp(ℓw′ - ℓ∑w′)
    return dot(w′_norm, ℓG) - ℓEw′
end

function adaptation_objective(::PathBackwardKLMin, ℓw::AbstractVector, ℓG::AbstractVector)
    N        = length(ℓw)
    ℓ∑w      = logsumexp(ℓw)
    ℓw_norm  = @. ℓw - ℓ∑w
    ℓw′      = ℓw_norm + ℓG
    ℓEw′     = logsumexp(ℓw′) - log(N)
    return -mean(ℓG) + ℓEw′
end

function adaptation_objective(
    ::AnnealedFlowTransport, ℓw::AbstractVector, ℓG::AbstractVector
)
    ℓw′     = ℓw + ℓG
    ∑ℓw′    = logsumexp(ℓw′)
    w′_norm = @. exp(ℓw′ - ∑ℓw′)
    return dot(w′_norm, -ℓG)
end

function golden_section_search(
    f, a::Real, b::Real; n_iters::Int=10, abstol::Real=1e-4
)
    ϕinv = (√5 - 1)/2
    for _ in 1:n_iters
        c = b - (b - a) * ϕinv
        d = a + (b - a) * ϕinv

        if f(c) < f(d)
            b = d
        else
            a = c
        end

        if (b - a) ≤ abstol
            break
        end

        if !isfinite(b + a)
            throw(ErrorException("Golden section search failed at b = $b, a = $a with f((a+b)/2) = $(f((a+b)/2))"))
        end
    end
    (b + a) / 2
end

function zeroth_order_newton(
    f, x0::Real; n_iters::Int=10, stepsize::Real=1e-2, reltol::Real=1e-4
)
    h = stepsize
    x = x0
    for t in 1:n_iters
        fx  = f(x)
        fxp = f(x + h)
        fxm = f(x - h)
        x′  = x - h / 2 * (fxp - fxm) / (fxp - 2 * fx + fxm)

        @info("", t, x, fx, abs(x′ - x) / x)

        if abs(x′ - x) / abs(x) < reltol
            break
        end

        if !isfinite(x′)
            throw(ErrorException("Zeroth order Newton failed at x = $x with f(x) = $fx"))
        end

        x = x′
    end
    return x
end
