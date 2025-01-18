
abstract type AbstractAdaptor end

struct NoAdaptation <: AbstractAdaptor end

struct ForwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct BackwardKLMin <: AbstractAdaptor
    n_subsample::Int
end

struct AnnealedFlowTransport <: AbstractAdaptor
    n_subsample::Int
end

struct ESJDMax <: AbstractAdaptor
    n_subsample::Int
end

struct ChiSquareMin <: AbstractAdaptor
    n_subsample::Int
end

struct AcceptanceRateCtrl{Acc<:Real} <: AbstractAdaptor
    n_subsample::Int
    target_acceptance_rate::Acc
end

struct CondESSMax <: AbstractAdaptor
    n_subsample::Int
end

adaptation_objective(::NoAdaptation, ::AbstractVector, ::AbstractVector) = 0.0

function adaptation_objective(
    ::ForwardKLMin, ℓw::AbstractVector, ℓdPdQ::AbstractVector, ℓG::AbstractVector, args...
)
    ℓ∑w       = logsumexp(ℓw)
    ℓw_norm   = @. ℓw - ℓ∑w
    w_norm    = exp.(ℓw_norm)
    ∑ℓdPdQ    = logsumexp(ℓdPdQ)
    dPdQ_norm = @. exp(ℓdPdQ - ∑ℓdPdQ)
    return dot(w_norm .* dPdQ_norm, xexpx.(ℓG))
end

function adaptation_objective(
    ::BackwardKLMin, ℓw::AbstractVector, ℓdPdQ::AbstractVector, ℓG::AbstractVector, args...
)
    ∑ℓw       = logsumexp(ℓw)
    w_norm    = @. exp(ℓw - ∑ℓw)
    ∑ℓdPdQ    = logsumexp(ℓdPdQ)
    dPdQ_norm = @. exp(ℓdPdQ - ∑ℓdPdQ)
    return dot(w_norm .* dPdQ_norm, -ℓG)
end

function adaptation_objective(
    ::AnnealedFlowTransport,
    ℓw::AbstractVector,
    ℓdPdQ::AbstractVector,
    ℓG::AbstractVector,
    args...,
)
    ∑ℓw    = logsumexp(ℓw)
    w_norm = @. exp(ℓw - ∑ℓw)
    return dot(w_norm, -ℓG)
end

function adaptation_objective(
    ::ChiSquareMin, ℓw::AbstractVector, ℓdPdQ::AbstractVector, ℓG::AbstractVector, args...
)
    ℓ∑w        = logsumexp(ℓw)
    ℓw_norm    = ℓw .- ℓ∑w
    ∑ℓdPdQ     = logsumexp(ℓdPdQ)
    ℓdPdQ_norm = ℓdPdQ .- ∑ℓdPdQ
    return logsumexp(ℓw_norm + ℓdPdQ_norm + 2 * logsubexp(ℓG, 0))
end

function adaptation_objective(
    ::CondESSMax, ℓw::AbstractVector, ::AbstractVector, ℓG::AbstractVector, args...
)
    N         = length(ℓw)
    ℓ∑WG      = logsumexp(ℓw + ℓG)
    ℓEWG2     = logsumexp(ℓw + 2 * ℓG) - log(N)
    ℓcond_ess = 2 * ℓ∑WG - ℓEWG2
    return -ℓcond_ess
end

function adaptation_objective(
    adaptor::AcceptanceRateCtrl,
    ::AbstractVector,
    ::AbstractVector,
    ::AbstractVector,
    log_acceptance_rate,
    esjd,
)
    return (log_acceptance_rate - log(adaptor.target_acceptance_rate))^2
end

function adaptation_objective(
    ::ESJDMax, ::AbstractVector, ::AbstractVector, ::AbstractVector, acceptance_rate, esjd
)
    return -esjd
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
            x -= stepsize
            y = f(x)
            n_eval += 1
        end
    end
    throw(
        errorexception(
            "could not find a fesible initial stepsize after $(n_eval) steps: x = $(x), f(x) = $(y), lb = $(lb)",
        ),
    )
end

function golden_section_search(f, a::Real, b::Real; abstol::Real=1e-2)
    ϕinv    = (√5 - 1) / 2
    n_iters = 0
    while true
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
            throw(
                ErrorException(
                    "Golden section search failed at b = $b, a = $a with f((a+b)/2) = $(f((a+b)/2))",
                ),
            )
        end

        n_iters += 1
    end
    return (b + a) / 2, n_iters
end

function find_golden_section_search_interval(f, a::Real, δ::Real, r::Real)
    @assert r > 1

    b  = a
    y0 = f(a)
    y  = y0
    k  = 0
    while y0 ≥ y
        b = a + δ * r^k
        y′ = f(b)
        if y < y′
            break
        elseif !isfinite(y′)
            @warn "Degenerate objective value f(x) = $y′ for x = $b encountered during golden section search initial interval search."
            return a + δ * r^(k - 1), a + δ * r^(k - 2), k
        end
        y = y′
        k += 1
    end
    return a + δ * r^k, a + δ * r^(k - 1), k
end
