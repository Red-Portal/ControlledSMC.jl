
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

struct ESJDMax <: AbstractAdaptor
    n_subsample::Int
end

struct AcceptanceRateCtrl{Acc<:Real} <: AbstractAdaptor
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
    adaptor::AcceptanceRateCtrl, ::AbstractVector, ::AbstractVector, acceptance_rate, esjd
)
    return (acceptance_rate - adaptor.target_acceptance_rate)^2
end

function adaptation_objective(
    ::ESJDMax, ::AbstractVector, ::AbstractVector, acceptance_rate, esjd
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

function find_golden_section_search_interval(
    f,
    a::Real,
    ρ::Real,
    dir::Real;
    n_max_iters::Int=10,
)
    @assert ρ > 1

    b = a
    y = f(a)
    k = 0
    while k ≤ n_max_iters
        b = a + dir*ρ^k
        y′ = f(b)
        if y < y′
            break
        end
        y  = y′
        k += 1
    end
    return a + dir*ρ^k, a + dir*ρ^(k - 1), k
end


LineSearches.@with_kw struct ApproximatelyExactLineSearch{TI,TB} <:
                             LineSearches.AbstractLineSearch
    iterations::TI = 1_000
    beta::TB = 2 / (1 + √5)
end

function (ls::ApproximatelyExactLineSearch)(
    df::LineSearches.AbstractObjective,
    x::AbstractArray{T},
    s::AbstractArray{T},
    t0::Tt=real(T)(1),
    x_new::AbstractArray{T}=similar(x),
    ϕ_0=nothing,
    dϕ_0=nothing,
) where {T,Tt}
    β           = ls.beta
    n_max_evals = ls.iterations

    ϕ, _ = LineSearches.make_ϕ_dϕ(df, x_new, x, s)

    t = t0 / β
    y = isnothing(ϕ_0) ? ϕ(zero(t0)) : ϕ_0
    y′ = ϕ(t)
    α = β

    n_evals = 2

    if y′ ≤ y
        α = 1 / β
    end

    # Increase/decrease the step size as long as the objective is decreasing
    while true
        t = α * t
        y = y′
        y′ = ϕ(t)

        n_evals += 1
        if n_evals > n_max_evals
            throw(
                Optim.LineSearchException(
                    "Linesearch failed to converge, reached maximum iterations $(n_max_evals).", 
                ),
            )
        end

        if isfinite(y) && y′ ≥ y
            break
        end
    end

    # If increasing the step size failed, try decreasing it
    if t ≈ t0 / β
        t = t0
        α = β
        y′, y = y, y′

        while true
            t = α * t
            y = y′
            y′ = ϕ(t)

            n_evals += 1
            if n_evals > n_max_evals
                throw(
                    Optim.LineSearchException(
                        "Linesearch failed to converge, reached maximum iterations $(n_max_evals).",
                    ),
                )
            end

            if isfinite(y) && y′ > y
                break
            end
        end
    end

    if α < 1
        return t, y′
    else
        t = β^2 * t
        return t, ϕ(t)
    end
end

# function approx_exact_linesearch(
#     f, x, t0::Real, dir; beta::Real = 2/(1 + √5), n_max_evals=1_000
# )
#     β  = beta
#     d  = dir
#     t  = t0
#     y  = f(x)
#     y′ = f(x + t * d)
#     α  = β

#     n_evals = 2

#     if y′ ≤ y
#         α = 1 / β
#     end

#     while true
#         t = α * t
#         y = y′
#         y′ = f(x + t * d)

#         n_evals += 1
#         if n_evals > n_max_evals
#             throw(
#                 LineSearchs.LineSearchException(
#                     "Linesearch failed to converge, reached maximum iterations $(n_max_evals)."
#                 )
#             )
#         end

#         if y′ ≥ y
#             break
#         end
#     end

#     if t ≈ t0 / β
#         t = t0
#         α = β
#         y′, y = y, y′

#         while true
#             t = α * t
#             y = y′
#             y′ = f(x + t * d)

#             n_evals += 1
#             if n_evals > n_max_evals
#                 throw(
#                     LineSearchs.LineSearchException(
#                         "Linesearch failed to converge, reached maximum iterations $(n_max_evals)."
#                     )
#                 )
#             end

#             if y′ > y
#                 break
#             end
#         end
#     end

#     if α < 1 
#         return t
#     else
#         return β^2*t
#     end

#     return t, n_evals
# end
