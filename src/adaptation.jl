
abstract type AbstractAdaptor end

@kwdef struct BackwardKLMin{Reg<:Real} <: AbstractAdaptor
    n_subsample::Int
    regularization::Reg = 0.0
end

@kwdef struct ESJDMax{Reg<:Real} <: AbstractAdaptor
    n_subsample::Int
    regularization::Reg = 0.0
end

@kwdef struct AcceptanceRateControl{Acc<:Real,Reg<:Real} <: AbstractAdaptor
    n_subsample::Int
    target_acceptance_rate::Acc
    regularization::Reg = 0.0
end

function adaptation_objective(
    ::BackwardKLMin, ℓw::AbstractVector, ℓdPdQ::AbstractVector, ℓG::AbstractVector, args...
)
    ∑ℓw    = logsumexp(ℓw)
    w_norm = @. exp(ℓw - ∑ℓw)
    return dot(w_norm, -ℓG)
end

function adaptation_objective(
    adaptor::AcceptanceRateControl, log_acceptance_rate::Real, ::Real
)
    return (log_acceptance_rate - log(adaptor.target_acceptance_rate))^2
end

function adaptation_objective(::ESJDMax, ::Real, esjd::Real)
    return -esjd
end

function find_feasible_point(f, x0::Real, δ::Real, lb::Real)
    @assert x0 > lb

    x      = x0
    y      = f(x)
    n_eval = 1
    while x > lb
        if isfinite(y)
            return x, n_eval
        else
            x += δ
            y = f(x)
            n_eval += 1
        end
    end
    throw(
        ErrorException(
            "could not find a feasible initial stepsize after $(n_eval) steps: x = $(x), f(x) = $(y), lb = $(lb)",
        ),
    )
end

function golden_section_search(f, a::Real, b::Real, c::Real; abstol::Real=1e-2)
    @assert a < b && b < c

    ϕinv    = (√5 - 1) / 2
    ϕinvc   = 1 - ϕinv
    n_evals = 0

    x0, x1, x2, x3 = a, 0, 0, c

    if abs(c - b) > abs(b - a)
        x1 = b
        x2 = b + ϕinvc * (c - b)
    else
        x2 = b
        x1 = b - ϕinvc * (b - a)
    end

    f1 = f(x1)
    f2 = f(x2)
    n_evals += 2

    while abs(x0 - x3) ≥ abstol
        if f2 < f1 || !isfinite(f1)
            x0 = x1
            x1 = x2
            x2 = ϕinv * x2 + ϕinvc * x3
            f1 = f2
            f2 = f(x2)
            n_evals += 1
        elseif (f2 ≥ f1 || !isfinite(f2)) && isfinite(f1)
            x3 = x2
            x2 = x1
            x1 = ϕinv * x1 + ϕinvc * x0
            f2 = f1
            f1 = f(x1)
            n_evals += 1
        else
            throw(
                ErrorException(
                    "Golden section search failed at " *
                    "x0 = $x0, x1 = $x1, x2 = $x2, x3 = $x3" *
                    " with f(x1) = $f1, f(x2) = $f2",
                ),
            )
        end
    end
    return (x0 + x3) / 2, n_evals
end

function bracket_minimum(
    f, x0::Real, c::Real, r::Real; x_upper_limit::Real=Inf, x_lower_limit::Real=-Inf
)
    @assert c > 0
    @assert r > 1

    k  = 0
    y0 = f(x0)
    x  = x0
    y  = y0

    n_evals = 1
    x_plus  = x0
    x_minus = x0
    x_mid   = x0

    if isfinite(y)
        while true
            x′ = x0 + c * r^k
            y′ = f(x′)
            n_evals += 1
            if x′ ≥ x_upper_limit
                throw(
                    ErrorException(
                        "Bracket minimum first stage exceeded upper limit $(x_upper_limit) after $(k) iterations, " *
                        "where x0 = $(x0), x′ = $(x′), y = $(y), y′ = $(y′)",
                    ),
                )
            end
            if !isfinite(y′) || y < y′
                x_plus = x′
                x0     = x
                break
            end
            x = x′
            y = y′
            k += 1
        end
    else
        x_plus = x0
        x      = x0 - c/2
        y      = f(x)
        n_evals += 1
    end
    k = 0
    while true
        x′ = x0 - c * r^k
        y′ = f(x′)
        n_evals += 1
        if x′ ≤ x_lower_limit
            throw(
                ErrorException(
                    "Bracket minimum second stage exceeded lower limit $(x_lower_limit) after $(k) iterations, " *
                    "where x0 = $(x0), x′ = $(x′), y = $(y), y′ = $(y′)",
                ),
            )
        end
        if isfinite(y′) && isfinite(y) && (y < y′)
            x_minus = x′
            x_mid   = x
            break
        end
        x = x′
        y = y′
        k += 1
    end
    return x_minus, x_mid, x_plus, n_evals
end

function minimize(f, x0::Real, c::Real, r::Real, ϵ::Real)
    x_lower_limit = log(eps(Float64))
    n_eval_total  = 0

    x_minus, x_mid, x_plus, n_eval = bracket_minimum(f, x0, c, r; x_lower_limit)
    n_eval_total += n_eval

    x_opt, n_eval = golden_section_search(f, x_minus, x_mid, x_plus; abstol=ϵ)
    n_eval_total += n_eval

    # x_viz = range(-15, 5; length=64)
    # y_viz = @showprogress map(f, x_viz)
    # @info("", x0, x_minus, x_plus, x_opt, x_int)
    # Plots.plot(x_viz, y_viz, ylims=()) |> display
    # Plots.vline!([x0],      label="x0")      |> display
    # Plots.vline!([x_minus], label="x_minus") |> display
    # Plots.vline!([x_plus],  label="x_plus")  |> display
    # Plots.vline!([x_opt],   label="x_opt")   |> display
    # Plots.vline!([x_int],   label="x_int")   |> display
    # if readline() == "n"
    #    throw()
    # end

    return x_opt, n_eval_total
end
