
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
    ::BackwardKLMin,
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
    adaptor::AcceptanceRateControl, log_acceptance_rate::Real, ::Real,
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
            "could not find a fesible initial stepsize after $(n_eval) steps: x = $(x), f(x) = $(y), lb = $(lb)",
        ),
    )
end

function golden_section_search(f, a::Real, b::Real, x::Real; abstol::Real=1e-2)
    ϕinv    = (√5 - 1) / 2
    ϕinvc   = 1 - ϕinv
    n_evals = 0

    x0, x1, x2, x3 = a, 0, 0, b

    if abs(b - x) > abs(x - a)
        x1 = x
        x2 = x + ϕinvc*(b - x)
    else
        x2 = x
        x1 = x - ϕinvc*(x - a)
    end

    while abs(x1 - x2) ≥ abstol
        f1 = f(x1)
        f2 = f(x2)

        n_evals += 2
        if f2 < f1 || !isfinite(f1)
            x0 = x1
            x1 = x2
            x2 = ϕinv*x2 + ϕinvc*x3
            f1 = f2
            f2 = f(x2)
        elseif isfinite(f2)
            x3 = x2
            x2 = x1
            x1 = ϕinv*x1 + ϕinvc*x0
            f2 = f1
            f1 = f(x1)
        else
            throw(
                ErrorException(
                    "Golden section search failed at a = $a, b = $b, c = $c, d = $d, with f(c) = $fc, f(d) = $fd",
                ),
            )
        end
    end
    return (x1 + x2) / 2, n_evals
end

function bracket_minimum(f, a::Real, c::Real, r::Real; n_max_iters::Int=100)
    @assert r > 1

    y = f(a)
    for k in 1:n_max_iters
        b = a + c * r^k
        y′ = f(b)
        if !isfinite(y′) || y < y′
            return b, a + c * r^(k - 1), k
        end
        y      = y′
    end
    throw(
        ErrorException(
            "Bracket minimum didn't terminate within $(n_max_iters) iterations, where f(b) = $(y)",
        )
    )
end

function minimize(f, x0::Real, c::Real, r::Real, ϵ::Real)
    n_eval_total = 0 

    x_plus, _, n_eval  = bracket_minimum(f, x0, c, r)   
    n_eval_total      += n_eval

    x_minus, x_int, n_eval = bracket_minimum(f, x_plus, -c, r)   
    n_eval_total          += n_eval

    x_opt, n_eval = golden_section_search(f, x_minus, x_plus, x_int; abstol=ϵ)
    n_eval_total += n_eval

    # x_viz = range(-15, 2; length=32)
    # y_viz = @showprogress map(f, x_viz)
    # Plots.plot(x_viz, y_viz; yscale=:log10) |> display
    # Plots.vline!([x0], label="x0") |> display
    # Plots.vline!([x_minus], label="x_minus") |> display
    # Plots.vline!([x_plus], label="x_plus") |> display
    # Plots.vline!([x_opt], label="x_opt") |> display
    # Plots.vline!([x_int], label="x_int") |> display

    # if readline() == "n"
    #     throw()
    # end

    x_opt, n_eval_total
end
