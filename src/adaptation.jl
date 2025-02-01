
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

function golden_section_search(f, a::Real, b::Real; abstol::Real=1e-2)
    ϕinv    = (√5 - 1) / 2
    n_evals = 2
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

        n_evals += 2
    end
    return (b + a) / 2, n_evals
end

function bracket_minimum(f, a::Real, c::Real, r::Real)
    @assert r > 1

    b  = a
    y0 = f(a)
    y  = y0
    k  = 0
    n_evals = 2
    while y0 ≥ y
        b = a + c * r^k
        y′ = f(b)
        if !isfinite(y′)
            return a + c * r^(k - 1), n_evals
        elseif y < y′
            break
        end
        y        = y′
        k       += 1
        n_evals += 1
    end
    return a + c * r^k, n_evals
end

function minimize(f, x0::Real, c::Real, r::Real, ϵ::Real)
    n_eval_total = 0 
    x_plus, n_eval  = bracket_minimum(f, x0, c, r)   
    n_eval_total   += n_eval

    x_minus, n_eval = bracket_minimum(f, x_plus, -c, r)   
    n_eval_total   += n_eval

    x_opt, n_eval = golden_section_search(f, x_minus, x_plus; abstol=ϵ)
    n_eval_total += n_eval

    # x_viz = range(-15, 2; length=32)
    # y_viz = @showprogress map(f, x_viz)
    # Plots.plot(x_viz, y_viz; yscale=:log10) |> display
    # Plots.vline!([x0], label="x0") |> display
    # Plots.vline!([x_minus], label="x_minus") |> display
    # Plots.vline!([x_plus], label="x_plus") |> display
    # Plots.vline!([x_opt], label="x_opt") |> display

    # if readline() == "n"
    #     throw()
    # end

    x_opt, n_eval_total
end
