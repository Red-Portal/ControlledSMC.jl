
using Plots
using DelimitedFiles

logplus(x) = log(max(x, 1))

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

    while abs(x3 - x0) ≥ abstol
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
    f,
    x0::Real,
    c::Real,
    r::Real;
    x_upper_limit::Real = Inf,
    x_lower_limit::Real = -Inf
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
            x  = x′
            y  = y′
            k += 1
        end
    else
        x = x0 - c/2
        y = f(x)
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
        x  = x′
        y  = y′
        k += 1
    end
    return x_minus, x_mid, x_plus, x0, n_evals
end

function main()
    x_under = -5
    x_over  = 5
    x_inf   = 100
    
    f(x) = if x ≤ x_under
        -(x - x_under)
    elseif x_under < x < x_over
        0
        #10.0*sin(2*π*10*((x - x_under) / (x_over - x_under)))
    elseif x_over ≤ x < x_inf
        (x - x_over)
    elseif x ≥ x_inf
        Inf
    else
        0
        # a = 1/(x_under - x_over)
        # b = -a * x_over
        # a*x + b
    end

    # x_under = 3
    # x_over  = x_under
    # x_inf   = 20

    # f(x) = if x < x_inf
    #     abs(x - x_over)^(2)
    # else
    #     Inf
    # end

    r  = 1.2
    c  = 0.5
    ϵ  = 1e-2
    x0 = 5
    #x_minus, x_mid, x_plus, n_evals_bm = bracket_minimum(f, x0, c, r)
    #x_opt, n_evals_gss = golden_section_search(f, x_minus, x_mid, x_plus; abstol=ϵ)

    # Plots.plot(-10:10, f)    |> display
    # Plots.vline!([x0],      label="x0")    |> display
    # Plots.vline!([x_plus],  label="x+")    |> display
    # Plots.vline!([x_mid],   label="x_mid") |> display
    # Plots.vline!([x_minus], label="x-")    |> display
    # Plots.vline!([x_opt],   label="x_opt") |> display

    # @info("", x_opt)

    #@info("",  x_plus, x_mid, x_minus, n_evals )
    #return

    x0s  = range(-100, 40; length=256)
    tups = map(x0s) do x0
        ζ2 = r^2 - 1
        ζ1 = r - 1

        x_minus, x_mid, x_plus, x0_ii, n_evals_bm = bracket_minimum(f, x0, c, r)
        x_opt, n_evals_gss = golden_section_search(f, x_minus, x_mid, x_plus; abstol=ϵ)

        case_i   = x0 < x_over - c
        case_ii  = !case_i
        case_iii = x0 < x_under - c
        case_iv  = !case_iii
        #case_v   = x0_ii < x_under + c
        #case_vi  = !case_v

        # x0_ii_lb = if x0 < x_under - c
        #     x0 + max(x_under - x0, 0)/r
        # else
        #     x0
        # end

        # x_plus_bound = if case_i
        #     x0 + r^2*max(x_over - x0, 0)
        # else
        #     x0 + c*r
        # end

        # x_minus_bound_casev = if case_iii
        #     x0 + 1/r*max(x_under - x0, 0) - c*r
        # elseif case_iv
        #     x0 - c*r
        # else
        #     throw()
        # end

        # x_minus_bound_casevi = if case_i
        #     x_under - ζ2*(x_over - x_under) - ζ1*ζ2*max(x_over - x0, 0)
        # elseif case_ii
        #     x_under - ζ2*max(x0 - x_under, 0) - ζ2*c
        # else
        #     throw()
        # end

        # x_minus_bound = if case_v
        #     x_minus_bound_casev
        # elseif case_vi
        #     x_minus_bound_casevi
        # else
        #     throw()
        # end

        U1 = if case_i && case_iii
            r^2*max(x_over - x0, 0) - 1/r*max(x_under - x0, 0) + c*r
        elseif case_i && case_iv
            r^2*max(x_over - x0, 0) + c*r
        elseif case_ii && case_iii
            2*c*r - 1/r*max(x_under - x0, 0)
        elseif case_ii && case_iv
            2*c*r
        else
            throw()
        end

        U2 = if case_i
            (x0 - x_under) + (r^2 + ζ1*ζ2)*max(x_over - x0, 0) + ζ2*(x_over - x_under)
        elseif case_ii
            (x0 - x_under) + ζ2*max(x0 - x_under, 0) + ζ2*c + c*r
        else
            throw()
        end

        C_bm = if case_i
            1/log(r)*log(max((x_over - x0)/c, 1)) +
                1/log(r)*log(max((
                    (x_over - x_under) + (r - 1)*max(x_over - x0, 0) )/c, 1)) + 8
        else
            1/log(r)*log(max((max(x0 - x_under, 0) )/c, 1)) + 6
        end

        C_gss = log(max(U1, U2)/ϵ)/log((1 + √5)/2) + 3

        n_evals_gss + n_evals_bm, C_gss + C_bm
    end

    Plots.plot(x0s,  [tup[1] for tup in tups])
    Plots.plot!(x0s, [tup[2] for tup in tups])
    Plots.vline!([x_under])
    Plots.vline!([x_over])
end

function simulate_bound()
    bound(x0, x_over, x_under, x_inf, r, c, ϵ) = begin
        ζ2 = r^2 - 1
        ζ1 = r - 1

        case_i   = x0 < x_over - c
        case_ii  = !case_i
        case_iii = x0 < x_under - c
        case_iv  = !case_iii

        U1 = if case_i && case_iii
            r^2*max(x_over - x0, 0) - 1/r*max(x_under - x0, 0) + c*r
        elseif case_i && case_iv
            r^2*max(x_over - x0, 0) + c*r
        elseif case_ii && case_iii
            2*c*r - 1/r*max(x_under - x0, 0)
        elseif case_ii && case_iv
            2*c*r
        else
            throw()
        end

        U2 = if case_i
            (x0 - x_under) + (r^2 + ζ1*ζ2)*max(x_over - x0, 0) + ζ2*(x_over - x_under)
        elseif case_ii
            (x0 - x_under) + ζ2*max(x0 - x_under, 0) + ζ2*c + c*r
        else
            throw()
        end

        C_bm = if case_i
            1/log(r)*log(max((x_over - x0)/c, 1)) +
                1/log(r)*log(max((
                    (x_over - x_under) + (r - 1)*max(x_over - x0, 0) )/c, 1)) + 8
        else
            1/log(r)*log(max((max(x0 - x_under, 0) )/c, 1)) + 6
        end
        C_gss = log(max(U1, U2)/ϵ)/log((1 + √5)/2) + 3
        C_gss + C_bm
    end

    x0      = 0.5
    ϵ       = 1e-2
    x_inf   = Inf
    x_under = 0
    x_over  = 0

    Plots.plot()

    # for c in [0.01, 0.1, 0.5, 1.0, 2, 5, 10]
    #     Plots.plot!(
    #         range(1.1, 2.0; length=2^10),
    #         r -> bound(x0, x_over, x_under, x_inf, r, c, ϵ),
    #         xlabel = "r", 
    #         ylims=[0, Inf],
    #         linewidth=2,
    #     ) |> display
    # end

    for r in [1.2, 1.6, 2.0, 2.4, 2.8],
        c in [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
        Plots.plot!(
            range(-20, 20; length=2^10),
            x0 -> bound(x0, x_over, x_under, x_inf, r, c, ϵ),
            xlabel = "x0", 
            ylims=[0, 50],
            linewidth=2,
        ) |> display

        xs = range(-20, 20; length=2^10)
        ys = map(x0 -> bound(x0, x_over, x_under, x_inf, r, c, ϵ), xs)

        open("data/pro/bound_xover=$(x_over)_xunder=$(x_under)_r=$(r)_c=$(c).txt", "w") do io
            writedlm(io, [xs ys], ',')
        end
    end
end
