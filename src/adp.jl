
function fit_quadratic(x::AbstractMatrix{T}, y::AbstractVector{T}) where {T<:Real}
    d, n = size(x, 1), size(x, 2)
    @assert size(x, 2) == length(y)

    X   = Array(vcat(x .^ 2, x, ones(T, 1, n))')
    β   = ones(2 * d + 1)
    ϵ   = zero(T)
    Xty = X' * y
    XtX = Hermitian(X' * X)

    func(β_) = sum(abs2, X * β_ - y)
    function grad!(g, β_)
        return g[:] = 2 * (XtX * β_ - Xty)
    end
    function hess!(H, β_)
        return H[:, :] = 2 * XtX
    end
    df = TwiceDifferentiable(func, grad!, hess!, β)

    lower = vcat(fill(ϵ, d), fill(typemin(T), d + 1))
    upper = fill(typemax(T), 2 * d + 1)
    dfc   = TwiceDifferentiableConstraints(lower, upper)
    res   = optimize(df, dfc, β, IPNewton())

    β = Optim.minimizer(res)
    a = β[1:d]
    b = β[(d + 1):(2 * d)]
    c = β[end]

    return a, b, c, sum(abs2, X * β - y)
end

function optimize_policy(sampler::AbstractControlledSMC, states; show_progress=true)
    (; path, policy) = sampler

    T           = length(path)
    prog        = ProgressMeter.Progress(T; enabled=show_progress)
    policy_next = deepcopy(policy)
    rmses       = Array{Float64}(undef, T)

    ψ_recur = last(policy)
    for t in T:-1:1
        (; particles, log_potential) = states[t]

        V = if t == T
            log_potential
        else
            ℓM = twist_double_mvnormal_logmarginal(
                sampler, t + 1, policy[t + 1], ψ_recur, states[t + 1]
            )
            log_potential + ℓM
        end

        Δa, Δb, Δc, rmse = fit_quadratic(particles, -V)
        rmse_norm        = rmse / sqrt(size(particles, 2))

        pm_next!(prog, (rmse=rmse_norm, iteration=t))

        a_next = policy[t].a + Δa
        b_next = policy[t].b + Δb
        c_next = policy[t].c + Δc

        policy_next[t] = (a=a_next, b=b_next, c=c_next)
        ψ_recur        = (a=Δa, b=Δb, c=Δc)
        rmses[t]       = rmse_norm
    end

    if show_progress
        println("\n")
        display(lineplot(
            1:T, reverse(rmses); title="ADP RMSE", xlabel="Iteration", ylabel="RMSE"
        ))
    end

    @set sampler.policy = policy_next
end
