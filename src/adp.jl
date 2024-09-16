
function optimize_policy(sampler::AbstractControlledSMC, states; show_progress=true)
    (; smc, path, policy) = sampler
    (; stepsize_proposal, stepsize_problem, precond) = smc
    h0, hT, Γ = stepsize_proposal, stepsize_problem, precond

    T           = length(path)
    prog        = ProgressMeter.Progress(T; enabled=show_progress)
    policy_next = deepcopy(policy)

    twist_recur = last(policy)
    for t in T:-1:1
        (; particles, log_potential) = states[t]

        V = if t == T
            log_potential
        else
            twist_prev     = policy[t+1]
            a_prev, b_prev = twist_prev.a, twist_prev.b

            # Need to fix the factor of 2 thing
            htp1       = h0
            qtp1       = states[t+1].q
            γ          = diag(Γ)
            K          = Diagonal(@. 1/(a_prev/htp1 + 1/γ))
            μ_twisted  = K*(Γ\qtp1 .- 2*htp1*b_prev)
            Σ_twisted  = 2*htp1*K
            logM       = twist_mvnormal_logmarginal(twist_recur, μ_twisted, Σ_twisted)
            log_potential + logM
        end

        Δa, Δb, Δc, rmse = fit_quadratic(particles, -V)

        pm_next!(prog, (rmse=rmse, iteration=t))

        a_next = policy[t].a + Δa
        b_next = policy[t].b + Δb
        c_next = policy[t].c + Δc

        policy_next[t] = (a=a_next, b=b_next, c=c_next)
        twist_recur    = (a=Δa, b=Δb, c=Δc)
    end
    @set sampler.policy = policy_next
end
