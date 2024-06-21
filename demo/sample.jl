
function smc(
    rng        ::Random.AbstractRNG,
    sampler    ::AbstractSMC,
    n_particles::Int,
    threshold  ::Real,
    proposal   ::MvNormal,
    logtarget,
)
    T    = length(sampler.path)
    x    = rand(rng, proposal, n_particles)
    logw = fill(-log(n_particles), n_particles)
    G    = zeros(n_particles)
    logZ = 0.0

    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    states[1] = (particles=x,)
    info[1]   = (iteration=1, ess=n_particles, logZ=logZ)

    for t in 2:T
        @inbounds for i in 1:size(x,2)
            xi_prev = x[:,i]
            xi_curr = mutate(rng, sampler, t, xi_prev, proposal, logtarget)
            Gi      = potential(
                sampler, sampler.backward, t, xi_curr, xi_prev, proposal, logtarget
            )
            x[:,i] = xi_curr
            G[i]   = Gi
        end

        w, logw, logZ, ess              = reweight(logw, G, logZ)
        x, w, logw, ancestor, resampled = resample(rng, x, w, logw, ess, threshold)

        states[t] = (particles=x, ancestor=ancestor)
        info[t]   = (iteration=t, ess=ess, logZ=logZ, resampled=resampled)
    end
    x, info
end
