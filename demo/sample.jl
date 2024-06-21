
function Base.rand(
    rng        ::Random.AbstractRNG,
               ::AbstractSMC,
    proposal,
    n_particles::Int,
)
    rand(rng, proposal, n_particles)
end

function potential_init(::AbstractSMC, x::AbstractMatrix, proposal, logtarget)
    n_particles = size(x, 2)
    zeros(n_particles)
end

function smc(
    rng        ::Random.AbstractRNG,
    sampler    ::AbstractSMC,
    n_particles::Int,
    threshold  ::Real,
    proposal   ::MvNormal,
    logtarget,
)
    T  = length(sampler.path)
    x  = rand(rng, sampler, proposal, n_particles)
    ℓG = potential_init(sampler, x, proposal, logtarget)
    ℓZ = 0.0
    ℓw = fill(-log(n_particles), n_particles)

    ℓw, ℓZ, ess              = reweight(ℓw, ℓG, ℓZ)
    x, ℓw, ancestor, resampled = resample(rng, x, ℓw, ess, threshold)

    states = Array{NamedTuple}(undef, T)
    info   = Array{NamedTuple}(undef, T)

    states[1] = (particles=x,)
    info[1]   = (iteration=1, ess=n_particles, logZ=ℓZ)

    for t in 2:T
        for i in 1:size(x,2)
            xi_prev = x[:,i]
            xi_curr = mutate(rng, sampler, t, xi_prev, proposal, logtarget)

            ℓGi = potential(
                sampler, sampler.backward, t, xi_curr, xi_prev, proposal, logtarget
            )

            x[:,i] = xi_curr
            ℓG[i]  = ℓGi
        end

        ℓw, ℓZ, ess              = reweight(ℓw, ℓG, ℓZ)
        x, ℓw, ancestor, resampled = resample(rng, x, ℓw, ess, threshold)

        states[t] = (particles=x, ancestor=ancestor)
        info[t]   = (iteration=t, ess=ess, logZ=ℓZ, resampled=resampled)
    end
    x, info
end
