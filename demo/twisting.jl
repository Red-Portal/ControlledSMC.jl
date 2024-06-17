
struct QuadTwisting{
    M1 <: AbstractMatrix,
    M2 <: AbstractMatrix,
    V <: AbstractVector,
    S <: Real
}
    K::M1
    A::M2
    b::V
    c::S
end

function Distributions.logpdf(twisting::QuadTwisting, x)
    @unpack A, b, c = twisting
    dot(x, A*x) + dot(x, b) + c
end

struct MvNormalTwisted{TW <: QuadTwisting, MVN <: MvNormal}
    twisting::TW 
    mvnormal::MVN
end

function Distributions.rand(
    rng        ::Random.AbstractRNG,
    mvtwist    ::MvNormalTwisted,
    n_particles::Int
)
    @unpack K, b, c = mvtwist.twisting
    mvnormal = mvtwist.mvnormal
    μ, Σ = mean(mvnormal), cov(mvnormal)
    rand(rng, MvNormal(K*(Σ\μ - b), K), n_particles)
end


function logmarginal(mvtwist::MvNormalTwisted)
    @unpack K, b, c = mvtwist.twisting
    mvnormal = mvtwist.mvnormal
    μ, Σ  = mean(mvnormal), cov(mvnormal)
    Σinvμ = Σ\μ
    z     = Σinvμ - b
    (
        -logdet(Σ)
        + logdet(K)
        + dot(z, K*z)
        - dot(μ, Σinvμ)
        - c
     )/2
end

function mutate_ula_twisted(
    rng, q, h, Γ, Γchol
)
    q + h*(Γchol*randn(rng, length(q)))
    #randn_twisted(rng, twisting, q′, Γ, h), q′ 
end

function logmarginal_ula_twisted(twisting, q, Γ, h, logπincr)
    @unpack K, A, b, c = twisting
    Γinvq = Γ\q 
    z     = Γinvq - h*b
    (
        -logdet(Γ)
        + logdet(K)
        + dot(z, K*z)/h
        - dot(q, Γinvq)/h
        - c
        + logπincr
     )/2
end

function optimize_policy(states)
    n_states = length(states)
    for state in reverse(last(states, n_states))
         
    end
    state = last(states)
    G0    = 
    M1ψ   = 

    A, b, c = quadratic_regression(state.particles, G0.*M1ψ)
end

    policy = vcat(
        QuadTwisting(cov(proposal), Zeros(d,d), Zeros(d), 0.0),
        fill(QuadTwisting(Eye(d), Zeros(d,d), Zeros(d), 0.0), length(schedule)-1)
    )
