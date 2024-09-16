
function rand_twist_mvnormal(
    rng    ::Random.AbstractRNG,
    twist,
    μs     ::AbstractMatrix,
    Σ      ::Union{<:Diagonal, <:FillArrays.Eye, <:PDMats.ScalMat},
)
    (; a, b,)  = twist
    A          = Diagonal(a)
    K          = Diagonal(inv(2*A + inv(Σ)))
    μs_twisted = K*(Σ\μs .- b)
    μs_twisted + unwhiten(K, randn(rng, size(μs)))
end

function twist_mvnormal_logmarginal(
    twist,
    μ     ::AbstractArray,
    Σ     ::Union{<:Diagonal, <:FillArrays.Eye, <:PDMats.ScalMat},
)
    (; a, b, c)  = twist
    A     = Diagonal(a)
    K     = Diagonal(inv(2*A + inv(Σ)))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z     = Σ\μ .- b
    ((-ℓdetΣ + ℓdetK) .+ (quad(K, z) - invquad(Σ, μ)))/2 .- c
end

# function rand_twist_mvnormal(
#     rng    ::Random.AbstractRNG,
#     twist,
#     μs     ::AbstractMatrix,
#     Σ      ::AbstractMatrix,
# )
#     (; a, b,)  = twist
#     K          = inv(2*Diagonal(a) + inv(PDMats.PDMat(Σ)))
#     μs_twisted = K*(Σ\μs .- b)
#     μs_twisted + unwhiten(K, randn(rng, size(μs)))
# end

# function twist_mvnormal_logmarginal(
#     twist,
#     μ     ::AbstractArray,
#     Σ     ::AbstractMatrix,
# )
#     (; a, b, c)  = twist
#     K     = inv(2*Diagonal(a) + inv(PDMats.PDMat(Σ)))
#     ℓdetΣ = logdet(Σ)
#     ℓdetK = logdet(K)
#     z     = Σ\μ .- b
#     ((-ℓdetΣ + ℓdetK) .+ (quad(K, z) - invquad(Σ, μ)))/2 .- c
# end

function twist_logdensity(twist, x)
    (; a, b, c) = twist
    -quad(Diagonal(a), x) - (b'*x)[1,:] .- c   
end

