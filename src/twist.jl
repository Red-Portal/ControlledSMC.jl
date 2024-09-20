
function twist_mvnormal_rand(
    rng::Random.AbstractRNG,
    twist,
    μs::AbstractMatrix,
    Σ::Union{<:Diagonal,<:FillArrays.Eye,<:PDMats.ScalMat},
)
    (; a, b)   = twist
    A          = Diagonal(a)
    K          = Diagonal(inv(2 * A + inv(Σ)))
    μs_twisted = K * (Σ \ μs .- b)
    return μs_twisted + unwhiten(K, randn(rng, size(μs)))
end

function twist_mvnormal_logmarginal(
    twist, μ::AbstractArray, Σ::Union{<:Diagonal,<:FillArrays.Eye,<:PDMats.ScalMat}
)
    (; a, b, c) = twist
    A = Diagonal(a)
    K = Diagonal(inv(2 * A + inv(Σ)))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z = Σ \ μ .- b
    return ((-ℓdetΣ + ℓdetK) .+ (PDMats.quad(K, z) - PDMats.invquad(Σ, μ))) / 2 .- c
end

function control_cov(A::BlockDiagonal2by2, Σ::BlockPDMat2by2)
    (; Linv)  = Σ
    Σinv      = transpose_square(Linv)
    Kinv      = 2 * A + Σinv
    L_Kinv    = cholesky(Kinv)
    Linv_Kinv = inv(L_Kinv)
    Kinv      = transpose_square(Linv_Kinv)
    return PDMats.PDMat(Kinv)
end

function twist_mvnormal_rand(
    rng::Random.AbstractRNG,
    twist,
    μs::BatchVectors2,
    Σ::BlockPDMat2by2,
)
    (; a, b)   = twist
    d = length(a) ÷ 2
    n = size(μs.x1, 2)
    b = BatchVectors2(repeat(b[1:d], 1, n), repeat(b[d+1:2*d], 1, n))
    A = BlockDiagonal2by2(Diagonal(a[1:d]), Diagonal(a[d+1:2*d]))
    K = control_cov(A, Σ)

    μs_twist = K.Σ * ((Σ \ μs) - b)
    z_st     = rand(rng, BatchMvNormal(μs_twist, K))
    vcat(z_st.x1, z_st.x2)
end

function twist_mvnormal_logmarginal(
    twist,
    μs::BatchVectors2,
    Σ::BlockPDMat2by2,
)
    (; a, b, c) = twist
    d     = length(a) ÷ 2
    n     = size(μs.x1, 2)
    A     = BlockDiagonal2by2(Diagonal(a[1:d]), Diagonal(a[d+1:2*d]))
    b     = BatchVectors2(repeat(b[1:d], 1, n), repeat(b[d+1:2*d], 1, n))
    K     = control_cov(A, Σ)
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z     = (Σ \ μs) - b
    return ((-ℓdetΣ + ℓdetK) .+ (PDMats.quad(K, z) - PDMats.invquad(Σ, μs))) / 2 .- c
end

function twist_logdensity(twist, x)
    (; a, b, c) = twist
    return -PDMats.quad(Diagonal(a), x) - (b' * x)[1, :] .- c
end
