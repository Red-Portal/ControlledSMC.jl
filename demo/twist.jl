
function rand_twist_mvnormal(
    rng    ::Random.AbstractRNG,
    twist,
    μs     ::AbstractMatrix,
    σ      ::AbstractVector,
)
    (; a, b,)  = twist
    K          = Diagonal(@. 1/(2*a + 1/σ))
    Σ          = Diagonal(σ)
    μs_twisted = K*(Σ\μs .- b)
    μs_twisted + unwhiten(K, randn(rng, size(μs)))
end

function twist_mvnormal_logmarginal(
    twist,
    μ     ::AbstractArray,
    σ     ::AbstractVector
)
    (; a, b, c)  = twist
    Σ     = Diagonal(σ)
    K     = Diagonal(@. 1/(2*a + 1/σ))
    ℓdetΣ = logdet(Σ)
    ℓdetK = logdet(K)
    z     = Σ\μ .- b
    ((-ℓdetΣ + ℓdetK) .+ (quad(K, z) - invquad(Σ, μ)))/2 .- c
end

function twist_logdensity(twist, x)
    (; a, b, c) = twist
    -quad(Diagonal(a), x) - (b'*x)[1,:] .- c   
end

function fit_quadratic(x, y)
    d, n = size(x,1), size(x,2)
    @assert size(x,2) == length(y)

    X   = vcat(x.^2, x, ones(1, n))' |> Array
    β   = ones(2*d + 1)
    ϵ   = 1e-5 #eps(eltype(x))
    Xty = X'*y
    XtX = Hermitian(X'*X)

    func(β_) = sum(abs2, X*β_ - y)
    function grad!(g, β_)
        g[:] = 2*(XtX*β_ - Xty)
    end
    function hess!(H, β_)
        H[:,:] = 2*XtX
    end
    df    = TwiceDifferentiable(func, grad!, hess!, β)

    lower = vcat(fill(ϵ, d), fill(-Inf, d+1))
    upper = fill(Inf, 2*d+1)
    dfc   = TwiceDifferentiableConstraints(lower, upper)
    res   = optimize(df, dfc, β, IPNewton())

    β = Optim.minimizer(res)
    a = β[1:d]
    b = β[d+1:2*d]
    c = β[end]
    
    a, b, c, sum(abs2, X*β - y)
end
