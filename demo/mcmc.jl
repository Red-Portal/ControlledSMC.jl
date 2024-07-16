
function euler_fwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x + h/2*Γ*∇logπt
end

function euler_bwd(logtarget, x, h, Γ)
    ∇logπt = ForwardDiff.gradient(logtarget, x)
    x - h/2*Γ*∇logπt
end

function leapfrog(logtarget, q, p, δ, M)
    n∇U = mapslices(Base.Fix1(ForwardDiff.gradient, logtarget), q, dims=1)
    p′   = p + δ/2*n∇U
    q′   = q + δ*(M\p′)
    n∇U′ = mapslices(Base.Fix1(ForwardDiff.gradient, logtarget), q′, dims=1)
    p′′   = p′ + δ/2*n∇U′
    q′, p′′
end

function klmc_cov(d, h, γ, σ2)
    # Durmus, Enfroy, Moulines, Stolts use h = γ, γ = κ

    η    = exp(-γ*h)
    σ2xx = σ2/(2*γ^2)*(2*h - (3 - 4*η + η^2)/γ)
    σ2xv = σ2/(2*γ^2)*(1 - η)^2
    σ2vv = σ2/(2*γ)*(1 - η^2)
    Σ    = zeros(2*d, 2*d)

    @inbounds for i in 1:d
        Σ[i,i] = σ2xx
    end
    @inbounds for i in d+1:2*d
        Σ[i,i] = σ2vv
    end
    @inbounds for i in 1:d
        Σ[i+d,i] = σ2xv
        Σ[i,i+d] = σ2xv
    end
    Hermitian(Σ)
end

function klmc_fwd(logtarget, x, v, h, γ)
    ∇U = -mapslices(Base.Fix1(ForwardDiff.gradient, logtarget), x, dims=1)
    η  = exp(-γ*h)

    μ_x = x + (1 - η)/γ*v - 1/(γ*γ)*(h*γ + η - 1)*∇U
    μ_v = η*v - (1 - η)/γ*∇U
    vcat(μ_x, μ_v)
end
