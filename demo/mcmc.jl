
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
