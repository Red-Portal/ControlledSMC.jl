
struct BlockPDMat2by2{
    Sigma   <: BlockHermitian2by2,
    Chol    <: BlockLowerTriangular2by2,
    CholInv <: BlockLowerTriangular2by2,
}
    Σ    :: Sigma
    L    :: Chol
    Linv :: CholInv
end

function LinearAlgebra.cholesky(Σ::BlockHermitian2by2)
    Σ11, Σ12, Σ22  = Σ.Σ11, Σ.Σ21, Σ.Σ22
    L11 = sqrt(Σ11)
    L12 = Σ12 / L11
    L22 = sqrt(Σ11 * Σ22 - Σ12 * Σ12) / L11
    return BlockLowerTriangular2by2(L11, L12, L22)
end

function LinearAlgebra.inv(L::BlockLowerTriangular2by2)
    (; L11, L21, L22)  = L
    Linv11 = inv(L11)
    Linv12 = -L21 / L11 / L22
    Linv22 = inv(L22)
    return BlockLowerTriangular2by2(Linv11, Linv12, Linv22)
end

function PDMats.PDMat(Σ::BlockHermitian2by2)
    L    = cholesky(Σ)
    Linv = inv(L)
    return BlockPDMat2by2(Σ, L, Linv)
end

function LinearAlgebra.logdet(Σ::BlockPDMat2by2)
    L11, L22 = Σ.L.L11, Σ.L.L22
    return 2 * logdet(L11) + 2 * logdet(L22)
end

function PDMats.quad(Σ::BlockPDMat2by2, x::BatchVectors2)
    (; L11, L21, L22)  = Σ.L
    (; x1, x2)         = x 
    r21 = sum(abs2, L11*x1 + L21*x2, dims=1)[1,:]
    r22 = sum(abs2, L22*x2, dims=1)[1,:]
    r21 + r22
end

function PDMats.invquad(Σ::BlockPDMat2by2, x::BatchVectors2)
    (; L11, L21, L22) = Σ.Linv
    (; x1, x2)        = x 
    r21 = sum(abs2, L11*x1, dims=1)[1,:]
    r22 = sum(abs2, L21*x1 + L22*x2, dims=1)[1,:]
    r21 + r22
end

function LinearAlgebra.:\(Σ::BlockPDMat2by2, x::BatchVectors2)
    (; L11, L21, L22)  = Σ.Linv
    Linvx = Σ.Linv * x
    BatchVectors2(
        L11*Linvx.x1 + L21*Linvx.x2,
        L22*Linvx.x2
    )
end
