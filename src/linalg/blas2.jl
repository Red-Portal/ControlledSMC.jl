
function LinearAlgebra.:*(
    L::BlockLowerTriangular2by2,
    x::BatchVectors2,
)
    (; L11, L21, L22) = L
    (; x1, x2) = x
    @assert size(L11, 2) == size(x1, 1)
    @assert size(L21, 2) == size(x1, 1)
    @assert size(L22, 2) == size(x2, 1)
    BatchVectors2(
        L11*x1,
        L21*x1 + L22*x2
    )
end

function LinearAlgebra.:*(
    Σ::BlockHermitian2by2,
    x::BatchVectors2,
)
    (; Σ11, Σ21, Σ22) = Σ
    (; x1, x2) = x
    @assert size(Σ11, 2) == size(x1, 1)
    @assert size(Σ21, 2) == size(x1, 1)
    @assert size(Σ22, 2) == size(x2, 1)
    BatchVectors2(
        Σ11*x1 + Σ21*x2,
        Σ21*x1 + Σ22*x2
    )
end

function LinearAlgebra.:+(
    D::BlockDiagonal2by2,
    Σ::BlockHermitian2by2
)
    (; Σ11, Σ21, Σ22) = Σ
    (; D1, D2)        = D
    BlockHermitian2by2(
        Σ11 + D1,
        Σ21,
        Σ22 + D2,
    )
end

function LinearAlgebra.:*(α::Real, D::BlockDiagonal2by2)
    BlockDiagonal2by2(α*D.D1, α*D.D2)
end
