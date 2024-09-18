
struct BlockHermitian2by2{B}
    Σ11::B
    Σ21::B
    Σ22::B
end

struct Block2by2{B}
    A11::B
    A21::B
    A12::B
    A22::B
end

struct BlockDiagonal2by2{B}
    D1::B
    D2::B
end

struct BlockCholesky2by2{B}
    L11::B
    L21::B
    L22::B
end

function LinearAlgebra.cholesky(Σ::BlockHermitian2by2)
    Σ11, Σ12, Σ22  = Σ.Σ11, Σ.Σ21, Σ.Σ22

    L11 = sqrt(Σ11)
    L12 = Σ12 / L11
    L22 = sqrt(Σ11 * Σ22 - Σ12 * Σ12) / L11
    return BlockCholesky2by2{typeof(L11)}(L11, L12, L22)
end

function LinearAlgebra.inv(L::BlockCholesky2by2)
    L11, L21, L22  = L.L11, L.L21, L.L22

    Linv11 = inv(L11)
    Linv12 = -L21 / L11 / L22
    Linv22 = inv(L22)
    return BlockCholesky2by2{typeof(L11)}(Linv11, Linv12, Linv22)
end

function Base.sqrt(D::BlockDiagonal2by2{<:Diagonal})
    BlockDiagonal2by2(sqrt(D.D1), sqrt(D.D2))
end

function LinearAlgebra.:*(D::BlockDiagonal2by2{<:Diagonal}, L::BlockCholesky2by2{<:Diagonal})
    BlockCholesky2by2(
        D.D1*L.L11,
        D.D2*L.L21,
        D.D2*L.L22,
    )
end

function LinearAlgebra.:*(D::BlockDiagonal2by2{<:Diagonal}, Σ::BlockHermitian2by2{<:Diagonal})
    Block2by2(
        D.D1*Σ.Σ11,
        D.D2*Σ.Σ21,
        D.D1*Σ.Σ21,
        D.D2*Σ.Σ22,
    )
end

function LinearAlgebra.:*(L::BlockCholesky2by2{<:Diagonal}, A::Block2by2{<:Diagonal})
    # [ L11   0 ] [ A11 A12 ]
    # [ L21 L22 ] [ A21 A22 ]
    # = [           L11*A11             L11*A12 ]
    #   [ L21*A11 + L22*A21   L21*A12 + L22*A22 ]
    (; L11, L21, L22)      = L
    (; A11, A12, A21, A22) = A
    Block2by2(
        L11*A11,
        L21*A11 + L22*A21,
        L11*A12,
        L21*A12 + L22*A22,
    )
end

function quad(Σ::BlockHermitian2by2{<:Diagonal}, D::BlockDiagonal2by2{<:Diagonal})
    (; D1, D2)        = D
    (; Σ11, Σ21, Σ22) = Σ
    BlockHermitian2by2(
        D1*D1*Σ11,
        D1*D2*Σ21,
        D2*D2*Σ22,
    )
end

function LinearAlgebra.:+(D::BlockDiagonal2by2{<:Diagonal}, Σ::BlockHermitian2by2{<:Diagonal})
    BlockHermitian2by2(
        Σ.Σ11 + D.D1,
        Σ.Σ21,
        Σ.Σ22 + D.D2,
    )
end

function LinearAlgebra.:-(
    Σ1::BlockHermitian2by2{<:Diagonal},
    Σ2::BlockHermitian2by2{<:Diagonal}
)
    BlockHermitian2by2(
        Σ1.Σ11 - Σ2.Σ11,
        Σ1.Σ21 - Σ2.Σ21,
        Σ1.Σ22 - Σ2.Σ22,
    )
end

function LinearAlgebra.:*(L::BlockCholesky2by2{<:Diagonal}, Σ::BlockHermitian2by2{<:Diagonal})
    # [ L11   0 ] [ Σ11 Σ21 ]
    # [ L21 L22 ] [ Σ21 Σ22 ]
    # = [           L11*Σ11             L11*Σ21 ]
    #   [ L21*Σ11 + L22*Σ21   L21*Σ21 + L22*Σ22 ]
    Block2by2(
        L.L11*Σ.Σ11,
        L.L11*Σ.Σ21,
        L.L21*Σ.Σ11 + L.L22*Σ.Σ21,
        L.L21*Σ.Σ21 + L.L22*Σ.Σ22,
    )
end

function transpose_square(
    A::Block2by2{<:Diagonal},
)
    # [A11 A21] × [A11 A12]
    # [A12 A22]   [A21 A22]
    #
    # = [A11*A11 + A21*A21  A11*A12 + A21*A22]
    #   [A12*A11 + A22*A21  A12*A12 + A22*A22]

    (; A11, A12, A21, A22) = A
    BlockHermitian2by2(
        A11*A11 + A21*A21,
        A12*A11 + A22*A21,
        A12*A12 + A22*A22
    )
end

# function LinearAlgebra.:*(α::Real, Σ::BlockHermitian2by2{<:Diagonal})
#     BlockCholesky2by2(α*Σ.Σ11, α*Σ.Σ21, α*Σ.Σ22)
# end

# function LinearAlgebra.:+(L1::BlockCholesky2by2{<:Diagonal}, L2::BlockCholesky2by2{<:Diagonal})
#     BlockCholesky2by2(
#         L1.L11 + L2.L11,
#         L1.L21 + L2.L21,
#         L1.L22 + L2.L22,
#     )
# end

# function LinearAlgebra.:+(D::BlockDiagonal2by2{<:Diagonal}, L::BlockCholesky2by2{<:Diagonal})
#     BlockCholesky2by2(
#         D.D1 + L.L11,
#         L.L21,
#         D.D2 + L.L22,
#     )
# end


# function transpose_square(
#     L::BlockCholesky2by2{<:Diagonal},
# )
#     # [U11 U12] × [L11   0]
#     # [  0 U22]   [L21 L22]
#     #
#     # = [U11×L11 + U12×L21  U12×L22]
#     #   [          U22×L21  U22×L22]
#     #
#     # = [L11^2 + L12^2  L21×L22]
#     #   [      L22×L21    L22^2]

#     (; L11, L21, L22) = L
#     BlockCholesky2by2(L11*L11 + L21*L21, L21*L22, L22*L22)
# end
