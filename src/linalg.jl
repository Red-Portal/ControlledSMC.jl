
struct BlockHermitian2by2{B}
    Σ11::B
    Σ21::B
    Σ22::B
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

function cholesky2by2(Σ::BlockHermitian2by2)
    Σ11, Σ12, Σ22  = Σ.Σ11, Σ.Σ21, Σ.Σ22

    L11 = sqrt(Σ11)
    L12 = Σ12 / L11
    L22 = sqrt(Σ11 * Σ22 - Σ12 * Σ12) / L11
    return BlockCholesky2by2{typeof(L11)}(L11, L12, L22)
end

function inv2by2(L::BlockCholesky2by2)
    L11, L21, L22  = L.L11, L.L21, L.L22

    Linv11 = inv(L11)
    Linv12 = -L21 / L11 / L22
    Linv22 = inv(L22)
    return BlockCholesky2by2{typeof(L11)}(Linv11, Linv12, Linv22)
end
