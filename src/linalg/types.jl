
struct BatchVectors2{
    B1 <: AbstractMatrix,
    B2 <: AbstractMatrix
}
    x1::B1
    x2::B2
end

struct BlockHermitian2by2{B}
    Σ11::B
    Σ21::B
    Σ22::B
end

function LinearAlgebra.Matrix(Σ_st::BlockHermitian2by2)
    d = size(Σ_st.Σ11, 1)
    Σ                           = zeros(2 * d, 2 * d)
    Σ[1:d, 1:d]                 = Σ_st.Σ11
    Σ[1:d, (d + 1):end]         = Σ_st.Σ21
    Σ[(d + 1):end, 1:d]         = Σ_st.Σ21
    Σ[(d + 1):end, (d + 1):end] = Σ_st.Σ22
    Σ
end

struct BlockMatrix2by2{B}
    A11::B
    A21::B
    A12::B
    A22::B
end

function LinearAlgebra.Matrix(A_st::BlockMatrix2by2)
    d = size(A_st.A11, 1)
    A                           = zeros(2 * d, 2 * d)
    A[1:d, 1:d]                 = A_st.A11
    A[1:d, (d + 1):end]         = A_st.A12
    A[(d + 1):end, 1:d]         = A_st.A21
    A[(d + 1):end, (d + 1):end] = A_st.A22
    A
end

struct BlockLowerTriangular2by2{B}
    L11::B
    L21::B
    L22::B
end

function LinearAlgebra.Matrix(L_st::BlockLowerTriangular2by2)
    d = size(L_st.L11, 1)
    L                           = zeros(2 * d, 2 * d)
    L[1:d, 1:d]                 = L_st.L11
    L[(d + 1):end, 1:d]         = L_st.L21
    L[(d + 1):end, (d + 1):end] = L_st.L22
    L
end

struct BlockDiagonal2by2{B}
    D1::B
    D2::B
end

Base.show(
    io::IO,
    Σ::Union{
        <:BlockHermitian2by2,
        <:BlockMatrix2by2,
        <:BlockLowerTriangular2by2,
    }
) = Base.show(io, Matrix(Σ))
