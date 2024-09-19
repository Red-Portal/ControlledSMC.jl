
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
