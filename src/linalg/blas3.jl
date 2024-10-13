
function transpose_square(L::BlockLowerTriangular2by2{<:Diagonal})
    # [L11 L21] × [L11    ]
    # [    L22]   [L21 L22]
    #
    # = [L11*L11 + L21*L21  L21*L22]
    #   [          A22*A21  L22*L22]

    (; L11, L21, L22) = L
    return BlockHermitian2by2(L11 * L11 + L21 * L21, L21 * L22, L22 * L22)
end
