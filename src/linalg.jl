
function cholesky2by2(Σ11, Σ12, Σ22)
    L11 = sqrt(Σ11)
    L12 = Σ12 / L11
    L22 = sqrt(Σ11 * Σ22 - Σ12 * Σ12) / L11
    L11, L12, L22
end

function inv2by2(L11, L12, L22)
    Linv11 = inv(L11)
    Linv12 = -L12 / L11 / L22
    Linv22 = inv(L22)
    Linv11, Linv12, Linv22
end
