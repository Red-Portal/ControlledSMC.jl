
function LinearAlgebra.:+(x1::BatchVectors2, x2::BatchVectors2)
    @assert size(x1.x1) == size(x2.x1)
    @assert size(x1.x2) == size(x2.x2)
    return BatchVectors2(x1.x1 + x2.x1, x1.x2 + x2.x2)
end

function LinearAlgebra.:-(x1::BatchVectors2, x2::BatchVectors2)
    @assert size(x1.x1) == size(x2.x1)
    @assert size(x1.x2) == size(x2.x2)
    return BatchVectors2(x1.x1 - x2.x1, x1.x2 - x2.x2)
end
