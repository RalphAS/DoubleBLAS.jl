import LinearAlgebra.reflectorApply!

# this should suffice to handle simple qr()

# TODO: multithreaded versions

function reflectorApply!(x::AbstractVector{DT}, τ::Number, A::StridedMatrix{DT}
                         ) where {DT <: Union{DoubleFloat{T},
                                             Complex{DoubleFloat{T}}}} where T
    has_offset_axes(x) && throw(ArgumentError("not implemented "
                                              * "for offset axes"))
    m, n = size(A)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), "
               * "which must match the first dimension of matrix A, $m"))
    end
    @inbounds begin
        for j = 1:n
            vAj = conj(τ)*(A[1, j] + dot(view(x,2:m), view(A,2:m,j)))
            A[1, j] -= vAj
            axpy!( -vAj, view(x,2:m), view(A,2:m,j))
        end
    end
    return A
end
