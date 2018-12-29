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

import LinearAlgebra.generic_matvecmul!

const gemv_mt_threshold = Ref(64.0)
mt_thresholds[:gemv] = gemv_mt_threshold;

function generic_matvecmul!(C::AbstractVector{DoubleFloat{T}}, tA, A::AbstractVecOrMat{DoubleFloat{T}}, B::AbstractVector{DoubleFloat{T}}) where {T <: AbstractFloat}
    has_offset_axes(C, A, B) && throw(ArgumentError("offset axes are not supported"))
    mB = length(B)
    mA, nA = lapack_size(tA, A)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"))
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end
    use_threads = (nthreads() > 1) &&
        (Float64(mB)*Float64(mA) > gemv_mt_threshold[])

    if (tA == 'T') || (tA == 'C')
        if use_threads
            liA = LinearIndices(A)
            @threads for k = 1:mA
                @inbounds C[k] = _dot(nA,A,liA[1,k],B,1,Vec{Npref,T})
            end
        else
            @inbounds for k = 1:mA
                C[k] = dot(view(A,:,k),B)
            end
        end
    else
        #=
        You may ask why there's no MT version here.
        All my attempts so far have encountered some bizarre race condition
        that fails tests in weird ways.
        No, I don't mean the obvious race if one stupidly tries to thread
        the i loop - it's the sensible alternatives that SOMETIMES fail.
        So if you are reading this and can do better, please file a PR.
        =#
        fill!(C,zero(T))
        for i = 1:nA
            axpy!(B[i],view(A,:,i),C)
        end
    end
    C
end
