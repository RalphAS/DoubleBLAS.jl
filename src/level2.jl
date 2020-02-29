import LinearAlgebra.reflectorApply!

# this should suffice to handle simple qr()

const qr_mt_threshold = Ref(64.0)
mt_thresholds[:qr] = qr_mt_threshold;
function reflectorApply!(x::AbstractVector{DT}, τ::Number, A::StridedMatrix{DT}
                         ) where {DT <: Union{DoubleFloat{T},
                                              Complex{DoubleFloat{T}}}} where T
    require_one_based_indexing(x)
    m, n = size(A)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), "
               * "which must match the first dimension of matrix A, $m"))
    end
    use_threads = (nthreads() > 1) &&
            (Float64(m)*Float64(n) > qr_mt_threshold[])
    if use_threads
        @threads for j = 1:n
            _mt_refl_loop1(τ,A,x,m,j)
        end
    else
        @inbounds begin
            for j = 1:n
                vAj = conj(τ)*(A[1, j] + dot(view(x,2:m), view(A,2:m,j)))
                A[1, j] -= vAj
                axpy!( -vAj, view(x,2:m), view(A,2:m,j))
                # for i = 2:m
                #    A[i,j] -= vAj * x[i]
                # end
            end
        end
    end
    return A
end
function _mt_refl_loop1(τ,A,x,m,j)
    @inbounds begin
        vAj = conj(τ)*(A[1, j] + dot(uview(x,2:m), uview(A,2:m,j)))
        A[1, j] -= vAj
        # axpy!( -vAj, uview(x,2:m), uview(A,2:m,j))
        for i = 2:m
            A[i,j] -= vAj * x[i]
        end
    end
end

const GEMV_WORKS = true
if GEMV_WORKS
import LinearAlgebra.generic_matvecmul!

const gemv_mt_threshold = Ref(512.0)
mt_thresholds[:gemv] = gemv_mt_threshold;
const gemtv_mt_threshold = Ref(64.0)
mt_thresholds[:gemtv] = gemtv_mt_threshold;

function generic_matvecmul!(C::AbstractVector{DoubleFloat{T}}, tA, A::AbstractVecOrMat{DoubleFloat{T}}, B::AbstractVector{DoubleFloat{T}}) where {T <: AbstractFloat}
    require_one_based_indexing(C, A, B)
    mB = length(B)
    mA, nA = lapack_size(tA, A)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"))
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end
    if (tA == 'T') || (tA == 'C')
        use_threads = (nthreads() > 1) &&
            (Float64(mB)*Float64(mA) > gemtv_mt_threshold[])

        if use_threads
            liA = LinearIndices(A)
            @threads for k = 1:mA
                @inbounds C[k] = _dot(nA,A,liA[1,k],B,1,Vec{Npref,T})
            end
        else
            @inbounds for k = 1:mA
                C[k] = dot(uview(A,:,k),B)
            end
        end
    else
        fill!(C,zero(T))
        use_threads = (nthreads() > 1) &&
            (Float64(mB)*Float64(mA) > gemv_mt_threshold[])

        if use_threads
            nt = nthreads()
            nd,nr = divrem(mA, nt)
            liA = LinearIndices(A)
            @threads for it=1:nt
            #    for it=1:nt # DEBUG
                j1 = it*nd
                j0 = j1-nd+1
                _mt_gemv_loop1(nA,A,liA,B,C,j0,j1)
            end

            if nr > 0
                j0 = nt*nd+1
                @inbounds for i = 1:nA
                    _axpy!(nr,B[i],A,liA[j0,i],C,j0,Vec{Npref,T})
                end
            end
        else
            @inbounds begin
                astride = stride(A,2)
                for k=1:mB
                    ioff = (k-1)*astride
                    b = B[k]
                    for i = 1:mA
                        C[i] += A[ioff + i] * b
                    end
                end
            end
#            liA = LinearIndices(A)
#            @inbounds for i = 1:nA
#                _axpy!(mA,B[i],A,liA[1,i],C,1,Vec{Npref,T})
#            end
        end
    end
    C
end

function _mt_gemv_loop1(nA,A::AbstractVecOrMat{DoubleFloat{T}},liA,
                        B,C,j0,j1) where {T}
# Julia v1.0+ compiler doesn't need help here
#    nd = j1-j0+1
#    @inbounds for i = 1:nA
#         _axpy!(nd,B[i],A,liA[j0,i],C,j0,Vec{Npref,T})
#    end
    @inbounds for i=1:nA
        b = B[i]
        for j=j0:j1
            C[j] += b * A[j,i]
        end
    end
end
end # if GEMV_WORKS
