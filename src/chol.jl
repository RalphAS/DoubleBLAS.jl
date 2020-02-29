import LinearAlgebra._chol!
using LinearAlgebra: checksquare

const chol_mt_threshold = Ref(64.0)
mt_thresholds[:chol] = chol_mt_threshold;
const chol_dot_threshold = Ref(48)

function _chol!(A::AbstractMatrix{DT}, ::Type{UpperTriangular}
      ) where {DT <: Union{DoubleFloat{T},Complex{DoubleFloat{T}}}} where T
    require_one_based_indexing(A)
    n = checksquare(A)
    use_threads = (nthreads() > 1) &&
        (Float64(n)^2 > chol_mt_threshold[])
    use_views = n > chol_dot_threshold[]
    for k = 1:n
        @inbounds begin
            if use_views
                vk = uview(A,1:k-1,k)
                A[k,k] -= dot(vk,vk)
            else
                for i = 1:k - 1
                    A[k,k] -= A[i,k]'A[i,k]
                end
            end
            Akk, info = _chol!(A[k,k], UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            if use_threads
                _mt_uchol_loop(n,k,A,AkkInv,Vec{Npref,T})
            else
                if n < chol_dot_threshold[]
                    for j = k + 1:n
                        for i = 1:k - 1
                            A[k,j] -= A[i,k]'A[i,j]
                        end
                        A[k,j] = AkkInv*A[k,j]
                    end
                else
                    liA = LinearIndices(A)
                    for j = k + 1:n
                    #for i = 1:k - 1
                    #    A[k,j] -= A[i,k]'A[i,j]
                    #end
                    # uview should work here since A isn't yet UT
                    # but it is currently not inferred
                    # @inbounds A[k,j] -= dot(uview(A,1:k-1,k),uview(A,1:k-1,j))
                        @inbounds A[k,j] -= _dot(k-1,A,liA[1,k],A,liA[1,j],Vec{Npref,T})
                        A[k,j] = AkkInv*A[k,j]
                    end
                end
            end
        end
    end
    return UpperTriangular(A), convert(BlasInt, 0)
end

# function barrier helps inference (even for non-threaded path)
function _mt_uchol_loop(n,k,A,AkkInv,::Type{Vec{N,T}}) where {N,T}
    liA = LinearIndices(A)
    @threads for j = k + 1:n
#        @inbounds A[k,j] -= dot(uview(A,1:k-1,k),uview(A,1:k-1,j))
        @inbounds A[k,j] -= _dot(k-1,A,liA[1,k],A,liA[1,j],Vec{N,T})
        @inbounds A[k,j] = AkkInv*A[k,j]
    end
end

function _chol!(A::AbstractMatrix{DoubleFloat{T}}, ::Type{LowerTriangular}) where T
    require_one_based_indexing(A)
    n = checksquare(A)
    use_threads = (nthreads() > 1) &&
        (Float64(n)^2 > chol_mt_threshold[])
    @inbounds begin
        for k = 1:n
            #for i = 1:k - 1
            #    A[k,k] -= A[k,i]*A[k,i]'
            #end
            vk = view(A,k,1:k-1)
            A[k,k] -= dot(vk,vk)
            Akk, info = _chol!(A[k,k], LowerTriangular)
            if info != 0
                return LowerTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(Akk)
            if k<n
                if use_threads
                    _mt_lchol_loop(n,k,A,AkkInv)
                else
                    for j = 1:k - 1
                        #@simd for i = k + 1:n
                        #    A[i,k] -= A[i,j]*A[k,j]'
                        #end
                        axpy!(-A[k,j]',uview(A,k+1:n,j),uview(A,k+1:n,k))
                    end
                    for i = k + 1:n
                        A[i,k] *= AkkInv'
                    end
                end
            end
        end
     end
    return LowerTriangular(A), convert(BlasInt, 0)
end

# can't just thread the j-loop because of dependence
#    @threads for j = 1:k - 1
#        @inbounds _axpy!(n-k,-A[k,j]',A,liA[k+1,j],A,liA[k+1,k],VT)
#    end

function _mt_lchol_loop(n,k,A,AkkInv)
    # liA = LinearIndices(A)
    @threads for i = k + 1:n
        # for j = 1:k - 1
            # A[i,k] -= A[i,j]*A[k,j]'
        # end
        # can't use LI until we have strided version of dot
        @inbounds A[i,k] -= dot(uview(A,k,1:k-1),uview(A,i,1:k-1))
    end
    @threads for i = k + 1:n
        @inbounds A[i,k] *= AkkInv'
    end
end
