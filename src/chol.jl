import LinearAlgebra._chol!
using LinearAlgebra: checksquare

const chol_mt_threshold = Ref(64.0)
mt_thresholds[:chol] = chol_mt_threshold;

function _chol!(A::AbstractMatrix{DoubleFloat{T}}, ::Type{UpperTriangular}) where T
    @assert !has_offset_axes(A)
    n = checksquare(A)
    use_threads = (nthreads() > 1) &&
        (Float64(n)^2 > chol_mt_threshold[])
    @inbounds begin
        for k = 1:n
            #for i = 1:k - 1
            #    A[k,k] -= A[i,k]'A[i,k]
            #end
            vk = view(A,1:k-1,k)
            A[k,k] -= dot(vk,vk)
            Akk, info = _chol!(A[k,k], UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            if use_threads
                _mt_uchol_loop(n,k,A,AkkInv,Vec{Npref,T})
            else
                for j = k + 1:n
                    #for i = 1:k - 1
                    #    A[k,j] -= A[i,k]'A[i,j]
                    #end
                    A[k,j] -= dot(view(A,1:k-1,k),view(A,1:k-1,j))
                    A[k,j] = AkkInv*A[k,j]
                end
            end
        end
    end
    return UpperTriangular(A), convert(BlasInt, 0)
end

# function barrier helps inference (even for non-threaded path)
function _mt_uchol_loop(n,k,A,AkkInv,VT)
    liA = LinearIndices(A)
    @threads for j = k + 1:n
        # A[k,j] -= dot(view(A,1:k-1,k),view(A,1:k-1,j))
        @inbounds A[k,j] -= _dot(k-1,A,liA[1,k],A,liA[1,j],VT)
        @inbounds A[k,j] = AkkInv*A[k,j]
    end
end

function _chol!(A::AbstractMatrix{DoubleFloat{T}}, ::Type{LowerTriangular}) where T
    @assert !has_offset_axes(A)
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
                    _mt_lchol_loop(n,k,A,AkkInv,Vec{Npref,T})
                else
                    for j = 1:k - 1
                        #@simd for i = k + 1:n
                        #    A[i,k] -= A[i,j]*A[k,j]'
                        #end
                        axpy!(-A[k,j]',view(A,k+1:n,j),view(A,k+1:n,k))
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

function _mt_lchol_loop(n,k,A,AkkInv,VT)
    liA = LinearIndices(A)
    @threads for i = k + 1:n
        # for j = 1:k - 1
            # A[i,k] -= A[i,j]*A[k,j]'
        # end
        @inbounds A[i,k] -= dot(view(A,k,1:k-1),view(A,i,1:k-1))
    end
    @threads for i = k + 1:n
        @inbounds A[i,k] *= AkkInv'
    end
end
