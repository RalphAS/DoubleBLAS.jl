import LinearAlgebra.generic_lufact!

const lu_mt_threshold = Ref(64.0)
mt_thresholds[:lu] = lu_mt_threshold;

function generic_lufact!(A::StridedMatrix{DoubleFloat{T}}, ::Val{Pivot} = Val(true);
                         check::Bool = true) where {T<:AbstractFloat,Pivot}
    m, n = size(A)
    minmn = min(m,n)
    info = 0
    ipiv = Vector{BlasInt}(undef, minmn)
    use_threads = (nthreads() > 1) &&
        (Float64(m)*Float64(n) > lu_mt_threshold[])
    liA = LinearIndices(A)

    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if Pivot
                amax = abs(zero(T))
                for i = k:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            ipiv[k] = kp
            if !iszero(A[kp,k])
                if k != kp
                    # Interchange
                    for i = 1:n
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end
            # Update the rest
            if use_threads
                _mt_lu_loop(m,n,k,A,Vec{Npref,T})
            else
                for j = k+1:n
                    #axpy!(-A[k,j],view(A,k+1:m,k),view(A,k+1:m,j),Vec{Npref,T})
                    _axpy!(m-k,-A[k,j],A,liA[k+1,k],A,liA[k+1,j],Vec{Npref,T})
                end
            end
        end
    end
    check && checknonsingular(info)
    return LU{DoubleFloat{T},typeof(A)}(A, ipiv, convert(BlasInt, info))
end

# Someday @threads will not stupefy the inference engine.

@noinline function _mt_lu_loop(m,n,k,A,VT)
    liA = LinearIndices(A)
    @threads for j = k+1:n
        @inbounds begin
            # axpy!(-A[k,j],view(A,k+1:m,k),view(A,k+1:m,j),VT)
            _axpy!(m-k,-A[k,j],A,liA[k+1,k],A,liA[k+1,j],VT)
        end
    end
end
