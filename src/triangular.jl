import LinearAlgebra.naivesub!

const trs_mt_threshold = Ref(64.0)
mt_thresholds[:trs] = trs_mt_threshold;

function ldiv!(A::UpperTriangular{DoubleFloat{T}},
                   B::AbstractMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, B)
    n = size(A.data, 2)
    m = size(B, 2)
    if !(n == size(B,1))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and first dimension of right hand side B, $(size(B,1)), must be equal"))
    end
    use_threads = (nthreads() > 1) &&
        (Float64(m)*Float64(n) > trs_mt_threshold[])

    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
    end
    @inbounds for j in n:-1:1
        if use_threads
            _mt_ldiv_loop1(m,j,A.data,B,Vec{Npref,T})
        else
            for k = 1:m
                xj = B[j,k] = A.data[j,j] \ B[j,k]
                axpy!(-xj,view(A.data,1:j-1,j),view(B,1:j-1,k),Vec{Npref,T})
            end
        end
    end
    B
end

@noinline function _mt_ldiv_loop1(m,j,A,B,VT)
    @threads for k = 1:m
        @inbounds begin
            xj = B[j,k] = A[j,j] \ B[j,k]
            axpy!(-xj,view(A,1:j-1,j),view(B,1:j-1,k),VT)
        end
    end
end

function ldiv!(A::UnitLowerTriangular{DoubleFloat{T}},
                   B::AbstractMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, B)
    n = size(A.data, 2)
    m = size(B, 2)
    if !(n == size(B,1))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and first dimension of right hand side B, $(size(B,1)), must be equal"))
    end
    use_threads = (nthreads() > 1) &&
        (Float64(m)*Float64(n) > trs_mt_threshold[])
    @inbounds for j in 1:n
        if use_threads
            _mt_ldiv_loop2(m,n,j,A.data,B,Vec{Npref,T})
        else
            for k = 1:m
                xj = B[j,k]
                axpy!(-xj, view(A.data,j+1:n,j), view(B,j+1:n,k), Vec{Npref,T})
            end
        end
    end
    B
end

@noinline function _mt_ldiv_loop2(m,n,j,A,B,VT)
    @threads for k = 1:m
        @inbounds begin
            xj = B[j,k]
            axpy!(-xj, view(A,j+1:n,j), view(B,j+1:n,k), VT)
        end
    end
end

function naivesub!(A::UpperTriangular{DoubleFloat{T}},
                   b::AbstractVector{DoubleFloat{T}},
                   x::AbstractVector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, b, x)
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
        xj = x[j] = A.data[j,j] \ b[j]
        #for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
        #    b[i] -= A.data[i,j] * xj
        #end
        axpy!(-xj,view(A.data,1:j-1,j),view(b,1:j-1),Vec{Npref,T})
    end
    x
end

function naivesub!(A::UnitLowerTriangular{DoubleFloat{T}},
                   b::AbstractVector{DoubleFloat{T}},
                   x::AbstractVector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, b, x)
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    @inbounds for j in 1:n
        xj = x[j] = b[j]
        # for i in j+1:n
        #     b[i] -= A.data[i,j] * xj
        # end
        axpy!(-xj, view(A.data,j+1:n,j), view(b,j+1:n), Vec{Npref,T})
    end
    x
end
