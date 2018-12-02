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
    liA = LinearIndices(A.data)
    liB = LinearIndices(B)
    use_threads = (nthreads() > 1) &&
        (Float64(m)*Float64(n) > trs_mt_threshold[])

    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
    end
    @inbounds for j in n:-1:1
        if use_threads
            _mt_ldiv_loop1(m,j,A.data,liA[1,j],B,liB,Vec{Npref,T})
        else
            for k = 1:m
                xj = B[j,k] = A.data[j,j] \ B[j,k]
                # _axpy!(-xj,view(A.data,1:j-1,j),view(B,1:j-1,k),Vec{Npref,T})
                if j>1
                    _axpy!(j-1,-xj,A.data,liA[1,j],B,liB[1,k],Vec{Npref,T})
                end
            end
        end
    end
    B
end

# We attempt to insure good inference by use of function barriers.
# We may be overdoing it, based on questionable  profiling results.
# (The tools used are driven beserk by threading)
# FIXME: Revisit when threading has been overhauled.

@noinline function _mt_ldiv_loop1(m,j,A,idxA,B,liB,VT)
    if j>1
        @threads for k = 1:m
            @inbounds _mt_ldiv_kern1(j,k,A,B,idxA,liB[1,k],VT)
        end
    else
        @threads for k = 1:m
            @inbounds B[j,k] = A[j,j] \ B[j,k]
        end
    end
    #=
    @threads for k = 1:m
        @inbounds begin
            xj = B[j,k] = A[j,j] \ B[j,k]
            # _axpy!(-xj,view(A,1:j-1,j),view(B,1:j-1,k),VT)
            if j>1
                _axpy!(j-1,-xj,A,liA[1,j],B,liB[1,k],VT)
            end
        end
    end
    =#
    nothing
end

@noinline function _mt_ldiv_kern1(j,k,A,B,idxA,idxB,VT)
    @inbounds begin
        xj = B[j,k] = A[j,j] \ B[j,k]
        # _axpy!(-xj,view(A,1:j-1,j),view(B,1:j-1,k),VT)
        _axpy!(j-1,-xj,A,idxA,B,idxB,VT)
    end
    nothing
end

function ldiv!(A::UnitLowerTriangular{DoubleFloat{T}},
                   B::AbstractMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, B)
    n = size(A.data, 2)
    m = size(B, 2)
    if !(n == size(B,1))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and first dimension of right hand side B, $(size(B,1)), must be equal"))
    end
    liA = LinearIndices(A.data)
    liB = LinearIndices(B)
    use_threads = (nthreads() > 1) &&
        (Float64(m)*Float64(n) > trs_mt_threshold[])
    @inbounds for j in 1:n-1
        if use_threads
            _mt_ldiv_loop2(m,n,j,A.data,liA[j+1,j],B,liB,Vec{Npref,T})
        else
            for k = 1:m
                xj = B[j,k]
                # _axpy!(-xj, view(A.data,j+1:n,j), view(B,j+1:n,k), Vec{Npref,T})
                if j<n
                    _axpy!(n-j,-xj,A.data,liA[j+1,j],B,liB[j+1,k],Vec{Npref,T})
                end
            end
        end
    end
    B
end

@noinline function _mt_ldiv_loop2(m,n,j,A,idxA,B,liB,VT)
    @threads for k = 1:m
        @inbounds _mt_ldiv_kern2(n,j,k,A,B,idxA,liB[j+1,k],VT)
        #=
        @inbounds begin
            xj = B[j,k]
            # _axpy!(-xj, view(A,j+1:n,j), view(B,j+1:n,k), VT)
            if j<n
                _axpy!(n-j,-xj,A,liA[j+1,j],B,liB[j+1,k],VT)
            end
        end
        =#
    end
    nothing
end

@noinline function _mt_ldiv_kern2(n,j,k,A,B,idxA,idxB,VT)
    @inbounds begin
        xj = B[j,k]
        # _axpy!(-xj, view(A,j+1:n,j), view(B,j+1:n,k), VT)
        if j<n
            _axpy!(n-j,-xj,A,idxA,B,idxB,VT)
        end
    end
    nothing
end

function naivesub!(A::UpperTriangular{DoubleFloat{T}},
                   b::AbstractVector{DoubleFloat{T}},
                   x::AbstractVector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, b, x)
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    liA = LinearIndices(A.data)
    liB = LinearIndices(b)
    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
        xj = x[j] = A.data[j,j] \ b[j]
        #for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
        #    b[i] -= A.data[i,j] * xj
        #end
        # _axpy!(-xj,view(A.data,1:j-1,j),view(b,1:j-1),Vec{Npref,T})
        if j>1
            _axpy!(j-1,-xj,A.data,liA[1,j],b,liB[1],Vec{Npref,T})
        end
    end
    x
end

function naivesub!(A::UnitLowerTriangular{DoubleFloat{T}},
                   b::StridedVector{DoubleFloat{T}},
                   x::AbstractVector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    @assert !has_offset_axes(A, b, x)
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    liA = LinearIndices(A.data)
    liB = LinearIndices(b)
    @inbounds for j in 1:n
        xj = x[j] = b[j]
        # for i in j+1:n
        #     b[i] -= A.data[i,j] * xj
        # end
        # _axpy!(-xj, view(A.data,j+1:n,j), view(b,j+1:n), Vec{Npref,T})
        if j<n
            _axpy!(n-j,-xj,A.data,liA[j+1,j],b,liB[j+1],Vec{Npref,T})
        end
    end
    x
end
