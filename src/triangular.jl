if VERSION < v"1.8"
const naivesub! = LinearAlgebra.naivesub!
else
const naivesub! = LinearAlgebra.ldiv!
end

const trs_mt_threshold = Ref(64.0)
mt_thresholds[:trs] = trs_mt_threshold;

function ldiv!(A::UpperTriangular{DoubleFloat{T}, Matrix{T}},
                   B::AbstractMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    require_one_based_indexing(A, B)
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
# We may be overdoing it, since early development was based on
# now-questionable profiling results.
# (The tools used were driven beserk by threading when I started,
# but JN seems to be fixing them these days.)
# TODO: Revisit when threading has been overhauled.
# TODO: clean up the noise that doesn't really help the compiler.

@noinline function _mt_ldiv_loop1(m,j,A,idxA,B,liB,::Type{Vec{N,T}}) where {N,T}
    if j>1
        @threads for k = 1:m
            ib0::Int = liB[1,k]
            let ib=ib0
                @inbounds _mt_ldiv_kern1(j,k,A,B,idxA,ib,Vec{N,T})
            end
        end
    else
        @threads for k = 1:m
            @inbounds B[j,k] = A[j,j] \ B[j,k]
        end
    end
    nothing
end

@noinline function _mt_ldiv_kern1(j,k,A,B::AbstractMatrix{DoubleFloat{T}},
                      idxA::Int,idxB::Int,::Type{Vec{N,T}}) where {N,T}
    @inbounds begin
        xj::DoubleFloat{T} = A[j,j] \ B[j,k]
        B[j,k] = xj
        # _axpy!(-xj,view(A,1:j-1,j),view(B,1:j-1,k),VT)
        _axpy!(j-1,-xj,A,idxA,B,idxB,Vec{N,T})
    end
    nothing
end

function ldiv!(A::UnitLowerTriangular{DoubleFloat{T},Matrix{DoubleFloat{T}}},
                   B::Matrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    require_one_based_indexing(A, B)
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
                # we know j<n from loop range
                _axpy!(n-j,-xj,A.data,liA[j+1,j],B,liB[j+1,k],Vec{Npref,T})
            end
        end
    end
    B
end

@noinline function _mt_ldiv_loop2(m,n,j,A,idxA,B,liB,::Type{Vec{N,T}}) where {N,T}
    @threads for k = 1:m
        @inbounds _mt_ldiv_kern2(n,j,k,A,B,idxA,liB[j+1,k],Vec{N,T})
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

@noinline function _mt_ldiv_kern2(n,j,k,A,B,idxA,idxB,::Type{Vec{N,T}}) where {N,T}
    @inbounds begin
        xj = B[j,k]
        # _axpy!(-xj, view(A,j+1:n,j), view(B,j+1:n,k), VT)
        if j<n
            _axpy!(n-j,-xj,A,idxA,B,idxB,Vec{N,T})
        end
    end
    nothing
end

# FIXME: also need a method for UpperTriangular{D, Adjoint{D, Matrix{T}}},...
# or (suggested) UT{D, S} where S<:Adjoint{D}
function naivesub!(A::UpperTriangular{DoubleFloat{T},Matrix{DoubleFloat{T}}},
                   b::Vector{DoubleFloat{T}},
                   x::Vector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    require_one_based_indexing(A, b, x)
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    liA = LinearIndices(A.data)
    liB = LinearIndices(b)
    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
        xj = x[j] = A.data[j,j] \ b[j]
        #    b[i] -= A.data[i,j] * xj
        #end
        # _axpy!(-xj,view(A.data,1:j-1,j),view(b,1:j-1),Vec{Npref,T})
        if j>1
            _axpy!(j-1,-xj,A.data,liA[1,j],b,liB[1],Vec{Npref,T})
        end
    end
    x
end

function naivesub!(A::UnitLowerTriangular{DoubleFloat{T},Matrix{DoubleFloat{T}}},
                   b::Vector{DoubleFloat{T}},
                   x::Vector{DoubleFloat{T}} = b) where {T <: AbstractFloat}
    require_one_based_indexing(A, b, x)
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
