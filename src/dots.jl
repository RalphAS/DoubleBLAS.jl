import LinearAlgebra.dot

function dot(xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}})  where {N, T <: AbstractFloat}
    _dot(xv,yv,Vec{Npref,T})
end

function _dot(xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}},::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(DimensionMismatch("arguments must have equal lengths"))
    z = zero(T)
    shi = Vec{N,T}(0)
    slo = Vec{N,T}(0)
    nd,nr = divrem(n, N)
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N

            xhi = vgethi(xv,i0,Vec{N,T})
            xlo = vgetlo(xv,i0,Vec{N,T})
            yhi = vgethi(yv,i0,Vec{N,T})
            ylo = vgetlo(yv,i0,Vec{N,T})

            zhi, zlo = dfvmul(xhi, xlo, yhi, ylo)
            shi, slo = dfvadd(shi, slo, zhi, zlo)
        end
    end
    shi, slo = add_(shi, slo) # this should canonicalize all at once
    s = DoubleFloat((shi[1], slo[1]))
    for j=2:N
        s += DoubleFloat((shi[j], slo[j]))
        end
    (nr == 0) && return s
    @inbounds begin
        @simd for i in (nd*N)+1:n
            s += xv[i]*yv[i]
        end
    end
    s
end

# warning: unsafe indexing
function _dot(n::Integer,
              xv::StridedVecOrMat{DoubleFloat{T}}, ix1::Integer,
              yv::StridedVector{DoubleFloat{T}},
              ::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
    z = zero(T)
    shi = Vec{N,T}(0)
    slo = Vec{N,T}(0)
    nd,nr = divrem(n, N)
    ixoff = ix1-1
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N
            ix0=ixoff + i0

            xhi = vgethi(xv,ix0,Vec{N,T})
            xlo = vgetlo(xv,ix0,Vec{N,T})
            yhi = vgethi(yv,i0,Vec{N,T})
            ylo = vgetlo(yv,i0,Vec{N,T})

            zhi, zlo = dfvmul(xhi, xlo, yhi, ylo)
            shi, slo = dfvadd(shi, slo, zhi, zlo)
        end
    end
    shi, slo = add_(shi, slo) # this should canonicalize all at once
    s = DoubleFloat((shi[1], slo[1]))
    for j=2:N
        s += DoubleFloat((shi[j], slo[j]))
        end
    (nr == 0) && return s
    @inbounds begin
        @simd for i in (nd*N)+1:n
            s += xv[ixoff+i]*yv[i]
        end
    end
    s
end

#=
function dot(xv::StridedVector{Complex{DoubleFloat{T}}},
             yv::StridedVector{Complex{DoubleFloat{T}}})  where {N, T <: AbstractFloat}
    _dotc(xv,yv,Vec{Npref,T})
end
=#

# reference version for testing
function dot1(x::StridedVector{T}, y::StridedVector{T}) where {T}
    n = length(x)
    (length(y) == n) || throw(ArgumentError("arguments must have equal lengths"))
    s = zero(T)
    @inbounds begin
        @simd for i in eachindex(x)
            s += x[i]*y[i]
        end
    end
    s
end
