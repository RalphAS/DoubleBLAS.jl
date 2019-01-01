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

function dot(xv::StridedVector{Complex{DoubleFloat{T}}},
             yv::StridedVector{Complex{DoubleFloat{T}}})  where {N, T <: AbstractFloat}
    _dot(xv,yv,Vec{Npref,T})
end

function _dot(xv::StridedVector{Complex{DoubleFloat{T}}}, yv::StridedVector{Complex{DoubleFloat{T}}},::Type{Vec{N,T}},tA=:c) where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(DimensionMismatch("arguments must have equal lengths"))
    z = zero(T)
    srhi = Vec{N,T}(0)
    srlo = Vec{N,T}(0)
    sihi = Vec{N,T}(0)
    silo = Vec{N,T}(0)
    nd,nr = divrem(n, N)
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N

            xrhi = vgethire(xv,i0,Vec{N,T})
            xrlo = vgetlore(xv,i0,Vec{N,T})
            yrhi = vgethire(yv,i0,Vec{N,T})
            yrlo = vgetlore(yv,i0,Vec{N,T})
            xihi = vgethiim(xv,i0,Vec{N,T})
            xilo = vgetloim(xv,i0,Vec{N,T})
            yihi = vgethiim(yv,i0,Vec{N,T})
            yilo = vgetloim(yv,i0,Vec{N,T})

            z1hi, z1lo = dfvmul(xrhi, xrlo, yrhi, yrlo)
            z2hi, z2lo = dfvmul(xrhi, xrlo, yihi, yilo)
            z3hi, z3lo = dfvmul(xihi, xilo, yrhi, yrlo)
            z4hi, z4lo = dfvmul(xihi, xilo, yihi, yilo)

            if tA == :c
                zrhi, zrlo = dfvadd(z1hi, z1lo, z4hi, z4lo)
                zihi, zilo = dfvsub(z2hi, z2lo, z3hi, z3lo)
            else
                zrhi, zrlo = dfvsub(z1hi, z1lo, z4hi, z4lo)
                zihi, zilo = dfvadd(z2hi, z2lo, z3hi, z3lo)
            end

            srhi, srlo = dfvadd(srhi, srlo, zrhi, zrlo)
            sihi, silo = dfvadd(sihi, silo, zihi, zilo)
        end
    end
    srhi, srlo = add_(srhi, srlo) # this should canonicalize all at once
    sihi, silo = add_(sihi, silo)
    sr = DoubleFloat((srhi[1], srlo[1]))
    si = DoubleFloat((sihi[1], silo[1]))
    for j=2:N
        sr += DoubleFloat((srhi[j], srlo[j]))
        si += DoubleFloat((sihi[j], silo[j]))
    end
    s = complex(sr,si)
    (nr == 0) && return s
    @inbounds begin
        if tA == :c
            @simd for i in (nd*N)+1:n
                s += conj(xv[i])*yv[i]
            end
        else
            @simd for i in (nd*N)+1:n
                s += xv[i]*yv[i]
            end
        end
    end
    s
end

# partial-vector versions to support level 2 and 3 methods

# warning: unsafe indexing
function _dot(n::Integer,
              xv::StridedVecOrMat{DoubleFloat{T}}, ix1::Integer,
              yv::StridedVecOrMat{DoubleFloat{T}}, iy1::Integer,
              ::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
    z = zero(T)
    shi = Vec{N,T}(0)
    slo = Vec{N,T}(0)
    nd,nr = divrem(n, N)
    ixoff = ix1-1
    iyoff = iy1-1
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N
            ix0=ixoff + i0
            iy0=iyoff + i0

            xhi = vgethi(xv,ix0,Vec{N,T})
            xlo = vgetlo(xv,ix0,Vec{N,T})
            yhi = vgethi(yv,iy0,Vec{N,T})
            ylo = vgetlo(yv,iy0,Vec{N,T})

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
            s += xv[ixoff+i]*yv[iyoff+i]
        end
    end
    s
end

function _dot(n::Integer,
              xv::StridedVecOrMat{Complex{DoubleFloat{T}}}, ix1::Integer,
              yv::StridedVecOrMat{Complex{DoubleFloat{T}}}, iy1::Integer,
              ::Type{Vec{N,T}},tA=:c) where {N, T <: AbstractFloat}
    z = zero(T)
    srhi = Vec{N,T}(0)
    srlo = Vec{N,T}(0)
    sihi = Vec{N,T}(0)
    silo = Vec{N,T}(0)
    nd,nr = divrem(n, N)
    ixoff = ix1-1
    iyoff = iy1-1
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N
            ix0=ixoff + i0
            iy0=iyoff + i0

            xrhi = vgethire(xv,ix0,Vec{N,T})
            xrlo = vgetlore(xv,ix0,Vec{N,T})
            yrhi = vgethire(yv,iy0,Vec{N,T})
            yrlo = vgetlore(yv,iy0,Vec{N,T})
            xihi = vgethiim(xv,ix0,Vec{N,T})
            xilo = vgetloim(xv,ix0,Vec{N,T})
            yihi = vgethiim(yv,iy0,Vec{N,T})
            yilo = vgetloim(yv,iy0,Vec{N,T})

            z1hi, z1lo = dfvmul(xrhi, xrlo, yrhi, yrlo)
            z2hi, z2lo = dfvmul(xrhi, xrlo, yihi, yilo)
            z3hi, z3lo = dfvmul(xihi, xilo, yrhi, yrlo)
            z4hi, z4lo = dfvmul(xihi, xilo, yihi, yilo)

            if tA == :c
                zrhi, zrlo = dfvadd(z1hi, z1lo, z4hi, z4lo)
                zihi, zilo = dfvsub(z2hi, z2lo, z3hi, z3lo)
            else
                zrhi, zrlo = dfvsub(z1hi, z1lo, z4hi, z4lo)
                zihi, zilo = dfvadd(z2hi, z2lo, z3hi, z3lo)
            end

            srhi, srlo = dfvadd(srhi, srlo, zrhi, zrlo)
            sihi, silo = dfvadd(sihi, silo, zihi, zilo)
        end
    end
    srhi, srlo = add_(srhi, srlo) # this should canonicalize all at once
    sihi, silo = add_(sihi, silo)
    sr = DoubleFloat((srhi[1], srlo[1]))
    si = DoubleFloat((sihi[1], silo[1]))
    for j=2:N
        sr += DoubleFloat((srhi[j], srlo[j]))
        si += DoubleFloat((sihi[j], silo[j]))
    end
    s = complex(sr,si)
    (nr == 0) && return s
    @inbounds begin
        if tA == :c
            @simd for i in (nd*N)+1:n
                s += conj(xv[ixoff+i])*yv[iyoff+i]
            end
        else
            @simd for i in (nd*N)+1:n
                s += xv[ixoff+i]*yv[iyoff+i]
            end
        end
    end
    s
end


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
