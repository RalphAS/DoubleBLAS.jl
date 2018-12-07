import LinearAlgebra.axpy!

function axpy!(a::DoubleFloat{T}, xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}})  where {N, T <: AbstractFloat}
    _axpy!(a,xv,yv,Vec{Npref,T})
end
function axpy!(a::Complex{DoubleFloat{T}}, xv::StridedVector{Complex{DoubleFloat{T}}}, yv::StridedVector{Complex{DoubleFloat{T}}})  where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(ArgumentError("arguments must have equal lengths"))
    _axpy!(n,a,xv,1,yv,1,Vec{Npref,T})
end


function _axpy!(a::DoubleFloat{T}, xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}},::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(ArgumentError("arguments must have equal lengths"))
    nd,nr = divrem(n, N)
    ahi = HI(a)
    alo = LO(a)

    shi = Vec{N,T}(ahi)    # this lets us avoid @generated (Thanks, SIMD.jl!)
    slo = Vec{N,T}(alo)
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N

            xhi = vgethi(xv,i0,Vec{N,T})
            xlo = vgetlo(xv,i0,Vec{N,T})
            yhi = vgethi(yv,i0,Vec{N,T})
            ylo = vgetlo(yv,i0,Vec{N,T})

            zhi, zlo = dfvmul(xhi, xlo, shi, slo)
            zhi, zlo = dfvadd(yhi, ylo, zhi, zlo)
            vputhilo!(yv,i0,zhi,zlo)

        end
    end
    (nr == 0) && return yv
    @inbounds begin
        @simd for i in (nd*N)+1:n
            yv[i] += a * xv[i]
        end
    end
    yv
end

# warning: unsafe indexing
function _axpy!(n, a::DoubleFloat{T},
                xv::StridedVecOrMat{DoubleFloat{T}}, ix1::Int,
                yv::StridedVecOrMat{DoubleFloat{T}}, iy1::Int,
                ::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
    nd,nr = divrem(n, N)
    ahi = HI(a)
    alo = LO(a)
    shi = Vec{N,T}(ahi)
    slo = Vec{N,T}(alo)
    ixoff = ix1-1
    iyoff = iy1-1
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N
            ix0 = ixoff + i0
            iy0 = iyoff + i0

            xhi = vgethi(xv,ix0,Vec{N,T})
            xlo = vgetlo(xv,ix0,Vec{N,T})
            yhi = vgethi(yv,iy0,Vec{N,T})
            ylo = vgetlo(yv,iy0,Vec{N,T})

            zhi, zlo = dfvmul(xhi, xlo, shi, slo)
            zhi, zlo = dfvadd(yhi, ylo, zhi, zlo)
            vputhilo!(yv,iy0,zhi,zlo)

        end
    end
    (nr == 0) && return yv
    @inbounds begin
        @simd for i in (nd*N)+1:n
            yv[iyoff+i] += a * xv[ixoff+i]
        end
    end
    yv
end

function _axpy!(n, a::Complex{DoubleFloat{T}},
                xv::StridedVecOrMat{Complex{DoubleFloat{T}}}, ix1::Integer,
                yv::StridedVecOrMat{Complex{DoubleFloat{T}}}, iy1::Integer,
                ::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
    nd,nr = divrem(n, N)
    arhi = HI(real(a))
    arlo = LO(real(a))
    srhi = Vec{N,T}(arhi)
    srlo = Vec{N,T}(arlo)
    aihi = HI(imag(a))
    ailo = LO(imag(a))
    sihi = Vec{N,T}(aihi)
    silo = Vec{N,T}(ailo)
    ixoff = ix1-1
    iyoff = iy1-1
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N
            ix0 = ixoff + i0
            iy0 = iyoff + i0

            xrhi = vgethire(xv,ix0,Vec{N,T})
            xrlo = vgetlore(xv,ix0,Vec{N,T})
            yrhi = vgethire(yv,iy0,Vec{N,T})
            yrlo = vgetlore(yv,iy0,Vec{N,T})
            xihi = vgethiim(xv,ix0,Vec{N,T})
            xilo = vgetloim(xv,ix0,Vec{N,T})
            yihi = vgethiim(yv,iy0,Vec{N,T})
            yilo = vgetloim(yv,iy0,Vec{N,T})

            z1hi, z1lo = dfvmul(xrhi, xrlo, srhi, srlo)
            z2hi, z2lo = dfvmul(xrhi, xrlo, sihi, silo)
            z3hi, z3lo = dfvmul(xihi, xilo, srhi, srlo)
            z4hi, z4lo = dfvmul(xihi, xilo, sihi, silo)

            zrhi, zrlo = dfvsub(z1hi, z1lo, z4hi, z4lo)
            zihi, zilo = dfvadd(z2hi, z2lo, z3hi, z3lo)

            zrhi, zrlo = dfvadd(yrhi, yrlo, zrhi, zrlo)
            zihi, zilo = dfvadd(yihi, yilo, zihi, zilo)
            vputhilo!(yv,iy0,zrhi,zrlo,zihi,zilo)

        end
    end
    (nr == 0) && return yv
    @inbounds begin
        @simd for i in (nd*N)+1:n
            yv[iyoff+i] += a * xv[ixoff+i]
        end
    end
    yv
end
