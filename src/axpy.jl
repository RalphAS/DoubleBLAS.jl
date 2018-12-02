
function _axpy!(a::DoubleFloat{T}, xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}},::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(ArgumentError("arguments must have equal lengths"))
    nd,nr = divrem(n, N)
    ahi = HI(a)
    alo = LO(a)

    shi = Vec{N,T}(ahi)    # this lets us avoid @generated (Thanks, SIMD.jl)
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
                xv::StridedVecOrMat{DoubleFloat{T}}, ix1::Integer,
                yv::StridedVecOrMat{DoubleFloat{T}}, iy1::Integer,
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
