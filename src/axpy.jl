
function axpy!(a::DoubleFloat{T}, xv::StridedVector{DoubleFloat{T}}, yv::StridedVector{DoubleFloat{T}},::Type{Vec{N,T}}) where {N, T <: AbstractFloat}
     n = length(xv)
    (length(yv) == n) || throw(ArgumentError("arguments must have equal lengths"))
    nd,nr = divrem(n, N)
    ahi = HI(a)
    alo = LO(a)
    # FIXME: this apparently needs to be @generated
#    shi = Vec{N,T}((ahi,ahi,ahi,ahi,ahi,ahi,ahi,ahi))
#    slo = Vec{N,T}((alo,alo,alo,alo,alo,alo,alo,alo))
    shi = Vec{N,T}((ahi,ahi,ahi,ahi,ahi,ahi,ahi,ahi))
    slo = Vec{N,T}((alo,alo,alo,alo,alo,alo,alo,alo))
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
