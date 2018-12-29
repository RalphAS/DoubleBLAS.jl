# these do most of the work in Schur decomposition, SVD, etc.
using LinearAlgebra: Givens

function _floatmin2(::Type{T}) where {T}
    twopar = 2one(T)
    twopar^trunc(Integer,log(floatmin(T)/eps(T))/log(twopar)/twopar)
end
const d32fm2 = _floatmin2(Double32)
const d64fm2 = _floatmin2(Double64)
LinearAlgebra.floatmin2(::Type{Double32}) = d32fm2
LinearAlgebra.floatmin2(::Type{Double64}) = d64fm2

@noinline function lmul!(G::Givens{DT}, A::AbstractVecOrMat{DT}
                       ) where {DT <: Complex{DoubleFloat{T}}} where T
    has_offset_axes(A) && throw(ArgumentError("not implemented "
                                              * "for offset axes"))
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    a1 = Vector{DT}(undef,n)
    a2 = Vector{DT}(undef,n)
    @inbounds begin
        for i=1:n
            a2[i] = A[G.i2,i]
        end
        for i=1:n
            a1[i] = A[G.i1,i]
        end
        rmul!(view(A,G.i1,:),G.c)
        axpy!(G.s,a2,view(A,G.i1,:))
        rmul!(view(A,G.i2,:),G.c)
        axpy!(-conj(G.s),a1,view(A,G.i2,:))
    end
    return A
end

@noinline function lmul!(G::Givens{DoubleFloat{T}},
                         A::StridedMatrix{DoubleFloat{T}}) where T
    has_offset_axes(A) && throw(ArgumentError("not implemented "
                                              * "for offset axes"))
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    _lmul!(G,view(A,G.i1,:),view(A,G.i2,:),Vec{Npref,T})
end
function _lmul!(G::Givens{DoubleFloat{T}}, xv::StridedVector{DoubleFloat{T}},
                yv::StridedVector{DoubleFloat{T}},::Type{Vec{N,T}}) where {N,T}
    n = size(xv,1)
    nd,nr = divrem(n, N)
    ahi, alo = HILO(G.c)
    chi = Vec{N,T}(ahi)
    clo = Vec{N,T}(alo)
    ahi, alo = HILO(G.s)
    shi = Vec{N,T}(ahi)
    slo = Vec{N,T}(alo)
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N

            xhi = vgethi(xv,i0,Vec{N,T})
            xlo = vgetlo(xv,i0,Vec{N,T})
            yhi = vgethi(yv,i0,Vec{N,T})
            ylo = vgetlo(yv,i0,Vec{N,T})

            uhi, ulo = dfvmul(xhi, xlo, chi, clo)
            vhi, vlo = dfvmul(yhi, ylo, shi, slo)
            zhi, zlo = dfvadd(uhi, ulo, vhi, vlo)
            vputhilo!(xv,i0,zhi,zlo)

            vhi, vlo = dfvmul(xhi, xlo, shi, slo)
            uhi, ulo = dfvmul(yhi, ylo, chi, clo)
            zhi, zlo = dfvsub(uhi, ulo, vhi, vlo)
            vputhilo!(yv,i0,zhi,zlo)
        end
    end
    (nr == 0) && return yv
    @inbounds begin
        @simd for i in (nd*N)+1:n
            t1 = xv[i]
            t2 = yv[i]
            xv[i] = G.c*xv[i] + G.s*t2
            yv[i] = G.c*yv[i] - G.s*t1 # conj(G.s) == G.s for Real
        end
    end
    nothing
end

#@inline function rmul!(A::AbstractMatrix{DT}, G::Givens{DT}
@noinline function rmul!(A::AbstractMatrix{Complex{DoubleFloat{T}}},
                         G::Givens{Complex{DoubleFloat{T}}}
                       ) where T
    has_offset_axes(A) && throw(ArgumentError("not implemented "
                                              * "for offset axes"))
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    a1 = Vector{Complex{DoubleFloat{T}}}(undef,m)
    a2 = Vector{Complex{DoubleFloat{T}}}(undef,m)
    @inbounds begin
        for i=1:m
            a1[i] = A[i,G.i1]
            a2[i] = A[i,G.i2]
        end
        rmul!(view(A,:,G.i1),G.c)
        rmul!(view(A,:,G.i2),G.c)
        axpy!(-conj(G.s),a2,view(A,:,G.i1))
        axpy!(G.s,a1,view(A,:,G.i2))
    end
    return A
end

@noinline function rmul!(A::StridedMatrix{DoubleFloat{T}},
                         G::Givens{DoubleFloat{T}}
                         ) where T
    has_offset_axes(A) && throw(ArgumentError("not implemented "
                                              * "for offset axes"))
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    _rmul!(view(A,:,G.i1),view(A,:,G.i2),G,Vec{Npref,T})
end
function _rmul!(xv::StridedVector{DoubleFloat{T}},
                yv::StridedVector{DoubleFloat{T}},
                G::Givens{DoubleFloat{T}}, ::Type{Vec{N,T}}) where {N,T}
    n = size(xv,1)
    nd,nr = divrem(n, N)
    ahi, alo = HILO(G.c)
    chi = Vec{N,T}(ahi)
    clo = Vec{N,T}(alo)
    ahi, alo = HILO(G.s)
    shi = Vec{N,T}(ahi)
    slo = Vec{N,T}(alo)
    @inbounds begin
        for i in 1:nd
            i0=(i-1)*N

            xhi = vgethi(xv,i0,Vec{N,T})
            xlo = vgetlo(xv,i0,Vec{N,T})
            yhi = vgethi(yv,i0,Vec{N,T})
            ylo = vgetlo(yv,i0,Vec{N,T})

            uhi, ulo = dfvmul(xhi, xlo, chi, clo)
            vhi, vlo = dfvmul(yhi, ylo, shi, slo)
            zhi, zlo = dfvsub(uhi, ulo, vhi, vlo)
            vputhilo!(xv,i0,zhi,zlo)

            vhi, vlo = dfvmul(xhi, xlo, shi, slo)
            uhi, ulo = dfvmul(yhi, ylo, chi, clo)
            zhi, zlo = dfvadd(uhi, ulo, vhi, vlo)
            vputhilo!(yv,i0,zhi,zlo)
        end
    end
    (nr == 0) && return yv
    @inbounds begin
        @simd for i in (nd*N)+1:n
            a1 = xv[i]
            a2 = yv[i]
            xv[i] = G.c*xv[i] - G.s*a2
            yv[i] = G.c*yv[i] + G.s*a1
        end
    end
    nothing
end
