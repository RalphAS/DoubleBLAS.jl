module DoubleBLAS1
# support for (fairly) efficient Linear Algebra with DoubleFloats

#FIXME: change to explicit lists because namespace pollution is epidemic
using DoubleFloats
using LinearAlgebra

using SIMD

# hope springs eternal in the coder's breast,
# for code rarely is but to be threaded
using Base.Threads

# steal some internals
using LinearAlgebra: has_offset_axes, lapack_size

# stuff to extend
import LinearAlgebra: rmul!, lmul!, ldiv!
# see subordinate files for some others

# most public functions are defined in LinearAlgebra

####################################
# SIMD stuff

# width of SIMD structures to use
const Npref = 8
# FIXME: select at install time.  Probably wants CPUID.
# Using Npref = 8 on a system w/ 256b SIMD regs imposes a 10% penalty
# for Double64 but presumably gives better performance on 512b systems.

@generated function vgethi(xv::StridedVector{DoubleFloat{T}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i])) for i in 1:N]...)))
    end
end

@generated function vgetlo(xv::StridedVector{DoubleFloat{T}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i])) for i in 1:N]...)))
    end
end

@generated function vgethire(xv::StridedVector{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i].re)) for i in 1:N]...)))
    end
end

@generated function vgetlore(xv::StridedVector{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i].re)) for i in 1:N]...)))
    end
end

@generated function vgethiim(xv::StridedVector{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i].im)) for i in 1:N]...)))
    end
end

@generated function vgetloim(xv::StridedVector{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i].im)) for i in 1:N]...)))
    end
end


include("ops.jl")
include("dots.jl")

include("gemm.jl")
end # module
