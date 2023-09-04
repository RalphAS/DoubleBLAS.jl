module DoubleBLAS
# support for (fairly) efficient Linear Algebra with DoubleFloats

#FIXME: change to explicit lists because namespace pollution is epidemic
using DoubleFloats
using LinearAlgebra

using SIMD
using UnsafeArrays

using Base.Threads

# steal some internals
using LinearAlgebra: lapack_size, BlasInt, checknonsingular, MulAddMul
using LinearAlgebra: _modify!


if VERSION < v"1.2"
    using LinearAlgebra: has_offset_axes
    require_one_based_indexing(x...) =
        has_offset_axes(x...) && throw(ArgumentError("not implemented "
                                                  * "for offset axes"))
else
    using Base: require_one_based_indexing
end

# stuff to extend
import LinearAlgebra: rmul!, lmul!, ldiv!
# see subordinate files for some others

export refinedldiv

# most public functions are defined in LinearAlgebra

################################################################
# SIMD config

# width of SIMD structures to use
const Npref = 8
# Using Npref = 8 on a system w/ 256b SIMD registers imposes a 10% penalty
# for most modest-size Double64 problems.
# There is no obvious penalty for large problems, and this
# presumably gives better performance on 512b systems.
# Kudos to the LLVM people.

################################################################
# Multi-threading internals

# TODO: implement set_num_threads
_default_nt() = (VERSION > v"1.9") ? Threads.threadpoolsize() : Threads.nthreads()

const mt_thresholds = Dict{Symbol,Any}()

"""
set_mt_threshold(n::Real, problem::Symbol)

Set the size threshold for multi-threading in the DoubleBLAS package to `n`,
for matrix operations of class `problem`.
"""
function set_mt_threshold(n::Real, problem::Symbol)
    if problem ∈ keys(mt_thresholds)
        destref = mt_thresholds[problem]
        destref[] = Float64(n)
    else
        throw(ArgumentError("unrecognized problem key $problem: valid keys are $(keys(mt_thresholds))"))
    end
    nothing
end

function get_mt_threshold(problem::Symbol)
    if problem ∈ keys(mt_thresholds)
        srcref = mt_thresholds[problem]
    else
        throw(ArgumentError("unrecognized problem key $problem: valid keys are $(keys(mt_thresholds))"))
    end
    srcref = mt_thresholds[problem]
    srcref[]
end

function _part_range(rg, nt, it)
    l,r = divrem(length(rg), nt)
    if l == 0
        it > r && return 0:0
        l,r = 1,0
    end
    fst = firstindex(rg) + (it - 1) * l
    lst = fst + l - 1
    if r > 0
        if it <= r
            fst += (it-1)
            lst += it
        else
            fst += r
            lst += r
        end
    end
    return fst:lst
end

################################################################
# SIMD internals

@generated function vgethi(xv::StridedVecOrMat{DoubleFloat{T}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i])) for i in 1:N]...)))
    end
end

@generated function vgetlo(xv::StridedVecOrMat{DoubleFloat{T}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i])) for i in 1:N]...)))
    end
end

@inline function vputhilo!(xv::StridedVecOrMat{DoubleFloat{T}},i0,
                              zhi::Vec{N,T}, zlo::Vec{N,T}) where {N,T}
    @inbounds for i=1:N
        xv[i0+i] = DoubleFloat{T}((zhi[i],zlo[i]))
    end
end

@generated function vgethire(xv::StridedVecOrMat{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i].re)) for i in 1:N]...)))
    end
end

@generated function vgetlore(xv::StridedVecOrMat{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i].re)) for i in 1:N]...)))
    end
end

@generated function vgethiim(xv::StridedVecOrMat{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(HI(xv[i0+$i].im)) for i in 1:N]...)))
    end
end

@generated function vgetloim(xv::StridedVecOrMat{Complex{DoubleFloat{T}}},i0,::Type{Vec{N,T}}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:meta, :inbounds))
        Vec{N,T}(tuple($([:(LO(xv[i0+$i].im)) for i in 1:N]...)))
    end
end

@inline function vputhilo!(xv::StridedVecOrMat{Complex{DoubleFloat{T}}},i0,
                           zrhi::Vec{N,T}, zrlo::Vec{N,T},
                           zihi::Vec{N,T}, zilo::Vec{N,T}) where {N,T}
    @inbounds for i=1:N
        xv[i0+i] = complex(DoubleFloat{T}((zrhi[i],zrlo[i])),
                           DoubleFloat{T}((zihi[i],zilo[i])))
    end
end


include("ops.jl")
include("dots.jl")
include("axpy.jl")
include("level2.jl")
include("givens.jl")

include("gemm.jl")
include("triangular.jl")

include("lu.jl")
include("chol.jl")

include("refine.jl")

end # module
