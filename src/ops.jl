# extend basic ops to SIMD wrappers

# credit: basically copied from DoubleFloats.jl and AccurateArithmetic.jl

# aka twosum
@inline function add_(x::Vec{N,T}, y::Vec{N,T}) where {N, T<:AbstractFloat}
    hi  = x + y
    v = hi - x
    lo = (x - (hi - v)) + (y - v)
    return hi, lo
end

# aka quicktwosum
@inline function add_hilo_(x::Vec{N,T}, y::Vec{N,T}) where {N, T<:AbstractFloat}
    hi = x + y
    lo = y - (hi - x)
    return hi, lo
end

@inline function mul_(x::Vec{N,T}, y::Vec{N,T}) where {N, T<:AbstractFloat}
    hi = x * y
    lo = fma(x, y, -hi)
    return hi, lo
end

@inline function dfvadd(xhi::Vec{N,T}, xlo::Vec{N,T}, yhi::Vec{N,T}, ylo::Vec{N,T}) where {N,T<:AbstractFloat}
    hi, lo = add_(xhi, yhi)
    thi, tlo = add_(xlo, ylo)
    c = lo + thi
    hi, lo = add_hilo_(hi, c)
    c = tlo + lo
    hi, lo = add_hilo_(hi, c)
    return hi, lo
end

@inline function dfvmul(xhi::Vec{N,T}, xlo::Vec{N,T}, yhi::Vec{N,T}, ylo::Vec{N,T}) where {N,T<:AbstractFloat}
    hi, lo = mul_(xhi, yhi)
    t = xlo * ylo
    t = fma(xhi, ylo, t)
    t = fma(xlo, yhi, t)
    t = lo + t
    hi, lo = add_hilo_(hi, t)
    return hi, lo
end
