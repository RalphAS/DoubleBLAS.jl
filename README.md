# DoubleBLAS

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/RalphAS/DoubleBLAS.jl.svg?branch=master)](https://travis-ci.com/RalphAS/DoubleBLAS.jl)
[![codecov.io](http://codecov.io/github/RalphAS/DoubleBLAS.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/DoubleBLAS.jl?branch=master)

This package is a draft implementation of SIMD-based basic linear algebra
routines for matrices with element types from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl).

The API is intended to be seamlessly compatible with the
`LinearAlgebra` standard library. That is, `using DoubleBLAS` will
extend various methods from `LinearAlgebra` so that various operations
such as matrix multiplication, LU and Cholesky factorization, and
`inv()` will employ more efficient methods than the generic ones.


## Multi-threading

Multi-threading (MT) is enabled for some sufficiently large problems.
We use `Base.Threads` (q.v. in the Julia manual), limited by the
`JULIA_NUM_THREADS` environment variable. On many systems there is
significant overhead for MT, so heuristic thresholds for switching
from simple versions to MT are provided. These may be adjusted with
the `set_mt_threshold(n::Real, problem::Symbol)` function.  Someday
default values for a given system might be set during package
installation or initialization, but currently they are notional.

# Large problems

For very large linear problems, one should use mixed-precision iterative
refinement.  I didn't find an implementation in other well-known packages,
so I provide `refinedldiv` here:
```julia
julia> n=4096; A=rand(Double64,n,n); x=rand(Double64,n);
julia> F=lu(Float64.(A));
julia> b=A*x;
julia> xx,cvg = refinedldiv(A,F,b); # elapsed time: 1.4 sec
julia> norm(xx-x)/norm(x)
2.1020613807856875066955678325842648e-27
```

# Acknowledgements
Most of the arithmetic was copied from DoubleFloats.jl and
AccurateArithmetic.jl. Linear algebra routines were adapted from the Julia
standard library.
