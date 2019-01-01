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

The package name is perhaps a bit misleading: only a modest fraction
of the BLAS forest is implemented, the interface is (mostly) Julian
rather then BLAS-like, and some extra related methods are
included to make important parts of `LinearAlgebra` and its supplements
work efficiently.

The API is intended to be seamlessly compatible with the
`LinearAlgebra` standard library. That is, `using DoubleBLAS` will
extend various methods from `LinearAlgebra` so that frequent operations
such as matrix multiplication, LU and Cholesky factorization, and
`inv()` will employ more efficient methods than the generic ones.

## Warning

The arithmetic used in this package is the straightforward
double-double variety which does not respect all IEEE-754
rules. (Kahan calls it "fast but grubby", but concedes that it may be
useful for linear algebra, so here we are.) Underflow is largely
similar to IEEE-754, but overflow and treatment of NaN and infinities
are non-conforming (and complicated). Users are advised to make sure
that vectors and matrices are scaled to avoid overflow. (This is good
advice in general for linear algebra, but especially important here.)

## Multi-threading

Multi-threading (MT) is enabled for some sufficiently large problems.
We use `Base.Threads` (q.v. in the Julia manual), limited by the
`JULIA_NUM_THREADS` environment variable. On many systems there is
significant overhead for MT, so heuristic thresholds for switching
from simple versions to MT are provided. These may be adjusted with
the `set_mt_threshold(n::Real, problem::Symbol)` function.  Someday
tuned values for a given system might be set during package
installation or initialization, but currently they are notional
or based on very limited testing.

# Iterative refinement

DoubleFloats are especially useful for mixed-precision iterative
refinement.  This can be used to improve solutions of moderately
poorly-conditioned problems.
I didn't find an implementation in other well-known packages,
so I provide `refinedldiv` here:
```julia
julia> n=4096; A=rand(Double64,n,n); x=rand(Double64,n);
julia> b=A*x;
julia> F=lu(Float64.(A));
julia> bf = Float64.(b);
julia> xf = F \ bf;
julia> norm(xf-x)/norm(x)
6.745241585354585e-11

julia> xx,cvg = refinedldiv(A,F,b);
julia> norm(xx-x)/norm(x)
2.1020613807856875066955678325842648e-27
```
This takes a couple of seconds vs. more than a minute for a full `Double64`
factorization (with several threads).

# Acknowledgements
Most of the arithmetic was copied from DoubleFloats.jl and
AccurateArithmetic.jl. Linear algebra routines were adapted from the Julia
standard library.
