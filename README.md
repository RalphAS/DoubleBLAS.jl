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
routines for matrices with element types from DoubleFloats.jl.

Some design decisions and implementation notes:
* The API is intended to be seamlessly compatible with the LinearAlgebra standard library.
* Matrices (sometimes fully dense) are copied and/or materialized whenever it helps, since we expect use of DoubleFloats for huge problems to be rare.
* Basic DoubleFloat ops are expensive, so only moderate attention has been given to cache management.
* The bottlenecks in our code are likely due to instruction latency. Someday experiments with unrolling might be worthwhile.


# Acknowledgements
Most of the arithmetic was copied from DoubleFloats.jl and
AccurateArithmetic.jl. Linear algebra routines were adapted from the Julia
standard library.
