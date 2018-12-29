using DoubleBLAS
using LinearAlgebra
using DoubleFloats: Double32, Double64
using Test, Random

let nt = Threads.nthreads()
    nts = Sys.CPU_THREADS
    @info "running tests with $nt threads of $nts"
end

Random.seed!(1234)

include("level1.jl")
include("givens.jl")

function gemmcheck(T,m,n,k,tol)
    A = rand(T,m,k)
    B = rand(T,k,n)
    C = rand(T,m,n)
    Ab = big.(A)
    Bb = big.(B)
    C = A * B
    Cb = Ab * Bb
    e = (T <: Complex) ? eps(real(T)) : eps(T)
    @test opnorm(Cb - big.(C),1) / (opnorm(A,1) * opnorm(B,1)) < tol * k * e

    Ar = Matrix(reshape(A,k,m))
    Abr = Matrix(reshape(Ab,k,m))
    C = Ar' * B
    Cb = Abr' * Bb
    @test opnorm(Cb - big.(C),1) / (opnorm(A,1) * opnorm(B,1)) < tol * k * e
    Br = Matrix(reshape(B,n,k))
    Bbr = Matrix(reshape(Bb,n,k))
    C = Ar' * Br'
    Cb = Abr' * Bbr'
    @test opnorm(Cb - big.(C),1) / (opnorm(A,1) * opnorm(B,1)) < tol * k * e

    #FIXME: needs cases w/ simple transpose for complex types
end

@testset "gemm $T" for T in (Double64, Double32)
    # make sure to exercise the clean-up loops
    mA, nA, nB = 129, 67, 33
    gemmcheck(T,mA,nB,nA,1)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS.get_mt_threshold(:gemm)
        DoubleBLAS.set_mt_threshold(1.0e12,:gemm)
        gemmcheck(T,mA,nB,nA,1)
        DoubleBLAS.set_mt_threshold(t,:gemm)
    end
end

@testset "gemm $T" for T in (Complex{Double64}, Complex{Double32})
    # make sure to exercise the clean-up loops
    mA, nA, nB = 129, 67, 33
    gemmcheck(T,mA,nB,nA,1)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS.get_mt_threshold(:gemm)
        DoubleBLAS.set_mt_threshold(1.0e12,:gemm)
        gemmcheck(T,mA,nB,nA,1)
        DoubleBLAS.set_mt_threshold(t,:gemm)
    end
end

function gemvcheck(T,m,n,k,tol)
    A = rand(T,m,k)
    B = rand(T,k)
    C = rand(T,m)
    Ab = big.(A)
    Bb = big.(B)
    C = A * B
    Cb = Ab * Bb
    e = (T <: Complex) ? eps(real(T)) : eps(T)
    @test norm(Cb - big.(C),1) / (opnorm(A,1) * norm(B,1)) < tol * k * e

    Ar = Matrix(reshape(A,k,m))
    Abr = Matrix(reshape(Ab,k,m))
    C = Ar' * B
    Cb = Abr' * Bb
    @test norm(Cb - big.(C),1) / (opnorm(A,1) * norm(B,1)) < tol * k * e
end

@testset "gemv $T" for T in (Double64, Double32)
    # make sure to exercise the clean-up loops
    mA, nA, nB = 129, 67, 33
    gemvcheck(T,mA,nB,nA,1)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS.get_mt_threshold(:gemv)
        DoubleBLAS.set_mt_threshold(1.0e12,:gemv)
        gemvcheck(T,mA,nB,nA,1)
        DoubleBLAS.set_mt_threshold(t,:gemv)
    end
end

function lucheck(T,k,tol)
    A = rand(T,k,k)
    if (T <: Complex)
        κ = cond(ComplexF64.(A),Inf)
    else
        κ = cond(Float64.(A),Inf)
    end
    F = lu(A)
    xb = rand(T,k)
    b = A * xb
    x = F \ b
    e = eps(real(T))
    @test norm(x - xb, Inf) / norm(xb, Inf) < tol * κ * e

    Ai = inv(A)
    # CHECKME: why did I pick this norm?
    @test opnorm(Ai * A - I, 1) / opnorm(A, 1) < tol * k * e
end

@testset "lu $T" for T in (Double64, Double32, Complex{Double64})
    # make sure to exercise the clean-up loops
    nA = 67
    lucheck(T,nA,10)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS.get_mt_threshold(:lu)
        DoubleBLAS.set_mt_threshold(1.0e12,:lu)
        lucheck(T,nA,10)
        DoubleBLAS.set_mt_threshold(t,:lu)
    end
end

squareQ(Q::LinearAlgebra.AbstractQ) = (sq = size(Q.factors, 1); lmul!(Q, Matrix{eltype(Q)}(I, sq, sq)))

rectangularQ(Q::LinearAlgebra.AbstractQ) = convert(Array, Q)

function qrcheck(T,k,m,tol)
    A = rand(T,k,k)
    qra = qr(A)
    q, r = qra.Q, qra.R
    e = eps(real(T))
    @test norm(q' * squareQ(q) - I) < tol * k * e
    @test norm(A - q * r) / norm(A) < tol * k * e
    b = rand(T,k)
    @test norm(A * (qra \ b) - b) / norm(b) < tol * k * e
    A = rand(T,k,m)
    qra = qr(A)
    q, r = qra.Q, qra.R
    e = eps(real(T))
    @test norm(q' * squareQ(q) - I) < tol * k * e
    @test norm(q' * rectangularQ(q) - I) < tol * k * e
    @test norm(A - q * r) / norm(A) < tol * k * e
end

@testset "qr $T" for T in (Double64, Double32, Complex{Double64})
    # make sure to exercise the clean-up loops
    nA = 67
    mA = 49
    qrcheck(T,mA,nA,10)
end

function cholcheck(T,k,tol)
    A = rand(T,k,k)
    A = A * adjoint(A)
    # noncommutative multiplication means we need to be extra thorough
    A = A + adjoint(A)
    if (T <: Complex)
        κ = cond(ComplexF64.(A),Inf)
    else
        κ = cond(Float64.(A),Inf)
    end
    F = cholesky(A)
    xb = rand(T,k)
    b = A * xb
    x = F \ b
    e = eps(real(T))
    @test norm(x - xb,Inf) / norm(xb,Inf) < tol * κ * e

    C, info = LinearAlgebra._chol!(copy(A), LowerTriangular)
    F = LinearAlgebra.Cholesky(C.data,'L',info)
    x = F \ b
    @test norm(x - xb,Inf) / norm(xb,Inf) < tol * κ * e
end

# put this last since it depends on patches to DoubleFloats
@testset "chol $T" for T in (Double64, Double32, Complex{Double64})
    # make sure to exercise the clean-up loops
    nA = 67
    tol = 10
    cholcheck(T,nA,tol)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS.get_mt_threshold(:chol)
        DoubleBLAS.set_mt_threshold(1.0e12,:chol)
        cholcheck(T,nA,tol)
        DoubleBLAS.set_mt_threshold(t,:chol)
    end
end

include("refine.jl")
