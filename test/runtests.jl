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
end

function lucheck(T,k,tol)
    A = rand(T,k,k)
    Ab = big.(A)
    F = lu(A)
    Fb = lu(Ab)
    b = rand(T,k)
    x = F \ b
    xb = Fb \ big.(b)
    e = (T <: Complex) ? eps(real(T)) : eps(T)
    @test norm(x - xb) / norm(x) < tol * k * e

    Ai = inv(A)
    @test opnorm(Ai * A - I,1) / opnorm(A,1) < tol * k * e
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
end

@testset "lu $T" for T in (Double64, Double32)
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

