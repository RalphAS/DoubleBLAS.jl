using DoubleBLAS1
using LinearAlgebra
using DoubleFloats: Double32, Double64
using Test, Random

Random.seed!(1234)

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

@testset "gemm $T" for T in (Double64, Double32)
    # make sure to exercise the clean-up loops
    mA, nA, nB = 129, 67, 33
    gemmcheck(T,mA,nB,nA,1)
    # also run below MT threshold
    if Threads.nthreads() > 1
        t = DoubleBLAS1.get_mt_threshold(:gemm)
        DoubleBLAS1.set_mt_threshold(1.0e12,:gemm)
        gemmcheck(T,mA,nB,nA,1)
        DoubleBLAS1.set_mt_threshold(t,:gemm)
    end
end

@testset "gemm $T" for T in (Complex{Double64}, Complex{Double32})
    # make sure to exercise the clean-up loops
    mA, nA, nB = 129, 67, 33
    gemmcheck(T,mA,nB,nA,1)
end

