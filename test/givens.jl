function givenscheck(T,n,tol)
    e = eps(real(T))
    A = rand(T,n,n)
    Ab = big.(A)
    G,_ = givens(A[4,5],A[5,5],4,5)
    Gb,_ = givens(Ab[4,5],Ab[5,5],4,5)
    B = copy(A)
    Bb = copy(Ab)
    lmul!(G,B)
    lmul!(Gb,Bb)
    @test norm(B-Bb) < tol * n * e
    rmul!(A,G)
    rmul!(Ab,Gb)
    @test norm(A-Ab) < tol * n * e
end

@testset "givens $T" for T in (Double64, Double32, Complex{Double64})
    n = 67
    givenscheck(T,n,4)
end
