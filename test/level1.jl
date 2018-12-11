function dotcheck(T,n,tol)
    x = rand(T,n)
    y = rand(T,n)
    xb = big.(x)
    yb = big.(y)
    db = dot(xb,yb)
    e = eps(real(T))
    sxy = norm(xb)*norm(yb)
    @test abs(big(dot(x,y)) - db) / sxy < tol * n * e

    # check for correct handling of strides
    xs = zeros(T,2*n) .+ T(NaN)
    ys = zeros(T,3*n) .+ T(NaN)
    for j=1:n
        xs[2*j]=x[j]
        ys[3*j]=y[j]
    end
    vx = view(xs,2:2:2*n)
    vy = view(ys,3:3:3*n)
    @test abs(big(dot(x,vy)) - db) / sxy < tol * n * e
    @test abs(big(dot(vx,y)) - db) / sxy < tol * n * e
    @test abs(big(dot(vx,vy)) - db) / sxy < tol * n * e
end

@testset "dot $T" for T in (Double64, Double32, Complex{Double64})
    n = 67
    dotcheck(T,n,2)
end

function axpycheck(T,n,tol)
    x = rand(T,n)
    y = rand(T,n)
    xb = big.(x)
    yb = big.(y)
    if T <: Complex
        a = T(1.5 + 2.5im)
        ab = big(1.5) + 1im*big(2.5)
    else
        a = T(1.5)
        ab = big(1.5)
    end
    zb = LinearAlgebra.axpy!(ab,xb,yb)
    e = eps(real(T))
    @test norm(big.(axpy!(a,x,y)) - zb) / norm(zb) < tol * n * e
end

@testset "axpy $T" for T in (Double64, Double32, Complex{Double64})
    n = 67
    axpycheck(T,n,2)
end

function scalecheck(T,n,tol)
    x = rand(T,n)
    xb = big.(x)
    if T <: Complex
        a = T(1.5 + 2.5im)
        ab = big(1.5) + 1im*big(2.5)
    else
        a = T(1.5)
        ab = big(1.5)
    end
    e = eps(real(T))
    zb = copy(xb)
    LinearAlgebra.lmul!(ab,zb)
    z = copy(x)
    @test norm(big.(lmul!(a,z)) - zb) / norm(zb) < tol * n * e
    zb = copy(xb)
    LinearAlgebra.rmul!(zb,ab)
    z = copy(x)
    @test norm(big.(rmul!(z,a)) - zb) / norm(zb) < tol * n * e
end

@testset "scale $T" for T in (Double64, Double32, Complex{Double64})
    n = 67
    scalecheck(T,n,2)
end
