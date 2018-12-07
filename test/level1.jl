function dotcheck(T,n,tol)
    x = rand(T,n)
    y = rand(T,n)
    xb = big.(x)
    yb = big.(y)
    db = dot(xb,yb)
    e = (T <: Complex) ? eps(real(T)) : eps(T)
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
    a = T(1.5)
    ab = big(1.5)
    zb = LinearAlgebra.axpy!(ab,xb,yb)
    e = (T <: Complex) ? eps(real(T)) : eps(T)
    @test norm(big.(axpy!(a,x,y)) - zb) / norm(zb) < tol * n * e
end

@testset "axpy $T" for T in (Double64, Double32, Complex{Double64})
    n = 67
    axpycheck(T,n,2)
end
