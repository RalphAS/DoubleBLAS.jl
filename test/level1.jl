function dotcheck(T,n,tol)
    x = rand(T,n)
    y = rand(T,n)
    xb = big.(x)
    yb = big.(y)
    db = dot(xb,yb)
    e = (T <: Complex) ? eps(real(T)) : eps(T)
    @test abs(big(dot(x,y)) - db) / abs(db) < tol * n * e

    # check for correct handling of strides
    xs = zeros(T,2*n) .+ T(NaN)
    ys = zeros(T,3*n) .+ T(NaN)
    for j=1:n
        xs[2*j]=x[j]
        ys[3*j]=y[j]
    end
    vx = view(xs,2:2:2*n)
    vy = view(ys,3:3:3*n)
    @test abs(big(dot(x,vy)) - db) / abs(db) < tol * n * e
    @test abs(big(dot(vx,y)) - db) / abs(db) < tol * n * e
    @test abs(big(dot(vx,vy)) - db) / abs(db) < tol * n * e
end

@testset "dot $T" for T in (Double64, Double32)
    n = 67
    dotcheck(T,n,2)
end
