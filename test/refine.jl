# FIXME: this is grossly inadequate

@testset "refine" begin
    Thigh = Double64
    Tlow = Float64
    # make a moderately unfriendly matrix
    n = 128
    B = rand(Tlow,n,n)
    SF = svd(B)
    acond = 1/sqrt(eps(Tlow))
    A = (1/acond) * SF.U * diagm(0 => exp.((0:n-1)*log(acond)/(n-1))) * SF.Vt

    Ad = Thigh.(A)
    x = rand(Thigh,n)
    bd = Ad * x
    b = Tlow.(bd)
    Flow = lu(A)
    xlow = Flow \ b
    # make sure we didn't sneak through
    # println("low-prec error: ",norm(x-xlow) / norm(x))
    @assert norm(x-xlow) / norm(x) > 1e4 * n * acond * eps(Thigh)
    xhigh,cvg = refinedldiv(Ad, Flow, bd)
    # println("high-prec error: ",norm(x-xhigh) / norm(x))
    @test norm(x-xhigh) / norm(x) < n * acond * eps(Thigh)
end

