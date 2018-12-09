
"""
    refinedldiv(A::AbstractMatrix{T},F,B::AbstractVecOrMat{T}) => X, converged

Compute an accurate solution to a linear system ``A X = B.``

`F` must be a factorization object of `A` (typically produced by `lu`
or `cholesky`). Crucially, `eltype(F)` may have lower precision than
`T`.  Uses iterative refinement with intermediates of `eltype` `T` so
that the backward error is of order `eps(T)`. Note: this function
allocates several working matrices.

# Example
```julia
julia> A = rand(BigFloat,n,n); b = rand(BigFloat,n); F = lu(Float64.(A));
julia> x = refinedldiv(A,F,b)
```
Note that with the high precision of `BigFloat` full convergence is
often impossible, but the result is still quite accurate.

# Optional arguments
- `maxiter::Int`: default 10.
- `tol::Real`: tolerance for relative residual norm, scaled to `eps(T)`
- `verbose::Bool`: whether to report on residuals.
"""
function refinedldiv end

function refinedldiv(A::AbstractMatrix{DT}, F::Factorization{T},
                B::AbstractVecOrMat{DT};
                maxiter=10, tol=10, verbose=false
                ) where {DT <: AbstractFloat, T <: AbstractFloat}
    nrhs = size(B,2)
    n = size(A,1)
    cvtok = true
    anorm = opnorm(A,Inf)
    bnorms = [norm(view(B,:,j),Inf) for j=1:nrhs]
    tolf = tol * eps(DT)
    local X
    try
        X = T.(B)
    catch
        cvtok = false
    end
    cvtok || throw(ArgumentError("unable to convert to "
                                 * "designated narrow type $T"))
    ldiv!(F,X)
    Xd = DT.(X)
    Rd = B - A * Xd
    xnorms = [norm(view(Xd,:,j),1) for j=1:nrhs]
    # ω₂ from Arioli, Duff, Ruiz SIAM J.Matrix Anal.Appl. 13, 138 (1992)
    relresnorms = [norm(view(Rd,:,j),Inf) / (anorm * xnorms[j] + bnorms[j])
                   for j=1:nrhs]
    verbose && println("initial max rel resid norm: ",maximum(relresnorms))
    if all(relresnorms .< tolf)
        return Xd, true
    end
    for iter=1:maxiter
        R = T.(Rd)
        ldiv!(F,R)
        Xd .+= DT.(R)
        Rd = B - A * Xd

        relresnorms = [norm(view(Rd,:,j),Inf) / (anorm * xnorms[j] + bnorms[j])
                   for j=1:nrhs]
        verbose && println("iter $iter max rel resid norm: ",maximum(relresnorms))
        if all(relresnorms .< tolf)
            return Xd, true
        end
    end
    return Xd, false
end
