import LinearAlgebra.generic_matmatmul!

function generic_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
             tA::AbstractChar, tB::AbstractChar,
             A::StridedMatrix{DoubleFloat{T}},
             B::StridedMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}

    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if (nthreads() > 1) && (Float64(mB)*Float64(mA) > gemm_mt_threshold[])
        _mt_generic_matmatmul!(C,tA,tB,A,B)
    else
        _generic_matmatmul!(C,tA,tB,A,B)
    end
    C
end


function _mt_generic_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
              tA::AbstractChar, tB::AbstractChar,
              A::StridedMatrix{DoubleFloat{T}},
              B::StridedMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    if has_offset_axes(C, A, B)
        throw(ArgumentError("offset axes are not supported"))
    end
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if  mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result matrix C dimensions $(size(C)), needs ($mA,$nB)"))
    end

    if tA == 'N'
        At = Matrix(transpose(A))
    else
        At = A
    end

    Blines = [zeros(DoubleFloat{T},mB) for id in 1:nthreads()]
    if tB == 'N'
        @threads  for j = 1:nB
            id = threadid()
            Bline = Blines[id]
            gemm_kernN(C, At, B, Bline, j, mB, mA)
        end
    else
        @threads  for j = 1:nB
            id = threadid()
            Bline = Blines[id]
            gemm_kernT(C, At, B, Bline, j, mB, mA)
        end
    end
    C
end

function _generic_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
                            tA::AbstractChar, tB::AbstractChar,
              A::StridedMatrix{DoubleFloat{T}},
              B::StridedMatrix{DoubleFloat{T}}) where {T <: AbstractFloat}
    if has_offset_axes(C, A, B)
        throw(ArgumentError("offset axes are not supported"))
    end
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if  mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result matrix C dimensions $(size(C)), needs ($mA,$nB)"))
    end

    if tA == 'N'
        At = Matrix(transpose(A))
    else
        At = A
    end

    Bline = zeros(DoubleFloat{T},mB)
    for j = 1:nB
        @inbounds begin
            if tB == 'N'
                @simd for k = 1:mB
                    Bline[k] = B[k,j]
                end
            else
                @simd for k = 1:mB
                    Bline[k] = B[j,k]
                end
            end
            for i=1:mA
                asum = dot(view(At,:,i),Bline,Vec{Npref,T})
                C[i,j] = asum
            end
        end
    end
    C
end

# isolate the middle loops so that the inference engine has a fair chance

function gemm_kernN(C::StridedMatrix{DoubleFloat{T}}, At, B, Bline, j, mB, mA) where T
    @inbounds begin
        @simd for k = 1:mB
            Bline[k] = B[k,j]
        end
        for i=1:mA
            asum = dot(view(At,:,i),Bline,Vec{Npref,T})
            C[i,j] = asum
        end
    end
end

function gemm_kernT(C::StridedMatrix{DoubleFloat{T}}, At, B, Bline, j, mB, mA) where T
    @inbounds begin
        @simd for k = 1:mB
            Bline[k] = B[j,k]
        end
    end
    for i=1:mA
        asum = dot(view(At,:,i),Bline,Vec{Npref,T})
        C[i,j] = asum
    end
end

function generic_matmatmul!(C::StridedMatrix{Complex{DoubleFloat{T}}},
                            tA::AbstractChar, tB::AbstractChar,
              A::StridedMatrix{Complex{DoubleFloat{T}}},
              B::StridedMatrix{Complex{DoubleFloat{T}}}) where {T <: AbstractFloat}
    if has_offset_axes(C, A, B)
        throw(ArgumentError("offset axes are not supported"))
    end
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if  mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result matrix C dimensions $(size(C)), needs ($mA,$nB)"))
    end

    (tB ∈ ['N','C','T']) || throw(ArgumentError("unrecognized value of tB"))
    (tA ∈ ['N','C','T']) || throw(ArgumentError("unrecognized value of tA"))

    reAt = Matrix{DoubleFloat{T}}(undef, nA, mA)
    imAt = Matrix{DoubleFloat{T}}(undef, nA, mA)
    if tA == 'N'
        @inbounds for j=1:nA
            @simd for i=1:mA
                z = A[i,j]
                reAt[j,i] = z.re
                imAt[j,i] = z.im
            end
        end
    elseif tA == 'C'
        @inbounds for i=1:mA
            @simd for j=1:nA
                z = A[j,i]
                reAt[j,i] = z.re
                imAt[j,i] = -z.im
            end
        end
    elseif tA == 'T'
        @inbounds for i=1:mA
            @simd for j=1:nA
                z = A[j,i]
                reAt[j,i] = z.re
                imAt[j,i] = z.im
            end
        end
    end
    reBline = zeros(DoubleFloat{T},mB)
    imBline = zeros(DoubleFloat{T},mB)
    for j = 1:nB
        @inbounds begin
            if tB == 'N'
                @simd for k = 1:mB
                    z = B[k,j]
                    reBline[k] = z.re
                    imBline[k] = z.im
                end
            elseif tB == 'C'
                @simd for k = 1:mB
                    z = B[j,k]
                    reBline[k] = z.re
                    imBline[k] = -z.im
                end
            else
                @simd for k = 1:mB
                    z = B[j,k]
                    reBline[k] = z.re
                    imBline[k] = z.im
                end
            end
            for i=1:mA
                asum1 = dot(view(reAt,:,i),reBline,Vec{Npref,T})
                asum2 = dot(view(reAt,:,i),imBline,Vec{Npref,T})
                asum3 = dot(view(imAt,:,i),reBline,Vec{Npref,T})
                asum4 = dot(view(imAt,:,i),imBline,Vec{Npref,T})
                C[i,j] = complex(asum1-asum4, asum2+asum3)
            end
        end
    end
    C
end
