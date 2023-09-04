import LinearAlgebra.generic_matmatmul!

const gemm_mt_threshold = Ref(64.0)
mt_thresholds[:gemm] = gemm_mt_threshold;

function generic_matmatmul!(C::AbstractMatrix{DoubleFloat{T}}, tA, tB, A::AbstractMatrix{DoubleFloat{T}}, B::AbstractMatrix{DoubleFloat{T}},
                            _add::MulAddMul=MulAddMul()) where T
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    mC, nC = size(C)

    if iszero(_add.alpha)
        return _rmul_or_fill!(C, _add.beta)
    end
    if mA == nA == mB == nB == mC == nC == 2
        return matmul2x2!(C, tA, tB, A, B, _add)
    end
    if mA == nA == mB == nB == mC == nC == 3
        return matmul3x3!(C, tA, tB, A, B, _add)
    end
    _xgeneric_matmatmul!(C, tA, tB, A, B, _add)
end
function _xgeneric_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
                            tA::AbstractChar, tB::AbstractChar,
                            A::StridedMatrix{DoubleFloat{T}},
                            B::StridedMatrix{DoubleFloat{T}},
                            _add::MulAddMul;
                            ntasks = _default_nt()
                            ) where {T <: AbstractFloat}

    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if ((ntasks > 1) && (Float64(mB)*Float64(mA) > gemm_mt_threshold[])
        && _add.alpha == 1 && _add.beta == 0)
        _mt_generic_matmatmul!(C,tA,tB,A,B,ntasks)
    else
        __generic_matmatmul!(C,tA,tB,A,B,_add)
    end
    C
end


function _mt_generic_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
                                tA::AbstractChar, tB::AbstractChar,
                                A::StridedMatrix{DoubleFloat{T}},
                                B::StridedMatrix{DoubleFloat{T}},
                                ntasks) where {T <: AbstractFloat}
    require_one_based_indexing(C, A, B)
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

    Blines = [zeros(DoubleFloat{T},mB) for id in 1:ntasks]
    if tB == 'N'
        @threads  for j = 1:ntasks
            Bline = Blines[j]
            for jj in _part_range(1:nB, ntasks, j)
                gemm_kernN(C, At, B, Bline, jj, mB, mA)
            end
        end
    else
        @threads  for j = 1:ntasks
            Bline = Blines[j]
            for jj in _part_range(1:nB, ntasks, j)
                gemm_kernT(C, At, B, Bline, jj, mB, mA)
            end
        end
    end
    C
end

function __generic_matmatmul!(C::StridedMatrix{DoubleFloat{T}},
                            tA::AbstractChar, tB::AbstractChar,
                             A::StridedMatrix{DoubleFloat{T}},
                             B::StridedMatrix{DoubleFloat{T}},
                             _add::MulAddMul) where {T <: AbstractFloat}
    require_one_based_indexing(C, A, B)
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if  mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result matrix C dimensions $(size(C)), needs ($mA,$nB)"))
    end

    if iszero(_add.alpha) || isempty(A) || isempty(B)
        return _rmul_or_fill!(C, _add.beta)
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
                # benchmarking showed view to beat uview here
                asum = _dot(view(At,:,i),Bline,Vec{Npref,T})
                _modify!(_add, asum, C, (i,j))
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
            asum = _dot(uview(At,:,i),Bline,Vec{Npref,T})
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
        asum = _dot(uview(At,:,i),Bline,Vec{Npref,T})
        C[i,j] = asum
    end
end

function _generic_matmatmul!(C::StridedMatrix{Complex{DoubleFloat{T}}},
                            tA::AbstractChar, tB::AbstractChar,
                            A::StridedMatrix{Complex{DoubleFloat{T}}},
                            B::StridedMatrix{Complex{DoubleFloat{T}}},
                            _add::MulAddMul=MulAddMul()
                            ) where {T <: AbstractFloat}
    require_one_based_indexing(C, A, B)
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

    if iszero(_add.alpha) || isempty(A) || isempty(B)
        return _rmul_or_fill!(C, _add.beta)
    end

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
    if ((ntasks > 1) && (Float64(mB)*Float64(mA) > gemm_mt_threshold[])
        && _add.alpha == 1 && _add.beta == 0)
        _mt_gemmwrap(C,mA,mB,nB,tB,reAt,imAt,B,Vec{Npref,T})
    else
        _gemmcore(C,mA,mB,nB,tB,reAt,imAt,B,Vec{Npref,T},_add)
    end
    C
end

function _gemmcore(C::AbstractMatrix{Complex{DoubleFloat{T}}},
                   mA,mB,nB,tB,reAt,imAt,B,::Type{Vec{N,T}},
                   _add::MulAddMul
                   ) where {N,T <: AbstractFloat}
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
                asum1 = _dot(uview(reAt,:,i),reBline,Vec{N,T})
                asum2 = _dot(uview(reAt,:,i),imBline,Vec{N,T})
                asum3 = _dot(uview(imAt,:,i),reBline,Vec{N,T})
                asum4 = _dot(uview(imAt,:,i),imBline,Vec{N,T})
                s = complex(asum1-asum4, asum2+asum3)
                _modify!(_add, s, C, (i,j))
            end
        end
    end
end

function _mt_gemmwrap(C::AbstractMatrix{Complex{DoubleFloat{T}}},
                      mA,mB,nB,tB,reAt,imAt,B,::Type{Vec{N,T}},
                      nt::Integer
                      ) where {N, T <: AbstractFloat}
    reBlines = [zeros(DoubleFloat{T},mB) for id in 1:nt]
    imBlines = [zeros(DoubleFloat{T},mB) for id in 1:nt]
    if tB == 'N'
        @threads for j = 1:nt
            reBline = reBlines[j]
            imBline = imBlines[j]
            for jj in _part_range(1:nB, nt, j)
            _mt_gemmcoreN(C, reAt, imAt, B, reBline, imBline, jj, mB, mA,
                Vec{N,T})
            end
        end
    elseif tB == 'C'
        @threads for j = 1:nt
            reBline = reBlines[j]
            imBline = imBlines[j]
            for jj in _part_range(1:nB, nt, j)
                _mt_gemmcoreC(C, reAt, imAt, B, reBline, imBline, jj, mB, mA,
                              Vec{N,T})
            end
        end
    elseif tB == 'T'
        @threads for j = 1:nt
            reBline = reBlines[j]
            imBline = imBlines[j]
            for jj in _part_range(1:nB, nt, j)
                _mt_gemmcoreT(C, reAt, imAt, B, reBline, imBline, jj, mB, mA,
                              Vec{N,T})
            end
        end
    end
    nothing
end

function _mt_gemmcoreN(C, reAt, imAt, B, reBline, imBline, j, mB, mA,
                       ::Type{Vec{N,T}}) where {N,T}
    @inbounds begin
        @simd for k = 1:mB
            z = B[k,j]
            reBline[k] = z.re
            imBline[k] = z.im
        end
        for i=1:mA
            asum1 = _dot(uview(reAt,:,i),reBline,Vec{N,T})
            asum2 = _dot(uview(reAt,:,i),imBline,Vec{N,T})
            asum3 = _dot(uview(imAt,:,i),reBline,Vec{N,T})
            asum4 = _dot(uview(imAt,:,i),imBline,Vec{N,T})
            C[i,j] = complex(asum1-asum4, asum2+asum3)
        end
    end
    nothing
end

function _mt_gemmcoreT(C, reAt, imAt, B, reBline, imBline, j, mB, mA,
                       ::Type{Vec{N,T}}) where {N,T}
    @inbounds begin
        @simd for k = 1:mB
            z = B[j,k]
            reBline[k] = z.re
            imBline[k] = z.im
        end
        for i=1:mA
            asum1 = _dot(uview(reAt,:,i),reBline,Vec{N,T})
            asum2 = _dot(uview(reAt,:,i),imBline,Vec{N,T})
            asum3 = _dot(uview(imAt,:,i),reBline,Vec{N,T})
            asum4 = _dot(uview(imAt,:,i),imBline,Vec{N,T})
            C[i,j] = complex(asum1-asum4, asum2+asum3)
        end
    end
    nothing
end

function _mt_gemmcoreC(C, reAt, imAt, B, reBline, imBline, j, mB, mA,
                       ::Type{Vec{N,T}}) where {N,T}
    @inbounds begin
        @simd for k = 1:mB
            z = B[j,k]
            reBline[k] = z.re
            imBline[k] = -z.im
        end
        for i=1:mA
            asum1 = _dot(uview(reAt,:,i),reBline,Vec{N,T})
            asum2 = _dot(uview(reAt,:,i),imBline,Vec{N,T})
            asum3 = _dot(uview(imAt,:,i),reBline,Vec{N,T})
            asum4 = _dot(uview(imAt,:,i),imBline,Vec{N,T})
            C[i,j] = complex(asum1-asum4, asum2+asum3)
        end
    end
    nothing
end
