
module Inc

export IncMat, copy, getindex, setindex!, from_mat, manifest, mult_full!,
mult!, apply_elem_bin_op!

import Base: setindex!, getindex, copy

type IncMat{T}
    A :: Array{T, 2}
    blocksize :: Tuple{Int, Int}
    dirty :: Array{Bool, 2}
    shape :: Tuple{Int, Int}
end

copy(A :: IncMat) = IncMat(copy(A.A), A.blocksize, copy(A.dirty), A.shape)

@inline function ind(ind, blocksize)
    xb = div(ind[1]-1, blocksize[1]) + 1
    xi = (ind[1]-1) % blocksize[1] + 1
    yb = div(ind[2]-1, blocksize[2]) + 1
    yi = (ind[2]-1) % blocksize[2] + 1
    xb, yb, xi, yi
end

@inline function getindex{T}(A :: IncMat{T}, inds...)
    xb, yb, xi, yi = ind(inds, A.blocksize)
    A.A[xb,yb][xi,yi]
end

@inline function setindex!{T}(A :: IncMat{T}, val :: eltype(T), inds...)
    xb, yb, xi, yi = ind(inds, A.blocksize)
    A.dirty[xb,yb] = true
    A.A[xb,yb][xi,yi] = val
end

function from_mat{T}(A :: SparseMatrixCSC{T}, blocksize :: Tuple{Int, Int})
    xb, yb, _, _ = ind(size(A), blocksize)

    # initialize inc mat
    B = IncMat{SparseMatrixCSC{T}}(Array{SparseMatrixCSC{T}}((xb, yb)),
            blocksize, fill(true, (xb, yb)), size(A))
    for x in 1:xb
        for y in 1:yb
            B.A[x,y] = spzeros(blocksize...)
        end
    end

    # fill inc mat entries
    # TODO: this is probably slow
    rows = rowvals(A)
    vals = nonzeros(A)
    for i in 1:size(A, 2)
        for j in nzrange(A, i)
            B[i,rows[j]] = vals[j]
        end
    end

    B
end

function from_mat{T}(A :: Array{T,2}, blocksize :: Tuple{Int, Int})
    xb, yb, _, _ = ind(size(A), blocksize)

    # initialize inc mat
    # TODO: is probably slow
    B = IncMat{Array{T}}(Array{Array{T}}((xb, yb)),
            blocksize, fill(true, (xb, yb)), size(A))
    for x in 1:xb
        for y in 1:yb
            B.A[x,y] = zeros(blocksize...)
        end
    end

    for x in 1:size(A, 1)
        for y in 1:size(A, 2)
            B[x,y] = A[x,y]
        end
    end

    B
end

from_mat{T}(A :: Array{T,1}, blocksize :: Tuple{Int, Int}) =
    from_mat(reshape(A, (size(A,1), 1)), blocksize)

function manifest{T}(A :: IncMat{Array{T}})
    xs, ys = A.shape
    B = Array{T}(A.shape)
    for i in 1:xs
        for j in 1:ys
            B[i,j] = A[i,j]
        end
    end
    B
end

function manifest{T}(A :: IncMat{SparseMatrixCSC{T}})
    x, y = size(A.A)
    xb, yb = A.blocksize
    xs = x * xb
    ys = y * yb
    I = Array{Int64,1}()
    J = Array{Int64,1}()
    V = Array{Int64,1}()
    for i in 1:x
        for j in 1:y
            xx, yy, vv = findnz(A.A[i,j])
            append!(I, map(x -> x+(i-1)*xb, xx))
            append!(J, map(x -> x+(i-1)*yb, yy))
            append!(V, vv)
        end
    end

    sparse(I, J, V, A.shape[1], A.shape[2])
end

function mult_full!(A :: IncMat, B :: IncMat, C :: IncMat)
    # TODO: check blocksize
    if A.shape[2] != B.shape[1] || A.shape[1] != C.shape[1] || B.shape[2] != C.shape[2]
        error("Matrix Dimensions do not match")
    end
    for i in 1:size(A.A, 1)
        for j in 1:size(B.A, 2)
            C.A[i,j] = spzeros(C.blocksize...)
            for k in 1:size(A.A, 2)
                C.A[i,j] += A.A[i,k] * B.A[k,j]
            end
        end
    end
end

function mult!{T}(A :: IncMat, B :: IncMat, C :: IncMat{T}, store_elem=true, check_block=true)
    if A.shape[2] != B.shape[1] || A.shape[1] != C.shape[1] || B.shape[2] != C.shape[2]
        throw(DimensionMismatch("Matrix dimensions do not match"))
    end
    if (A.blocksize[2] != B.blocksize[1] || A.blocksize[1] != C.blocksize[1]
        || B.blocksize[2] != C.blocksize[2])
        throw(DimensionMismatch("Block dimensions do not match"))
    end

    is = size(A.A, 1)
    js = size(B.A, 2)
    ks = size(A.A, 2)
    if store_elem
        tmp = Array{T}((is, js, ks))
    end
    () -> begin
        for i in 1:is
            for j in 1:js
                # TODO: could probably store dirty status per column
                # check if row/col is dirty
                if !check_block || any(A.dirty[i,:]) || any(B.dirty[:,j])
                    if T <: DenseArray
                        C.A[i,j] = zeros(C.blocksize...)
                    else
                        C.A[i,j] = spzeros(C.blocksize...)
                    end
                    for k in 1:ks
                        if store_elem
                            if A.dirty[i,k] || B.dirty[k,j] # check if blocks are dirty
                                tmp[i,j,k] = A.A[i,k] * B.A[k,j]
                            end
                            C.A[i,j] += tmp[i,j,k]
                        else
                            C.A[i,j] += A.A[i,k] * B.A[k,j]
                        end
                    end
                    C.dirty[i,j] = true
                end
            end
        end
        # TODO: more efficient dirty?
        fill!(A.dirty, false)
        fill!(B.dirty, false)
        return
    end
end

"""
Apply an elementwise binary operation to A and B
"""
function apply_elem_bin_op!(A :: IncMat, B :: IncMat, C :: IncMat, f)
    if A.shape != B.shape || A.shape != C.shape
        throw(DimensionMismatch("Matrix dimensions do not match"))
    end
    if A.blocksize != B.blocksize || A.blocksize != C.blocksize
        throw(DimensionMismatch("Block dimensions do not match"))
    end

    x,y = size(A.A)
    () -> begin
        for i in 1:x
            for j in 1:y
                if A.dirty[i,j] || B.dirty[i,j]
                    C.A[i,j] = f(A.A[i,j], B.A[i,j])
                    C.dirty[i,j]
                    A.dirty[i,j] = false
                    B.dirty[i,j] = false
                end
            end
        end
    end
end

end

using Inc

A = from_mat(speye(1000), (10,10))
B = from_mat(ones(1000), (10,1))
C = from_mat(zeros(1000), (10,1))
f = mult!(A,B,C)
f()
fill!(A.dirty, true)
@time f()
@time f()
