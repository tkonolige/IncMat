
import Base: setindex!, getindex

type IncMat{T}
    A :: Array{T, 2}
    blocksize :: Tuple{Int, Int}
    dirty :: Array{Bool, 2}
    shape :: Tuple{Int, Int}
end

function ind(ind, blocksize)
    xb = div(ind[1]-1, blocksize[1]) + 1
    xi = (ind[1]-1) % blocksize[1] + 1
    yb = div(ind[2]-1, blocksize[2]) + 1
    yi = (ind[2]-1) % blocksize[2] + 1
    xb, yb, xi, yi
end

function getindex{T}(A :: IncMat{T}, inds...)
    xb, yb, xi, yi = ind(inds, A.blocksize)
    A.A[xb,yb][xi,yi]
end

function setindex!{T}(A :: IncMat{T}, val :: eltype(T), inds...)
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

function from_mat{T}(A :: Array{T}, blocksize :: Tuple{Int, Int})
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

function mult!{T}(A :: IncMat, B :: IncMat, C :: IncMat{T})
    # TODO: check blocksize
    if A.shape[2] != B.shape[1] || A.shape[1] != C.shape[1] || B.shape[2] != C.shape[2]
        error("Matrix Dimensions do not match")
    end

    is = size(A.A, 1)
    js = size(B.A, 2)
    ks = size(A.A, 2)
    tmp = Array{T}((is, js, ks))
    () -> begin
        for i in 1:is
            for j in 1:js
                # TODO: could probably store dirty status
                if any(A.dirty[i,:]) || any(A.dirty[:,j]) # check if row/col is dirty
                    C.A[i,j] = spzeros(C.blocksize...)
                    for k in 1:ks
                        if A.dirty[i,k] || B.dirty[k,j] # check if blocks are dirty
                            tmp[i,j,k] = A.A[i,k] * B.A[k,j]
                        end
                        C.A[i,j] += tmp[i,j,k]
                    end
                    C.dirty[i,j] = true
                end
            end
        end
        # TODO: more efficient dirty?
        fill!(A.dirty, false)
        fill!(B.dirty, false)
    end
end

A = from_mat(speye(100), (2,2))
B = from_mat(speye(100), (2,2))
C = from_mat(speye(100), (2,2))
f = mult!(A,B,C)
