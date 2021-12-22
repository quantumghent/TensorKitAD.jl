struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
		qr(A) -> Tuple{AbstractMatrix, AbstractMatrix}
private QR method, call LinearAlgebra.qr to achieve forward calculation, while
return Tuple type
"""
function qr(A)
	res = LinearAlgebra.qr(A)
	Matrix(res.Q), res.R
end

"""
		qr(A, pivot) -> Tuple{AbstractMatrix, AbstractMatrix, AbstractVector}
"""
function qr(A::AbstractMatrix, pivot::Val{true})
    res = LinearAlgebra.qr(A, pivot)
    Q, R, P = Matrix(res.Q), res.R, res.P
end

"""
		lq(A) -> Tuple{AbstractMatrix, AbstractMatrix}
"""
function lq(A)
		res = LinearAlgebra.lq(A)
    res.L, Matrix(res.Q)
end


function trtrs!(c1::Char, c2::Char, c3::Char, r::AbstractMatrix, b::AbstractVecOrMat)
	#@show rank(r),size(r);
	#@show norm(r),norm(b),norm(inv(r))
	LinearAlgebra.LAPACK.trtrs!(c1, c2, c3, r, b)
end

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix
copy the lower triangular to upper triangular.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i=1:m
        A[i,i] = real(A[i,i])
        for j=i+1:n
            @inbounds A[i,j] = conj(A[j,i])
        end
    end
    A
end

"""
    qr_back_fullrank(q, r, dq, dr) -> Matrix
backward for QR decomposition, for input matrix (in forward pass) with M > N.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
function qr_back_fullrank(q, r, dq, dr)
    dqnot0 = !(typeof(dq) <: Nothing)
    drnot0 = !(typeof(dr) <: Nothing)
    (!dqnot0 && !drnot0) && return nothing
    ex = drnot0 && dqnot0 ? (r*dr' - dq'*q) : (dqnot0 ? (-dq'*q) : (r*dr'))
    b = (dqnot0 ? dq : ZeroAdder()) + q*copyltu!(ex);
    trtrs!('U', 'N', 'N', r, do_adjoint(b))'
end

do_adjoint(A::Matrix) = Matrix(A')

"""
    qr_back(A, q, r, dq, dr) -> Matrix
backward for QR decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function qr_back(A, oq, or, odq, odr)
    dqnot0 = !(typeof(odq) <: Nothing)
    drnot0 = !(typeof(odr) <: Nothing)
    (!dqnot0 && !drnot0) && return nothing

	N = size(or,2);
	M = 0;
	ref = or[1,1];
	while M<size(or,1)
		abs(or[M+1,M+1]/ref)<1e-12 && break
		M+=1
	end

	q = view(oq,:,1:M)
	dq = dqnot0 ? view(odq,:,1:M) : odq;
	r = view(or,1:M,:)
	dr = drnot0 ? view(odr,1:M,:) : odr;

    N == M && return qr_back_fullrank(q, r, dq ,dr)

    B = view(A,:,M+1:N)
    U = view(r,:,1:M)
    D = view(r,:,M+1:N)
    if drnot0
        dD = view(dr,:,M+1:N);
        da = qr_back_fullrank(q, U, dqnot0 ? (dq+B*dD') : (B*dD'), view(dr,:,1:M));
        db = q*dD
    else
        da = qr_back_fullrank(q, U, dq, nothing);
        db = zero(B)
    end
    hcat(da, db)
end

"""
    lq_back(A, l, q, dl, dq) -> Matrix
backward for LQ decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function lq_back_fullrank(L, Q, dL, dQ)
    M = ZeroAdder()
    dL === nothing || (M += L'*dL)
    dQ === nothing || (M -= dQ*Q')
    C = copyltu!(M)*Q
    if dQ !== nothing
        C += dQ
    end
    #inv(L)' * C
    trtrs!('L', 'C', 'N', L, C)
end

"""
    lq_back(A, L, Q, dL, dQ) -> Matrix
backward for QR decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    HaiJun's paper.
"""
function lq_back(A, oL, oQ, odL, odQ)
    dunot0 = !(typeof(odQ) <: Nothing)
    dlnot0 = !(typeof(odL) <: Nothing)
    (!dunot0 && !dlnot0) && return nothing

	N = size(oL,1);
	M = 0;
	ref = oL[1,1];
	while M < size(oL,2)
		abs(oL[M+1,M+1]/ref)<1e-12 && break
		M+=1
	end

	L = view(oL,:,1:M)
	dL = dlnot0 ? view(odL,:,1:M) : odL;
	Q = view(oQ,1:M,:)
	dQ = dunot0 ? view(odQ,1:M,:) : odQ;

    M == N && return lq_back_fullrank(L, Q, dL, dQ)
    B = view(A,M+1:N,:)
    U = view(L,1:M,:)
    D = view(L,M+1:N,:)
    if dlnot0
        dD = view(dL,M+1:N,:);
        da = lq_back_fullrank(U, Q, view(dL,1:M,:), dunot0 ? (dQ+dD'*B) : (dD'*B));
        db = dD*Q
    else
        da = lq_back_fullrank(U, Q, nothing, dQ);
        db = zero(B)
    end
    vcat(da, db)
end
