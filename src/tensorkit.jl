#you cannot really define the pullback wrt a function, because you don't have an inner product
@non_differentiable TensorKit.TensorMap(f::Function,storagetype,cod,dom)

@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)

function ChainRulesCore.rrule(::typeof(dot),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(b * c')
        ∂b = @thunk(a * c)
        return (NoTangent(), ∂a, ∂b)
    end
    return dot(a,b),pullback
end

function ChainRulesCore.rrule(::typeof(+),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NoTangent(), c, c)
    return a+b,pullback
end

function ChainRulesCore.rrule(::typeof(-),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NoTangent(),c,-c)
    return a-b,pullback
end

function ChainRulesCore.rrule(::typeof(norm),a::AbstractTensorMap,p)
    p == 2 || throw(ArgumentError("DIY"))

    na = norm(a)
    function pullback(c)
        ∂a = @thunk(a*(c'+c)/(na*2))
        return (NoTangent(), ∂a)
    end
    return na,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(a'*c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::Number)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(dot(a,c))
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::Number,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(dot(b,c))
        ∂b = @thunk(a'*c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(permute),tensor,leftind,rightind=())
    function pullback(c)
        ∂a = @thunk begin
            invperm = TupleTools.invperm(tuple(leftind...,rightind...));

            permute(c,tuple(invperm[1:numout(tensor)]...),tuple(invperm[numout(tensor)+1:end]...))
        end
        return (NoTangent(), ∂a, NoTangent(),NoTangent())
    end

    return permute(tensor,leftind,rightind),pullback
end

function ChainRulesCore.rrule(::Type{<:TensorMap},d::DenseArray,args...)
    function pullback(tm)
        ∂d = @thunk(convert(Array,tm))
        (NoTangent(),∂d,[NoTangent() for a in args]...)
    end
    TensorMap(d,args...),pullback
end

#pullback rule based on tom's krylovkit rule
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), t::AbstractTensorMap;kwargs...)
    T = eltype(t);

    (U,S,V) = tsvd(t;kwargs...);

    F = similar(S);
    for (k,dst) in blocks(F)

        src = blocks(S)[k]

        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            dst[i,j] = (i == j) ? zero(eltype(S)) : 1/(src[j,j]^2-src[i,i]^2+1e-12)
        end
    end


    function pullback(v)
        dU,dS,dV = v

        dA = zero(t);
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            dA += U*_elementwise_mult(dS,one(dS))*V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            J = _elementwise_mult((U'*dU),F)
            dA += U*(J+J')*S*V
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            VpdV = V*dV';
            K = _elementwise_mult(VpdV,F)
            dA += U*S*(K+K')*V
        end
        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.ZeroTangent() && T <: Complex
            L = _elementwise_mult(VpdV,one(F))
            dA += 1/2*U*pinv(S)*(L' - L)*V
        end

        if codomain(t)!=domain(t)
            pru = U*U';
            prv = V'*V;
            dA += (one(pru)-pru)*dU*pinv(S)*V
            dA += U*pinv(S)*dV*(one(prv)-prv)
        end

        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U,S,V), pullback
end


function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end


function ChainRulesCore.rrule(::typeof(leftorth),tensor,leftind=codomainind(tensor),rightind=domainind(tensor);alg=QRpos())

    (permuted,permback) = ChainRulesCore.rrule(permute,tensor,leftind,rightind);
    (q,r) = leftorth(permuted;alg=alg);

    if alg isa TensorKit.QR || alg isa TensorKit.QRpos
        pullback = v-> backwards_leftorth_qr(permuted,q,r,v[1],v[2])
    else
        pullback = v-> @assert false
    end

    (q,r),v-> (NoTangent(),permback(pullback(v))[2], NoTangent(), NoTangent(), NoTangent());
end

function backwards_leftorth_qr(A,q,r,dq,dr)
    out = similar(A);
    dr = dr isa ZeroTangent ? zero(r) : dr;
    dq = dq isa ZeroTangent ? zero(q) : dq;


    if sectortype(A) == Trivial
        copyto!(out.data,qr_back(A.data, q.data, r.data, dq.data, dr.data));
    else
        for b in keys(blocks(A))
            cA = A[b];
            cq = q[b];
            cr = r[b];
            cdq = dq[b];
            cdr = dr[b];

            copyto!(out[b],qr_back(cA, cq, cr, cdq, cdr));
        end
    end
    #@show norm(A),norm(dq),norm(dr),norm(out)
    out
end

function ChainRulesCore.rrule(::typeof(rightorth),tensor,leftind=codomainind(tensor),rightind=domainind(tensor);alg=LQpos())

    (permuted,permback) = ChainRulesCore.rrule(permute,tensor,leftind,rightind);
    (l,q) = rightorth(permuted;alg=alg);

    if alg isa TensorKit.LQ || alg isa TensorKit.LQpos
        pullback = v-> backwards_rightorth_lq(permuted,l,q,v[1],v[2])
    else
        pullback = v-> @assert false
    end

    (l,q),v-> (NoTangent(),permback(pullback(v))[2], NoTangent(), NoTangent(), NoTangent());
end

function backwards_rightorth_lq(A,l,q,dl,dq)
    out = similar(A);
    dl = dl isa ZeroTangent ? zero(l) : dl;
    dq = dq isa ZeroTangent ? zero(q) : dq;

    if sectortype(A) == Trivial
        copyto!(out.data,lq_back(A.data, l.data, q.data, dl.data, dq.data));
    else
        for b in keys(blocks(A))
            cA = A[b];
            cl = l[b];
            cq = q[b];
            cdl = dl[b];
            cdq = dq[b];

            copyto!(out[b],lq_back(cA, cl, cq, cdl, cdq));
        end
    end
    #@show norm(A),norm(l),norm(q),norm(out)
    out
end
