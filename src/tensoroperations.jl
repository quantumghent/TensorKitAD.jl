function ChainRulesCore.rrule(::typeof(TensorOperations.cached_similar_from_indices),args...)
    TensorOperations.cached_similar_from_indices(args...),x->(NoTangent(),[NoTangent() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.similar_from_indices),args...)
    TensorOperations.similar_from_indices(args...),x->(NoTangent(),[NoTangent() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.scalar),arg)
    function pullback(v)
        NoTangent(),fill!(similar(arg), v)
    end
    TensorOperations.scalar(arg),pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.contract!),α,A,CA,B,CB,β,C,oindA,cindA,oindB,cindB,leftind,rightind,syms=nothing)
    _flipconj(s::Symbol) = s == :C ? :N : :C;

    res = TensorOperations.contract!(α,A,CA,B,CB,β,copy(C),oindA,cindA,oindB,cindB,leftind,rightind,syms);

    function pullback(v)
        dα = begin
            t = res-β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));
            oindv = invCperm[1:length(oindA)];
            cindv = invCperm[length(oindA)+1:end];

            invA = TupleTools.invperm(tuple(oindA...,cindA...));

            fCB = _flipconj(CB);
            vC = :N
            if CA == :C
                fCB = CB;
                vC = :C
            end

            TensorOperations.contract!(α',v,vC,B,fCB,zero(β),zero(A),oindv,cindv,cindB,oindB,invA,())
        end
        dCA = NoTangent()

        dB = begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));
            oindv = invCperm[1:length(oindA)];
            cindv = invCperm[length(oindA)+1:end];


            invB = TupleTools.invperm(tuple(cindB...,oindB...));

            fCA = _flipconj(CA);
            vC = :N
            if CB == :C
                fCA = CA;
                vC = :C
            end

            TensorOperations.contract!(α',A,fCA,v,vC,zero(β),zero(B),cindA,oindA,cindv,oindv,invB,())
        end

        dCB = NoTangent()
        dβ= begin
            dot(C,v)
        end
        dC = begin
            β'*v
        end

        doindA = NoTangent()
        dcindA = NoTangent()
        doindB = NoTangent()
        dcindB = NoTangent()
        dleftind = NoTangent()
        drightind = NoTangent()
        dsyms = NoTangent()

        return NoTangent(),dα,dA,dCA,dB,dCB,dβ,dC,doindA,dcindA,doindB,dcindB,dleftind,drightind,dsyms
    end

    res,pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.add!),α,A,CA,β,C,leftind,rightind)
    res = TensorOperations.add!(α,A,CA,β,copy(C),leftind,rightind);

    function pullback(v)
        dα = begin
            t = res - β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));
            TensorOperations.add!(CA == :N ? α' : α,v,CA,zero(β),zero(A),invCperm,())
        end

        dCA = NoTangent()

        dβ = (dot(C,v))

        dC = (β'*v)

        dleftind = NoTangent()
        drightind = NoTangent()

        return NoTangent(),dα,dA,dCA,dβ,dC,dleftind,drightind
    end
    res,pullback
end

#=
#to do trace! properly I guess I'd need isomorphism to be generically defined for matrices?
function ChainRulesCore.rrule(::typeof(TensorOperations.trace!),α,A,CA,β,C,leftind,rightind,cind1,cind2)
    @show "getting traced"
    _flipconj(s::Symbol) = s == :C ? :N : :C;

    orig_C = copy(C);
    res = TensorOperations.trace!(α,A,CA,β,C,leftind,rightind,cind1,cind2);

    function pullback(v)

        dα = @thunk begin
            t = res-beta*orig_C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = @thunk begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));

        end

        dCA = NoTangent()

        dβ= @thunk(dot(orig_C,v))
        dC = @thunk(β'*v)
        dleftind = NoTangent()
        drightind = NoTangent()
        dcind1 = NoTangent()
        dcind2 = NoTangent()

        return NoTangent(),dα,dA,dCA,dβ,dC,dleftind,drightind,dcind1,dcind2
    end
    res,pullback
end
=#
