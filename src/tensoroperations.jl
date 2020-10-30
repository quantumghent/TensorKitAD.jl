function ChainRulesCore.rrule(::typeof(TensorOperations.cached_similar_from_indices),args...)
    TensorOperations.cached_similar_from_indices(args...),x->(DoesNotExist(),[DoesNotExist() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.similar_from_indices),args...)
    TensorOperations.similar_from_indices(args...),x->(DoesNotExist(),[DoesNotExist() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.scalar),arg)
    function pullback(v)
        NO_FIELDS,arg/norm(arg) #will fail for norm(arg)==0
    end
    TensorOperations.scalar(arg),pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.contract!),α,A,CA,B,CB,β,C,oindA,cindA,oindB,cindB,leftind,rightind,syms=nothing)
    orig_C = copy(C);
    _flipconj(s::Symbol) = s == :C ? :N : :C;

    function pullback(v)

        dα = @thunk begin
            res = TensorOperations.contract!(one(α),A,CA,B,CB,zero(β),zero(C),oindA,cindA,oindB,cindB,leftind,rightind)
            dot(res,v)
        end

        dA = @thunk begin
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

            TensorOperations.contract!(α',v,vC,B,fCB,zero(β),zero(A),oindv,cindv,cindB,oindB,invA[1:length(oindA)],invA[length(oindA)+1:end])
        end

        dCA = DoesNotExist()

        dB = @thunk begin
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

            TensorOperations.contract!(α',A,fCA,v,vC,zero(β),zero(B),cindA,oindA,cindv,oindv,invB[1:length(cindB)],invB[length(cindB)+1:end])
        end

        dCB = DoesNotExist()
        dβ= @thunk(dot(orig_C,v))
        dC = @thunk(dβ'*v)
        doindA = DoesNotExist()
        dcindA = DoesNotExist()
        doindB = DoesNotExist()
        dcindB = DoesNotExist()
        dleftind = DoesNotExist()
        drightind = DoesNotExist()
        dsyms = DoesNotExist()

        return NO_FIELDS,dα,dA,dCA,dB,dCB,dβ,dC,doindA,dcindA,doindB,dcindB,dleftind,drightind,dsyms
    end
    TensorOperations.contract!(α,A,CA,B,CB,β,C,oindA,cindA,oindB,cindB,leftind,rightind,syms),pullback
end
