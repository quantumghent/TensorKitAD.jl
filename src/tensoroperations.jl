function ChainRulesCore.rrule(::typeof(TensorOperations.cached_similar_from_indices),args...)
    TensorOperations.cached_similar_from_indices(args...),x->(DoesNotExist(),[DoesNotExist() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.similar_from_indices),args...)
    TensorOperations.similar_from_indices(args...),x->(DoesNotExist(),[DoesNotExist() for a in args]...)
end

function ChainRulesCore.rrule(::typeof(TensorOperations.scalar),arg)
    function pullback(v)
        NO_FIELDS,fill!(similar(arg), v)
    end
    TensorOperations.scalar(arg),pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.contract!),α,A,CA,B,CB,β,C,oindA,cindA,oindB,cindB,leftind,rightind,syms=nothing)
    _flipconj(s::Symbol) = s == :C ? :N : :C;

    res = TensorOperations.contract!(α,A,CA,B,CB,β,copy(C),oindA,cindA,oindB,cindB,leftind,rightind,syms);

    function pullback(v)
        dα = @thunk begin
            t = res-β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
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

            TensorOperations.contract!(α',v,vC,B,fCB,zero(β),zero(A),oindv,cindv,cindB,oindB,invA,())
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

            TensorOperations.contract!(α',A,fCA,v,vC,zero(β),zero(B),cindA,oindA,cindv,oindv,invB,())
        end

        dCB = DoesNotExist()
        dβ= @thunk begin
            dot(C,v)
        end
        dC = @thunk begin
            β'*v
        end

        doindA = DoesNotExist()
        dcindA = DoesNotExist()
        doindB = DoesNotExist()
        dcindB = DoesNotExist()
        dleftind = DoesNotExist()
        drightind = DoesNotExist()
        dsyms = DoesNotExist()

        return NO_FIELDS,dα,dA,dCA,dB,dCB,dβ,dC,doindA,dcindA,doindB,dcindB,dleftind,drightind,dsyms
    end

    res,pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.add!),α,A,CA,β,C,leftind,rightind)
    res = TensorOperations.add!(α,A,CA,β,copy(C),leftind,rightind);

    function pullback(v)
        dα = @thunk begin
            t = res - β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = @thunk begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));
            TensorOperations.add!(CA == :N ? α' : α,v,CA,zero(β),zero(A),invCperm,())
        end

        dCA = DoesNotExist()

        dβ = @thunk(dot(C,v))

        dC = @thunk(β'*v)
        dleftind = DoesNotExist()
        drightind = DoesNotExist()

        return NO_FIELDS,dα,dA,dCA,dβ,dC,dleftind,drightind
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

        dCA = DoesNotExist()

        dβ= @thunk(dot(orig_C,v))
        dC = @thunk(β'*v)
        dleftind = DoesNotExist()
        drightind = DoesNotExist()
        dcind1 = DoesNotExist()
        dcind2 = DoesNotExist()

        return NO_FIELDS,dα,dA,dCA,dβ,dC,dleftind,drightind,dcind1,dcind2
    end
    res,pullback
end
=#
