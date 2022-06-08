@non_differentiable TensorOperations.cached_similar_from_indices(args...)
@non_differentiable TensorOperations.similar_from_indices(args...)

ChainRulesCore.rrule(::typeof(TensorOperations.scalar),arg) =
    TensorOperations.scalar(arg),v -> (NoTangent(),fill!(similar(arg), v))


_flipconj(s::Symbol) = s == :C ? :N : :C;

function ChainRulesCore.rrule(::typeof(TensorOperations.contract!),α,A,CA,B,CB,β,C,oindA,cindA,oindB,cindB,leftind,rightind,syms=nothing)
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



            c_dA = TensorOperations.contract!(α',v,vC,B,fCB,zero(β),similar(A,eltype(res)),oindv,cindv,cindB,oindB,invA,())

            (!(eltype(A)<:Complex) && (eltype(c_dA)<:Complex)) ? real(c_dA) : c_dA
        end
        dCA = NoTangent()

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

            c_dB = TensorOperations.contract!(α',A,fCA,v,vC,zero(β),similar(B,eltype(res)),cindA,oindA,cindv,oindv,invB,())
            (!(eltype(B)<:Complex) && (eltype(c_dB)<:Complex)) ? real(c_dB) : c_dB
        end

        dCB = NoTangent()
        dβ= @thunk dot(C,v)
        dC = @thunk β'*v

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
        dα = @thunk begin
            t = res - β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = @thunk begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...));
            c_dA = TensorOperations.add!(CA == :N ? α' : α,v,CA,zero(β),similar(A,eltype(res)),invCperm,())
            (!(eltype(A)<:Complex) && (eltype(c_dA)<:Complex)) ? real(c_dA) : c_dA
        end

        dCA = NoTangent()

        dβ = @thunk (dot(C,v))

        dC = @thunk (β'*v)

        dleftind = NoTangent()
        drightind = NoTangent()

        return NoTangent(),dα,dA,dCA,dβ,dC,dleftind,drightind
    end
    res,pullback
end

function ChainRulesCore.rrule(::typeof(TensorOperations.trace!),α,A,CA,β,C,leftind,rightind,cind1,cind2)
    res = TensorOperations.trace!(α,A,CA,β,copy(C),leftind,rightind,cind1,cind2);

    function pullback(v)

        dα = @thunk begin
            t = res-β*C
            if α != zero(α)
                t/=α
            end
            dot(t,v)
        end

        dA = @thunk begin
            invCperm = TupleTools.invperm(tuple(leftind...,rightind...,cind1...,cind2...));
            #nli = invCperm[1:numout(A)]
            #nri = invCperm[numout(A)+1:end]

            tracer = TensorOperations.similar_from_indices(eltype(A), cind1, cind2,A,CA)
            tracer = one(tracer);

            TensorOperations.contract!(CA == :N ? α' : α,v,CA,tracer,CA,zero(α),similar(A),ntuple(x->x,length(leftind)+length(rightind)),(),ntuple(x->x,length(cind1)+length(cind2)),(),invCperm,());

        end

        dCA = NoTangent()

        dβ = @thunk(dot(C,v))
        dC = @thunk(β'*v)
        dleftind = NoTangent()
        drightind = NoTangent()
        dcind1 = NoTangent()
        dcind2 = NoTangent()

        return NoTangent(),dα,dA,dCA,dβ,dC,dleftind,drightind,dcind1,dcind2
    end
    res,pullback
end
