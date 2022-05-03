function ChainRulesCore.rrule(::typeof(MPSKit.InfiniteMPS),A::AbstractVector;kwargs...)


    #initial guess for CR
    CL = PeriodicArray([isomorphism(Matrix{eltype(A[i])},space(A[mod1(i+1,end)],1),space(A[mod1(i+1,end)],1)) for i in 1:length(A)]);

    (AL,CL) = MPSKit.uniform_leftorth!(copy.(A),CL,A);
    (AR,CR) = MPSKit.uniform_rightorth!(copy.(A),copy.(CL),A);
    C = CL.*CR;

    Cns = norm.(C);
    C./= Cns;
    CL./= sqrt.(Cns);
    CR./= sqrt.(Cns);

    AC = PeriodicArray(AL.*C);
    ts = InfiniteMPS(PeriodicArray(AL),PeriodicArray(AR),PeriodicArray(C),AC)

    function infmps_pullback(x̄)
        # need a better way to handle zerotangents
        AL_bar = x̄.AL isa ZeroTangent ? zero.(AL) : x̄.AL;
        AL_bar = map(zip(AL_bar,AL)) do (a,b)
            a isa ZeroTangent ? zero(b)  : a
        end

        AR_bar = x̄.AR isa ZeroTangent ? zero.(AR) : x̄.AR;
        AR_bar = map(zip(AR_bar,AR)) do (a,b)
            a isa ZeroTangent ? zero(b)  : a
        end

        C_bar = x̄.CR isa ZeroTangent ? zero.(C) : x̄.CR;
        C_bar = map(zip(C_bar,C)) do (a,b)
            a isa ZeroTangent ? zero(b)  : a
        end

        AC_bar = x̄.AC isa ZeroTangent ? zero.(ts.AC) : x̄.AC;
        AC_bar = map(zip(AC_bar,AC)) do (a,b)
            a isa ZeroTangent ? zero(b)  : a
        end

        # absorb AC_bar in AL_bar and C_bar
        AL_bar += AC_bar.*adjoint.(C)
        C_bar += adjoint.(AL).*AC_bar;

        # C_bar -> CL_bar, CR_bar
        CL_bar = C_bar.*adjoint.(CR)
        CR_bar = adjoint.(CL).*C_bar

        lambdas_R = zeros(eltype(first(AL)),length(AL));
        lambdas_L = zeros(eltype(first(AL)),length(AL));
        for i in 1:length(AL)
            lambdas_R[i] = @plansor A[i][1 2;3]*CR[i][3;4]*conj(CR[mod1(i-1,end)][1;5])*conj(AR[i][5 2;4])
            lambdas_R[i] *= Cns[mod1(i-1,end)]

            lambdas_L[i] = @plansor CL[mod1(i-1,end)][1;2]*A[i][2 3;4]*conj(AL[i][1 3;5])*conj(CL[i][5;4])
            lambdas_L[i] *= Cns[i]
        end
        # I think these are always the same
        lambdas_R = PeriodicArray(lambdas_R.^-1);
        lambdas_L = PeriodicArray(lambdas_L.^-1);

        VLs = leftnull.(AL);
        VRs = rightnull.(MPSKit._transpose_tail.(AR));

        XL_bar = map(zip(VLs,AL_bar)) do (V,A)
            V'*A
        end
        XR_bar = map(zip(VRs,AR_bar)) do (V,A)
            MPSKit._transpose_tail(A)*V'
        end

        BL = map(zip(XL_bar,CL)) do (x,c)
            x*inv(c)'
        end
        BR = map(zip(XR_bar,circshift(CR,1))) do (x,c)
            inv(c)'*x
        end


        # first two contributions to Ā
        Ā = map(zip(lambdas_L,circshift(CL,1),VLs,BL)) do (l,cl,vl,b)
            @tensor temp[-1 -2;-3] := conj(l*cl[1;-1])*vl[1 -2;2]*b[2;-3]
        end
        Ā += map(zip(lambdas_R,CR,VRs,BR)) do (l,cr,vr,b)
            @tensor temp[-1 -2;-3] := l'*b[-1;1]*vr[1;2 -2]*conj(cr[-3;2])
        end


        CL_bar += circshift(map(zip(lambdas_L,A,VLs,BL)) do (l,a,v,b)
            TransferMatrix(v,a)*b*l'
        end,-1)

        CR_bar += map(zip(lambdas_R,A,VRs,BR)) do (l,a,v,b)
            l'*b*TransferMatrix(MPSKit._transpose_front(v),a)
        end

        # solve for left bit
        tosol = RecursiveVec(CL_bar);
        lvec = RecursiveVec(map(zip(CL,CR)) do (cl,cr)
            cl
        end)
        rvec = RecursiveVec(map(zip(CL,CR)) do (cl,cr)
            cl*cr*cr'
        end)
        tosol-=rvec*dot(lvec,tosol)/length(lambdas_L);

        (sol,convhist) = linsolve(tosol,tosol,GMRES()) do x
            y = copy(x.vecs);
            for i in 1:length(y)
                y[i] -= TransferMatrix(AL[mod1(i+1,end)],A[mod1(i+1,end)])*x.vecs[mod1(i+1,end)]*lambdas_L[i+1]'
            end
            RecursiveVec(y)
        end
        convhist.converged == 0 && @warn "left failed to converge $(convhist.normres)"
        CL_bar = sol.vecs;


        # solve for right bit
        tosol = RecursiveVec(CR_bar);
        lvec = RecursiveVec(map(zip(CL,CR)) do (cl,cr)
            cl'*cl*cr
        end)
        rvec = RecursiveVec(map(zip(CL,CR)) do (cl,cr)
            cr
        end)
        tosol-=lvec*dot(rvec,tosol)/length(lambdas_L);
        (sol,convhist) = linsolve(tosol,tosol,GMRES()) do x
            y = copy(x.vecs);
            for i in 1:length(y)
                y[i] -= lambdas_R[i]'*x.vecs[mod1(i-1,end)]*TransferMatrix(AR[i],A[i])
            end
            RecursiveVec(y)
        end
        convhist.converged == 0 && @warn "right failed to converge $(convhist.normres)"
        CR_bar = sol.vecs;

        # and finally the contributions to Ā
        Ā += map(zip(lambdas_L,circshift(CL,1),AL,CL_bar)) do (l,cl,al,b)
            @tensor temp[-1 -2;-3] := conj(l*cl[1;-1])*al[1 -2;2]*b[2;-3]
        end
        Ā += map(zip(lambdas_R,CR,AR,circshift(CR_bar,1))) do (l,cr,ar,b)
            @tensor temp[-1 -2;-3] := l'*b[-1;1]*ar[1 -2;2]*conj(cr[-3;2])
        end


        ∂self = NoTangent()


        return ∂self, Ā
    end
    return ts, infmps_pullback
end
