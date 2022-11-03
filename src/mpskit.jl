function ChainRulesCore.rrule(::Type{MPSKit.InfiniteMPS}, A::AbstractVector; kwargs...)
    #initial guess for CR
    CL = PeriodicArray([isomorphism(Matrix{eltype(A[i])}, space(A[mod1(i + 1, end)], 1), space(A[mod1(i + 1, end)], 1)) for i in 1:length(A)])

    # Orthogonalize
    (AL, CL) = MPSKit.uniform_leftorth!(copy.(A), CL, A)
    (AR, CR) = MPSKit.uniform_rightorth!(copy.(A), copy.(CL), A)
    C = CL .* CR

    # Normalize
    Cns = norm.(C)
    C ./= Cns
    CL ./= sqrt.(Cns)
    CR ./= sqrt.(Cns)

    AC = PeriodicArray(AL .* C)
    ts = InfiniteMPS(PeriodicArray(AL), PeriodicArray(AR), PeriodicArray(C), AC)

    function infmps_pullback(x̄)
        # need a better way to handle zerotangents
        AL_bar = x̄.AL isa ZeroTangent ? zero.(AL) : x̄.AL
        AL_bar = map(zip(AL_bar, AL)) do (a, b)
            a isa ZeroTangent ? zero(b) : a
        end

        AR_bar = x̄.AR isa ZeroTangent ? zero.(AR) : x̄.AR
        AR_bar = map(zip(AR_bar, AR)) do (a, b)
            a isa ZeroTangent ? zero(b) : a
        end

        C_bar = x̄.CR isa ZeroTangent ? zero.(C) : x̄.CR
        C_bar = map(zip(C_bar, C)) do (a, b)
            a isa ZeroTangent ? zero(b) : a
        end

        AC_bar = x̄.AC isa ZeroTangent ? zero.(ts.AC) : x̄.AC
        AC_bar = map(zip(AC_bar, AC)) do (a, b)
            a isa ZeroTangent ? zero(b) : a
        end

        # absorb AC_bar in AL_bar and C_bar
        AL_bar += AC_bar .* adjoint.(C)
        C_bar += adjoint.(AL) .* AC_bar

        lambdas_R = zeros(eltype(first(AL)), length(AL))
        lambdas_L = zeros(eltype(first(AL)), length(AL))
        for i in 1:length(AL)
            lambdas_R[i] = @plansor A[i][1 2; 3] * CR[i][3; 4] * conj(CR[mod1(i - 1, end)][1; 5]) * conj(AR[i][5 2; 4])
            lambdas_R[i] *= Cns[mod1(i - 1, end)]

            lambdas_L[i] = @plansor CL[mod1(i - 1, end)][1; 2] * A[i][2 3; 4] * conj(AL[i][1 3; 5]) * conj(CL[i][5; 4])
            lambdas_L[i] *= Cns[i]
        end
        # I think these are always the same
        lambdas_R = PeriodicArray(lambdas_R .^ -1)
        lambdas_L = PeriodicArray(lambdas_L .^ -1)

        #VLs = leftnull.(AL);
        #VRs = rightnull.(MPSKit._transpose_tail.(AR));

        inv_C = inv.(C)#.*inv.(CL);

        BL = AL_bar .* adjoint.(inv_C)
        BR = adjoint.(circshift(inv_C, 1)) .* MPSKit._transpose_tail.(AR_bar)

        BL_proj = adjoint.(AL) .* BL
        BR_proj = BR .* adjoint.(MPSKit._transpose_tail.(AR))

        BL_rest = BL - AL .* BL_proj
        BR_rest = BR - BR_proj .* MPSKit._transpose_tail.(AR)

        #BL = adjoint.(VLs).*AL_bar.*adjoint.(inv_C);
        #BR = adjoint.(circshift(inv_C,1)).*MPSKit._transpose_tail.(AR_bar).*adjoint.(VRs);

        # first two contributions to Ā
        Ā = map(zip(lambdas_L, circshift(CL, 1), BL_rest .* adjoint.(CR))) do (l, cl, b)
            @plansor temp[-1 -2; -3] := conj((l * cl)[1; -1]) * b[1 -2; -3]
        end

        Ā += map(zip(lambdas_R, CR, adjoint.(circshift(CL, 1)) .* BR_rest)) do (l, cr, b)
            @plansor temp[-1 -2; -3] := (l' * b)[-1; 2 -2] * conj(cr[-3; 2])
        end

        CL_bar = C_bar + circshift(map(zip(AL, AR, BL_proj, BL)) do (al, ar, b_proj, b)
                MPSKit._transpose_tail(b) * adjoint(MPSKit._transpose_tail(ar)) - TransferMatrix(al, ar) * b_proj
            end, -1)

        CR_bar = C_bar + map(zip(AL, AR, BR_proj, BR)) do (al, ar, b_proj, b)
            al' * MPSKit._transpose_front(b) - b_proj * TransferMatrix(ar, al)
        end

        # solve for left bit
        tosol = RecursiveVec(CL_bar)
        lvec = RecursiveVec(C)
        rvec = RecursiveVec(C)
        tosol -= rvec * dot(lvec, tosol) / length(lambdas_L)

        (sol, convhist) = linsolve(tosol, tosol, GMRES()) do x
            y = copy(x.vecs)
            for i in 1:length(y)
                y[i] -= TransferMatrix(AL[mod1(i + 1, end)], AR[mod1(i + 1, end)]) * x.vecs[mod1(i + 1, end)]
            end

            RecursiveVec(y)
        end
        convhist.converged == 0 && @warn "left failed to converge $(convhist.normres)"
        CL_bar = sol.vecs


        # solve for right bit
        tosol = RecursiveVec(CR_bar)
        lvec = RecursiveVec(C)
        rvec = RecursiveVec(C)

        tosol -= lvec * dot(rvec, tosol) / length(lambdas_L)
        (sol, convhist) = linsolve(tosol, tosol, GMRES()) do x
            y = copy(x.vecs)
            for i in 1:length(y)
                y[i] -= x.vecs[mod1(i - 1, end)] * TransferMatrix(AR[i], AL[i])
            end
            RecursiveVec(y)
        end
        convhist.converged == 0 && @warn "right failed to converge $(convhist.normres)"
        CR_bar = sol.vecs

        CL_bar = CL_bar .* adjoint.(CR)
        CR_bar = adjoint.(CL) .* CR_bar

        # and finally the contributions to Ā
        Ā += map(zip(lambdas_L, circshift(CL, 1), AL, CL_bar)) do (l, cl, al, b)
            @tensor temp[-1 -2; -3] := conj(l * cl[1; -1]) * al[1 -2; 2] * b[2; -3]
        end
        Ā += map(zip(lambdas_R, CR, AR, circshift(CR_bar, 1))) do (l, cr, ar, b)
            @tensor temp[-1 -2; -3] := l' * b[-1; 1] * ar[1 -2; 2] * conj(cr[-3; 2])
        end


        ∂self = NoTangent()


        return ∂self, Ā
    end
    return ts, infmps_pullback
end

# honestly don't know what backwards should return. 
function ChainRulesCore.rrule(::Type{PeriodicArray{T,N}},d::Array{T,N}) where {T,N}
    function backwards(v)
        (NoTangent(),v)
    end
    PeriodicArray(d),backwards
end