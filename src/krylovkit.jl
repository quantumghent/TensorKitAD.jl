#pullback rule based on https://arxiv.org/pdf/1909.02659v3.pdf
function ChainRulesCore.rrule(::typeof(KrylovKit.svdsolve), A::AbstractMatrix, howmany, which, T; kwargs...)
    res = svdsolve(A, howmany, which, T; kwargs...)
    S, U, V, _ = res
    U = hcat(U...)
    V = hcat(V...)

    Scopies2 = repeat(S, outer = [1,size(S, 1)]) .^ 2
    F = 1 ./ (transpose(Scopies2) .- Scopies2)
    F[LinearAlgebra.diagind(F)] .= 0

    function pullback(v)
        dS, dU, dV = v

        #dU and dV might be Unions of nothing and 1D arrays if cost does not depend on all singular vectors
        #in that case: convert nothing to zeros
        if eltype(dU) isa Union #Union{Nothing,Array{Float64,1}}
            dU = [dU[i] == nothing ? zeros(T,size(U,2)) : dU[i] for i in 1:size(U,1)]
        end
        dU = hcat(dU...)

        if eltype(dV) isa Union #Union{Nothing,Array{Float64,1}}
            dV = [dV[i] == nothing ? zeros(T,size(V,2)) : dV[i] for i in 1:size(V,1)]
        end
        dV = hcat(dV...)

        #placeholder for derivative
        dA = nothing

        #A_s bar term
        if dS != ChainRulesCore.Zero()
            As = U*LinearAlgebra.Diagonal(dS)*V'
            dA == nothing ? dA = As : dA += As
        end

        #A_uo bar term
        if dU != ChainRulesCore.Zero()
            J = F .* (U'*dU)
            Auo = U*(J+J')*LinearAlgebra.Diagonal(S)*V'
            dA == nothing ? dA = Auo : dA += Auo
        end

        #A_vo bar term
        if dV != ChainRulesCore.Zero()
            VpdV = V'*dV
            K = F .* VpdV
            Avo = U*LinearAlgebra.Diagonal(S)*(K+K')*V'
            dA == nothing ? dA = Avo : dA += Avo
        end

        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.Zero() && T <: Complex
            L = Diagonal(VpdV)
            Ad = U*LinearAlgebra.Diagonal(1 ./ S)*(L' - L)*V'
            dA == nothing ? dA = Ad : dA += Ad
        end

        return NO_FIELDS, dA, NoTangent(), NoTangent(), NoTangent(), [NoTangent() for kwa in kwargs]...
    end
    return res, pullback
end
