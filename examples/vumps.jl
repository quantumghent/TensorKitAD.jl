using Revise, TensorKit, TensorKitAD, Zygote, OptimKit, KrylovKit, MUntested
using LinearAlgebra: LinearAlgebra

function compute_energy(peps, H) end

mutable struct InfiniteMPS{E,F}
    AL::E
    AR::E
    AC::E
    C::F
end

function InfiniteMPS(A::AbstractTensorMap)
    C = isomorphism(Matrix{eltype(A)}, space(A, 1), space(A, 1))
    AL, C = leftcanonical(A, C)
    AR, C = rightcanonical(AL, C)
    AC = AL * C
    return InfiniteMPS(AL, AR, AC, C)
end

function leftgauge(A, C; tol=1.0e-14, maxiter=100)
    C = C / norm(C)    # this takes a copy so original C unharmed
    δ = 2 * tol
    iter = 1
    local AL

    while iter < maxiter && δ > tol
        Cold = C
        @tensor CA[-1 -2 -3 -4] := C[-1 1] * A[1 -2 -3 -4]
        AL, C = leftorth(CA, (1, 2, 3), (4,); alg=QRpos())
        C = C / norm(C)

        δ = norm(C - Cold)
        iter += 1
    end

    return AL, C
end

function rightgauge(A, C; tol=1.0e-14, maxiter=100)
    C = C / norm(C)  # this takes a copy so original C unharmed
    δ = 2 * tol
    iter = 1
    local AR

    while iter < maxiter && δ > tol
        Cold = C
        @tensor AC[-1 -2 -3 -4] := A[-1 -2 -3 1] * C[1 -4]
        C, AR = rightorth(AC, (1,), (2, 3, 4); alg=LQpos())
        C = C / norm(C)

        δ = norm(C - Cold)
        iter += 1
    end

    return AR, C
end

function initMPS(pSpace, vSpace)
    return TensorMap(randn, ComplexF64, vSpace * pSpace * pSpace' ← vSpace)
end

function gaugefix(A)
    C = isomorphism(Matrix{eltype(A)}, space(A, 1), space(A, 1))
    AL, C = leftgauge(A, C)
    AR, C = rightgauge(AL, C)
    @tensor AC[-1 -2 -3 -4] := AL[-1 -2 -3 1] * C[1 -4]
    return AL, AR, AC, C
end

function structure(A)
    @show codomain(A) ← domain(A)
end

function vumps(operator, A; tol=1.0e-12, maxiter=100)
    AL, AR, AC, C = gaugefix(A)
    

    
    FL = Tensor(randn, eltype(A), space(A, 1) * space(operator, 4) * space(operator, 4)' * space(A, 1)')
    FR = Tensor(randn, eltype(A), space(A, 4)' * space(operator, 2)' * space(operator, 2) * space(A, 4))

    local λ
    
    iter = 1
    δ = 2 * tol
    while iter < maxiter && δ > tol
        λₗ, FL = leftfixedpoint(operator, AL, FL)
        λᵣ, FR = rightfixedpoint(operator, AR, FR)
        λ = (λₗ + λᵣ) / 2
        FL = FL / @tensor FL[1 2 3 4] * FR[4 3 2 1]
        
        AC = computeAC(operator, FL, FR, AC)
        C = computeC(FL, FR, C)
        
        AC = computeACC(AC, C)
        AL, AR, AC, C = gaugefix(AC)
        
        δ = compute_error(AC, operator, FL, FR)
        @info "Iteration $(iter): δ = $(δ)"
        iter += 1
    end
    
    return AC, λ
end

function leftfixedpoint(operator, AL, FL; tol=1.0e-12)
    function applyTransferL(x)
        return @tensor FL[-1 -2 -3 -4] :=
            x[1 4 2 7] *
            AL[7 8 9; -4] *
            operator[8 -3 3 2 5] *
            conj(operator[9 -2 6 4 5]) *
            conj(AL[1 3 6; -1])
    end
    vals, vecs = eigsolve(applyTransferL, FL, 1, :LM, Arnoldi(tol=tol))
    return first(vals), first(vecs)
end

function fixedpoints(A, Aꜛ, l₀, r₀, which, algorithm)
    valsL, vecsL = eigsolve(Aꜛ, l₀, 1, which, algorithm)
    valsR, vecsR = eigsolve(A, r₀, 1, which, algorithm)
    
    l = first(vecsL)
    r = first(vecsR)
    λₗ = first(valsL)
    λᵣ = first(valsR)
    
    λ = (λₗ + λᵣ) / 2
    return λ, l, r
end

function ChainRulesCore.rrule(::typeof(fixedpoints), A, Aꜛ, l₀, r₀, which, algorithm)
    λ, l, r = fixedpoints(A, Aꜛ, l₀, r₀, which, algorithm)
    
    function eigsolve_pullback(Q̄)
    function fixedpoints_pullback(̄  )
        
    
    
    
    
end



    function A(x)
        return @tensor y[-1 -2 -3 -4] := x[1 4 2 7] * 
        
    
    
    
    
end

function rightfixedpoint(operator, AR, FR; tol=1.0e-12)
    function applyTransferR(x)
        return @tensor FR[-1 -2 -3 -4] :=
            x[1 3 4 7] *
            AR[-1 2 6 1] *
            operator[2 3 8 -2 5] *
            conj(operator[6 4 9 -3 5]) *
            conj(AR[-4 8 9 7])
    end
    vals, vecs = eigsolve(applyTransferR, FR, 1, :LM, Arnoldi(tol=tol))
    return first(vals), first(vecs)
end



function computeAC(operator, FL, FR, AC; tol=1.0e-12)
    function applyAC(x)
        return @tensor AC[-1 -2 -3 -4] := 
            x[7 4 2 1] *
            FL[-1 9 8 7] * 
            operator[4 6 -2 8 5] * 
            conj(operator[2 3 -3 9 5]) * 
            FR[1 6 3 -4]
    end
    vals, vecs = eigsolve(applyAC, AC, 1, :LM, Arnoldi(tol=tol))
    return first(vecs)
end

function computeC(FL, FR, C; tol=1.0e-12)
    function applyC(x)
        return @tensor C[-1; -2] :=
            x[1 2] *
            FL[-1 3 4 1] *
            FR[2 4 3 -2]
    end
    vals, vecs = eigsolve(applyC, C, 1, :LM, Arnoldi(tol=tol))
    return first(vecs)
end

function computeACC(AC, C)
    Q1, _ = leftorth(AC, (1, 2, 3), (4,), alg=QRpos())
    Q2, _ = leftorth(C, alg=QRpos())
    return Q1 * adjoint(Q2)
end

function compute_error(AC, operator, FL, FR)
    AC′ = computeAC(operator, FL, FR, AC)
    nullspace = leftnull(AC, (1, 2, 3), (4,))
    @tensor temp[-1; -2] := conj(nullspace[1 2 3 -1]) * AC′[1 2 3 -2]
    return norm(temp)
end

struct TransferPEPS{A, B}
    top::A
    bot::B
end
Base.size(T::AbstractTensorMap{S,M,N}, i) where {S,N,M} = dim(space(T, i))

let
    d = ℂ^2
    D = ℂ^3
    χ = ℂ^20
    
    
    O = TensorMap(randn, ComplexF64, D * D * D' * D' ← d);
    O = O / norm(O)
    A = TensorMap(randn, ComplexF64, χ * D' * D ← χ);
    FL = Tensor(randn, eltype(A), space(A, 1) * space(O, 4) * space(O, 4)' * space(A, 1)')
    FR = Tensor(randn, eltype(A), space(A, 4)' * space(O, 2)' * space(O, 2) * space(A, 4))
    # A, λ = vumps(O, A);
    AL, AR, AC, C = gaugefix(A)
    λₗ, FL = leftfixedpoint(O, AL, FL)
    λᵣ, FR = rightfixedpoint(O, AR, FR)
    
    y, back = Zygote._pullback(leftfixedpoint, O, AL, FL)
    back(y)
end
