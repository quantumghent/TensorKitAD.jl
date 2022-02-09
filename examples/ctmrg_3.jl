using TensorKit, TensorKitAD, Zygote
import LinearAlgebra.diag, LinearAlgebra.diagm

struct peps_boundary{C, E}
    C1::C
    C2::C
    C3::C
    C4::C

    E1::E
    E2::E
    E3::E
    E4::E
end


function init_peps(χbond, χphys)
    Dbond = ℂ^χbond
    Dphys = ℂ^χphys
    
    return TensorMap(randn, ComplexF64, Dbond * Dbond * Dbond' * Dbond' ← Dphys)
end

function peps_boundary(χbond, χbound)
    Dbond = ℂ^χbond
    Dbound = ℂ^χbound
    # Create corner environment
    C1 = TensorMap(randn, ComplexF64, Dbound, Dbound)
    C2 = TensorMap(randn, ComplexF64, Dbound, Dbound)
    C3 = TensorMap(randn, ComplexF64, Dbound, Dbound)
    C4 = TensorMap(randn, ComplexF64, Dbound, Dbound)
    
    # Create edge environment
    E1 = TensorMap(randn, ComplexF64, Dbound * Dbond' * Dbond ← Dbound)
    E2 = TensorMap(randn, ComplexF64, Dbound * Dbond' * Dbond ← Dbound)
    E3 = TensorMap(randn, ComplexF64, Dbound * Dbond * Dbond' ← Dbound)
    E4 = TensorMap(randn, ComplexF64, Dbound * Dbond * Dbond' ← Dbound)
    
    # Normalize
    C1 = C1 / norm(C1)
    C2 = C2 / norm(C2)
    C3 = C3 / norm(C3)
    C1 = C4 / norm(C4)
    
    E1 = E1 / norm(E1)
    E2 = E2 / norm(E2)
    E3 = E3 / norm(E3)
    E4 = E4 / norm(E4)
    
    return peps_boundary(C1, C2, C3, C4, E1, E2, E3, E4)
end

function bound_dim(boundary::peps_boundary)
    return dim(domain(boundary.C1))
end

function contract_half(E_W, C_NW, E_N, C_NE, E_E, A)
    return @tensor U[-1 -2 -3 -4 -5 -6] := C_NW[1, 2] * E_N[2, 4, 7, 9] * E_N[9, 10, 11, 16] * C_NE[16, 17] * E_W[-1, 3, 5, 1] * A[4, 8, -2, 3, 6] * conj(A[7, 13, -3, 5, 6]) * A[10, 14, -4, 8, 12] * conj(A[11, 15, -5, 13, 12]) * E_E[17, 14, 15, -6]
end

function invsqrt(t::AbstractTensorMap)
    domain(t) == codomain(t) || error("invsqrt of a tensor only exists when domain == codomain.")
    I = sectortype(t)
    T = TensorKit.similarstoragetype(t, complex(float(eltype(t))))
    if I === Trivial
        data::T = diagm(1 ./ sqrt.(diag(block(t, Trivial()))))
        return TensorMap(data, codomain(t), domain(t))
    else
        datadict = SectorDict{I, T}(c => diagm(1 ./ sqrt.(diag(b))) for (c, b) in blocks(t))
        return TensorMap(datadict, codomain(t), domain(t))
    end
end

function left_move(boundary, peps; trscheme = truncdim(bound_dim(boundary)))
    upper = contract_half(boundary.E4, boundary.C1, boundary.E1, boundary.C2, boundary.E2, peps)
    lower = contract_half(boundary.E2, boundary.C3, boundary.E3, boundary.C4, boundary.E4, permute(peps, (3, 4, 1, 2), (5, )))
    
    R = last(leftorth(upper, (6, 4, 5), (1, 2, 3)))
    R′= last(leftorth(lower, (1, 2, 3), (6, 4, 5)))
    R2′ = last(leftorth(lower, (1, 3, 2), (6, 4, 5)))
    
    @tensor RR[-1; -2] := R[-1 1 2 3] * R′[-2 1 2 3]
    @tensor RR2[-1; -2] := R[-1 1 2 3] * R2′[-2 3 2 1]
    @show (norm(RR - RR2))
    U, S, V = tsvd(RR; trunc=trscheme)
    s = inv(sqrt(S))
    
    @tensor P′[-1 -2 -3 -4] := R′[2 -1 -2 -3] * conj(V[1 2]) * s[1 -4]
    @tensor P[-1 -2 -3 -4] := R[2 -1 -2 -3] * conj(U[2 1]) * s[-4 1]
    
    @tensor C1new[-1; -2] := P′[1 2 3 -1] * boundary.C1[1 4] * boundary.E1[4 2 3 -2]
    @tensor C4new[-1; -2] := P[1 2 3 -2] * boundary.C4[4 1] * boundary.E3[-1 2 3 4]
    @tensor E4new[-1 -2 -3; -4] := P′[1 2 3 -1] * boundary.E4[1 4 5 6] * peps[7 -2 2 4 8] * conj(peps[9 -3 3 5 8]) * P[6 7 9 -4]
    return peps_boundary(C1new / norm(C1new), boundary.C2, boundary.C3, C4new / norm(C4new), boundary.E1, boundary.E2, boundary.E3, E4new / norm(E4new))
end

function left_rotate(peps, boundary)
    return permute(peps, (2, 3, 4, 1), (5, )), peps_boundary(boundary.C2, boundary.C3, boundary.C4, boundary.C1, boundary.E2, boundary.E3, boundary.E4, boundary.E1)
end

function compute_boundaries(peps, boundary; numiter=100)
    for i in 1:(4 * numiter)
        boundary = left_move(boundary, peps)
        peps, boundary = left_rotate(peps, boundary)
    end
    return boundary
end

function compute_norm(peps, boundary)
    return @tensor boundary.C1[1,2] * boundary.E1[2,4,7,3] * boundary.C2[3,9] * boundary.E2[9,10,11,17] * boundary.C3[17,16] * boundary.E3[16,14,15,13] * boundary.C4[13,12] * boundary.E4[12,5,6,1] * peps[4,10,14,5,8] * conj(peps[7,11,15,6,8])
end

## Tests
χbond = 3;
χphys = 2;
χbound = 4;

A = init_peps(χbond, χphys);
B = peps_boundary(χbond, χbound);

B = left_move(B, A)
A, B = left_rotate(A, B)

B2 = compute_boundaries(A, B)
N = compute_norm(A, B2)
B3 = compute_boundaries(A, B2)
N2 = compute_norm(A, B3)

@show N
@show N2
