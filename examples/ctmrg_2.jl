using TensorKit, TensorKitAD, Zygote


mutable struct peps_boundary{C,E}
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

function left_move!(boundary, peps; trscheme = truncdim(bound_dim(boundary)))
    ## Insert a column of edge-peps-edge
    @tensor C1[-1 -2 -3;-4] := boundary.C1[-1, 1] * boundary.E1[1,-2,-3,-4]
    @tensor E4[-1 -2 -3 -4 -5 -6 -7 -8] := boundary.E4[-1, 1, 2,-8] * peps[-6,-4,-2, 1, 3] * conj(peps[-7,-5,-3, 2, 3])
    @tensor C4[-1; -4 -3 -2] := boundary.C4[1,-4] * boundary.E3[-1,-2,-3,1]
    
    ## Compute isometry for disentangling
    @tensor ZZd[-1,-2,-3;-6,-5,-4] := C1[-1,-2,-3, 1] * conj(C1[-6,-4,-5, 1]) + conj(C4[1,-1,-3,-2]) * C4[1,-6,-5,-4]
    Z = tsvd(ZZd; trunc = trscheme)[1]
    
    ## Renormalize updated boundary tensors
    @tensor C1new[-1;-2] := conj(Z[1, 2, 3,-1]) * C1[1, 2, 3,-2]
    @tensor E4new[-1 -2 -3;-4] := conj(Z[1, 2, 3, -1]) * E4[1, 2, 3,-2,-3, 4, 5, 6] * Z[6, 4, 5, -4]
    @tensor C4new[-1;-2] := C4[-1,3,2,1] * Z[3, 1, 2,-2]
    
    boundary.C1 = C1new / norm(C1new)
    boundary.E4 = E4new / norm(E4new)
    boundary.C4 = C4new / norm(C4new)
end

function left_rotate(boundary, peps)
    peps = permute(peps, (2, 3, 4, 1), (5, ))
    boundary = peps_boundary(boundary.C2, boundary.C3, boundary.C4, boundary.C1, boundary.E2, boundary.E3, boundary.E4, boundary.E1)
    return boundary, peps
end

function bound_dim(boundary::peps_boundary)
    return dim(domain(boundary.C1))
end


function compute_boundaries(peps, boundary=init_boundaries(peps); numiter=100)
    for i in 1:(4 * numiter)
        left_move!(boundary, peps)
        boundary, peps = left_rotate(boundary, peps)
    end
    
    return boundary
end

function compute_norm(peps, boundary)
    return @tensor boundary.C1[1,2] * boundary.E1[2,4,7,3] * boundary.C2[3,9] * boundary.E2[9,10,11,17] * boundary.C3[17,16] * boundary.E3[16,14,15,13] * boundary.C4[13,12] * boundary.E4[12,5,6,1] * peps[4,10,14,5,8] * conj(peps[7,11,15,6,8])
end

function optimize_boundary(peps, boundary; numiter=100, kwargs...)
    for i in 1:4*numiter
        left_move!(boundary, peps)
        boundary, peps = left_rotate(boundary, peps)
    end
    return peps, boundary
end

function contract_energy(peps, boundary, H)
    N = compute_norm(peps, boundary)
    @tensor E_horizontal = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,11]) * H[11,20,12,18] * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,20]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    boundary, peps = left_rotate(boundary, peps)
    @tensor E_vertical = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,11]) * H[11,20,12,18] * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,20]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    return (E_horizontal + E_vertical) / N
end

function ham_ising(J, K)
    Sx = zeros(ComplexF64, 2, 2)
    Sx[1,2] = 1
    Sx[2,1] = 1
    
    Sz = zeros(ComplexF64, 2, 2)
    Sz[1,1] = 1
    Sz[2,2] = -1
    
    Dphys = ComplexSpace(2)
    σx = TensorMap(Sx, Dphys, Dphys)
    σz = TensorMap(Sz, Dphys, Dphys)
    
    @tensor H[-1 -2 -3 -4] := J * σx[-4,-2] * σx[-3,-1]
    return H
end


## Tests
χbond = 3
χphys = 2
χbound = 4

T = init_peps(χbond, χphys)
B = peps_boundary(χbond, χbound)

T, B = optimize_boundary(T, B)
N1 = compute_norm(T, B)
T, B = optimize_boundary(T, B)
N2 = compute_norm(T, B)


H = ham_ising(1,0)
contract_energy(T, B, H)