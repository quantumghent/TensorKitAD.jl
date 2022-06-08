Base.Experimental.@compiler_options optimize=0 compile=min infer=no
using Revise, TensorKit,TensorKitAD, Zygote, OptimKit

import LinearAlgebra;

struct peps_boundary{C,E}
    C1::C
    C2::C
    C3::C
    C4::C

    E1::E
    E2::E
    E3::E
    E4::E
end

init_peps(Dbond, Dphys) = TensorMap(randn, ComplexF64, Dbond * Dbond * Dbond' * Dbond' ← Dphys)


function peps_boundary(Dbond, Dbound)
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


function northwest_corner(E4,C1,E1,peps)
    @tensor corner[-1 -2 -3;-4 -5 -6] := E4[-1 1 2;3]*C1[3;4]*E1[4 5 6;-4]*peps[5 -5 -2 1;7]*conj(peps[6 -6 -3 2;7])
end



function left_move(boundary, peps; trscheme = truncdim(dim(χbound)))
    Q1 = northwest_corner(boundary.E3,boundary.C3,boundary.E4,permute(peps,(4,1,2,3),(5,)));
    Q2 = northwest_corner(boundary.E4,boundary.C4,boundary.E1,peps);

    (U,S,V) = tsvd(Q1*Q2,trunc=trscheme,alg=SVD());
    isqS = sdiag_inv_sqrt(S);

    Q = isqS*U'*Q1;
    P = Q2*V'*isqS;

    ## Insert a column of edge-peps-edge
    @tensor C1[-1;-2] := boundary.C1[1,2] * boundary.E1[2,3,4,-2]*Q[-1;1 3 4]
    @tensor C4[-1;-2] := boundary.C4[1,4] * boundary.E3[-1,2,3,1]*P[4 2 3;-2]
    @tensor E4[-1 -2 -3;-4] := boundary.E4[1 2 3;4]*peps[5 -2 7 2;9]*conj(peps[6 -3 8 3;9])*P[4 5 6;-4]*Q[-1;1 7 8]

    C1 = C1 / norm(C1)
    E4 = E4 / norm(E4)
    C4 = C4 / norm(C4)

    peps_boundary(C1,boundary.C2,boundary.C3,C4,boundary.E1,boundary.E2,boundary.E3,E4);
end

#=
function double_bit(E4,C1,E1,C2,E2,peps)
    @tensor derp[-1 -2 -3;-4 -5 -6] := E4[-1 1 2;3]*C1[3;4]*E1[4 5 6;7]*E1[7 8 9;10]*C2[10;11]*E2[11 12 13;-4]*
        peps[5 14 -2 1;16]*conj(peps[6 15 -3 2;16])*peps[8 12 -5 14;17]*conj(peps[9 13 -6 15;17])
end

function old_left_move(boundary, peps; trscheme = truncdim(dim(χbound)))
    above = double_bit(boundary.E4,boundary.C1,boundary.E1,boundary.C2,boundary.E2,peps);

    rpeps = permute(peps,(3,4,1,2),(5,));
    below = double_bit(boundary.E2,boundary.C3,boundary.E3,boundary.C4,boundary.E4,rpeps);

    @tensor blurp[-1 -2 -3;-4 -5 -6] := above[1 2 3;-1 -2 -3]*below[-4 -5 -6;1 2 3]
    (U,S,V) = tsvd(blurp,trunc = trscheme);
    isqS = sdiag_inv_sqrt(S)

    @tensor Pa[-1;-2 -3 -4] := isqS[4,-1]*conj(V[4;1 2 3])*below[1 2 3;-2 -3 -4]
    @tensor Pb[-1 -2 -3;-4] := isqS[-4,4]*conj(U[1 2 3;4])*above[-1 -2 -3;1 2 3]

    ## Insert a column of edge-peps-edge
    @tensor C1[-1;-2] := boundary.C1[1, 2] * boundary.E1[2,3,4,-2]*Pa[-1;1 3 4]
    @tensor C4[-1;-2] := boundary.C4[1,4] * boundary.E3[-1,2,3,1]*Pb[4 2 3;-2]
    @tensor E4[-1 -2 -3;-4] := boundary.E4[1 2 3;4]*peps[5 -2 7 2;9]*conj(peps[6 -3 8 3;9])*Pb[4 5 6;-4]*Pa[-1;1 7 8]

    C1 = C1 / norm(C1)
    E4 = E4 / norm(E4)
    C4 = C4 / norm(C4)

    peps_boundary(C1,boundary.C2,boundary.C3,C4,boundary.E1,boundary.E2,boundary.E3,E4);
end
=#
function sdiag_inv_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2)));
        end
    end
    toret
end
function TensorKitAD.ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap)
    toret = sdiag_inv_sqrt(S);
    toret,c̄ -> (TensorKitAD.ChainRulesCore.NoTangent(),-1/2*TensorKitAD._elementwise_mult(c̄,toret'^3))
end

function left_rotate(boundary, peps)
    peps = permute(peps, (2, 3, 4, 1), (5, ))
    boundary = peps_boundary(boundary.C2, boundary.C3, boundary.C4, boundary.C1, boundary.E2, boundary.E3, boundary.E4, boundary.E1)
    return boundary, peps
end

compute_norm(peps, boundary) =
    @tensor boundary.C1[1,2] * boundary.E1[2,4,7,3] * boundary.C2[3,9] * boundary.E2[9,10,11,17] * boundary.C3[17,16] * boundary.E3[16,14,15,13] * boundary.C4[13,12] * boundary.E4[12,5,6,1] * peps[4,10,14,5,8] * conj(peps[7,11,15,6,8])


function optimize_boundary(peps, boundary; numiter=10, kwargs...)
    spectra = [boundary.C1,boundary.C2,boundary.C3,boundary.C4]
    for i in 1:4*numiter
        boundary = left_move(boundary, peps)

        nspec = tsvd(boundary.C1,alg=SVD())[2];
        err = norm(spectra[mod1(i,4)]-nspec); # gebruiken zij meestal
        if err < 1e-12
            while mod1(i,4) != 1
                boundary, peps = left_rotate(boundary, peps)
                i += 1;
            end
            break
        end

        b = Zygote.bufferfrom(spectra);
        b[mod1(i,4)] = nspec;
        spectra = copy(b)

        boundary, peps = left_rotate(boundary, peps)
    end
    return peps, boundary
end

function contract_energy(peps, boundary, H)
    @tensor E_horizontal = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,11]) * H[11,20,12,18] * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,20]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    @tensor N_horizontal = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,12]) * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,18]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    E =  E_horizontal/N_horizontal

    boundary, peps = left_rotate(boundary, peps)
    @tensor E_horizontal = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,11]) * H[11,20,12,18] * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,20]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    @tensor N_horizontal = boundary.C1[1, 3] * boundary.E4[2,7,9,1] * boundary.C4[4,2] * boundary.E1[3,5,8,13] * boundary.E3[14,6,10,4] * peps[5,15,6,7,12] * conj(peps[8,19,10,9,12]) * boundary.E1[13,16,22,23] * boundary.E3[28,17,21,14] * peps[16,25,17,15,18] * conj(peps[22,26,21,19,18]) * boundary.C2[23,24] * boundary.E2[24,25,26,27] * boundary.C3[27,28]
    E +=  E_horizontal/N_horizontal

    return E
end

function ham_heis()

    Sx = zeros(ComplexF64, 2, 2)
    Sx[1,2] = 1
    Sx[2,1] = 1

    Sy = zeros(ComplexF64, 2, 2)
    Sy[1,2] = 1im
    Sy[2,1] = -1im

    Sz = zeros(ComplexF64, 2, 2)
    Sz[1,1] = 1
    Sz[2,2] = -1

    Dphys = ComplexSpace(2)
    σx = TensorMap(Sx, Dphys, Dphys)
    σy = TensorMap(Sy, Dphys, Dphys)
    σz = TensorMap(Sz, Dphys, Dphys)

    @tensor H[-1 -2 -3 -4] := -σx[-4,-2] * σx[-3,-1]-σy[-4,-2] * σy[-3,-1]+σz[-4,-2] * σz[-3,-1]
    return H
end

## Tests
const χbond = ℂ^2
const χphys = ℂ^2
const χbound = ℂ^20;


function cfun(x)
    (T,B) = x;
    H = ham_heis();

    function tfun(x)
        x,B = optimize_boundary(x,B,numiter=10);
        real(contract_energy(x,B,H))
    end

    E = tfun(T);
    grad = tfun'(T);

    @info E,norm(grad);
    flush(stderr)
    @assert !isnan(norm(grad))
    (E,grad)
end

function my_retract(x,η,α)
    (T,B) = x;
    #@info α;flush(stderr)
    T += α*η;
    T, B = optimize_boundary(T, B,numiter=100);
    (T,B),η
end

my_inner(x,η_1,η_2) = real(dot(η_1,η_2))

T = init_peps(χbond, χphys);
B = peps_boundary(χbond, χbound);
T, B = optimize_boundary(T, B,numiter=100);

let
    T = init_peps(χbond, χphys);
    B = peps_boundary(χbond, χbound);
    T, B = optimize_boundary(T, B,numiter=100);

    optimize(cfun, (T,B),ConjugateGradient(verbosity=2); inner=my_inner,retract=my_retract)
end
