#https://arxiv.org/pdf/1402.2859.pdf

using Revise,TensorKit,TensorKitAD,Zygote,Plots;
using LinearAlgebra:diag,Hermitian;

function peps_ising(β)
    T_buffer = Zygote.bufferfrom(zeros(ComplexF64,2,2,2,2));
    T_buffer[1,1,1,1] = 1;
    T_buffer[2,2,2,2] = 1;
    T = TensorMap(copy(T_buffer),ComplexSpace(2)*ComplexSpace(2), ComplexSpace(2)*ComplexSpace(2))


    Tz_buffer = Zygote.bufferfrom(zeros(ComplexF64,2,2,2,2));
    Tz_buffer[1,1,1,1] = -1;
    Tz_buffer[2,2,2,2] = 1;
    Tz = TensorMap(copy(Tz_buffer),ComplexSpace(2)*ComplexSpace(2), ComplexSpace(2)*ComplexSpace(2))

    dZ_buffer = Zygote.bufferfrom(zeros(ComplexF64,2,2));
    dZ_buffer[1,1] = exp(β)
    dZ_buffer[1,2] = exp(-β)
    dZ_buffer[2,1] = exp(-β)
    dZ_buffer[2,2] = exp(β)

    #the derivative of sqrt() is defined for hermitian matrices, so we not it as hermitian, and then convert it back to a normal array
    dZ = TensorMap(convert(Array,sqrt(Hermitian(copy(dZ_buffer)))), ComplexSpace(2), ComplexSpace(2))

    @tensor peps_tensor[-1 -2;-3 -4] := T[1,2,3,4]*dZ[-1,1]*dZ[-2,2]*dZ[3,-3]*dZ[4,-4]
    @tensor Z_tensor[-1 -2;-3 -4] := Tz[1,2,3,4]*dZ[-1,1]*dZ[-2,2]*dZ[3,-3]*dZ[4,-4]

    return peps_tensor, Z_tensor
end


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
function peps_boundary(D_bond, D_bound)
    a,b,c,d = TensorMap(randn, ComplexF64, D_bound, D_bound),TensorMap(randn, ComplexF64, D_bound, D_bound),TensorMap(randn, ComplexF64, D_bound, D_bound),TensorMap(randn, ComplexF64, D_bound, D_bound)
    e,f,g,h = TensorMap(randn, ComplexF64, D_bound*D_bond, D_bound),TensorMap(randn, ComplexF64, D_bound*D_bond, D_bound),TensorMap(randn, ComplexF64, D_bound*D_bond', D_bound),TensorMap(randn, ComplexF64, D_bound*D_bond', D_bound)
    return peps_boundary(a/sqrt(norm(a)), b/sqrt(norm(b)), c/sqrt(norm(c)), d/sqrt(norm(d)), e/sqrt(norm(e)), f/sqrt(norm(f)), g/sqrt(norm(g)), h/sqrt(norm(h)))
end


function renorm_left!(boundary, Tensor; trscheme = truncbelow(1e-5))
    @tensor C1_tilde[-1,-2;-3] := boundary.C1[-1,1]*boundary.E1[1,-2,-3]

    @tensor E4_tilde[-1,-2;-3,-4,-5] := boundary.E4[-1,1,-5]*Tensor[1,-2,-3,-4]
    @tensor C4_tilde[-1,-2,-3] := boundary.C4[1,-3]*boundary.E3[-1,-2,1]
    #construct isometry for this operation
    @tensor tmp[-1,-2;-3,-4] := C1_tilde[-1,-2,1]*conj(C1_tilde[-3,-4,1])  + conj(C4_tilde[1,-2,-1])*C4_tilde[1,-4,-3]

    isometry = tsvd(tmp; trunc=trscheme)[1]
    #contract neccecairy indices
    @tensor C1[-1;-2] := C1_tilde[1,2,-2]*conj(isometry[1,2,-1])
    @tensor E4[-1 -2;-3] := E4_tilde[1,2,-2,3,4]*isometry[4,3,-3]*conj(isometry[1,2,-1])
    @tensor C4[-1;-2] := C4_tilde[-1,1,2]*isometry[2,1,-2]

    boundary.C1 = C1/(norm(C1))
    boundary.E4 = E4/(norm(E4))
    boundary.C4 = C4/(norm(C4))
end
function rotate_peps(boundary, Tensor)
    @tensor new_tensor[-1 -2;-3 -4] := Tensor[-4,-1,-2,-3]
    return peps_boundary(boundary.C4, boundary.C1, boundary.C2, boundary.C3, boundary.E4, boundary.E1, boundary.E2, boundary.E3), new_tensor
end
function calc_norm(boundary, Tensor)
    return @tensor boundary.C1[1,2]*boundary.E1[2,5,3]*boundary.C2[3,4]*boundary.E2[4,10,12]*boundary.C3[12,11]*boundary.E3[11,9,8]*boundary.C4[8,6]*boundary.E4[6,7,1]*Tensor[7,9,10,5]
end

function optimize_boundary(Tensor, Tz,old_boundary=peps_boundary(ℂ^2,ℂ^3);numiter=100,kwargs...)

    for i in 1:numiter
        renorm_left!(old_boundary, Tensor; kwargs...)
        old_boundary, Tensor = rotate_peps(old_boundary, Tensor)

        renorm_left!(old_boundary, Tensor; kwargs...)
        old_boundary, Tensor = rotate_peps(old_boundary, Tensor)

        renorm_left!(old_boundary, Tensor; kwargs...)
        old_boundary, Tensor = rotate_peps(old_boundary, Tensor)

        renorm_left!(old_boundary, Tensor; kwargs...)
        old_boundary, Tensor = rotate_peps(old_boundary, Tensor)

    end

    return real(calc_norm(old_boundary, Tz)./calc_norm(old_boundary, Tensor)),old_boundary
end


βs = LinRange(0.0, 1, 20);
magns = fill(0.0,length(βs));
derivs = fill(0.0,length(βs));

for (i,β) in enumerate(βs)
    @show i;flush(stdout)
    #random boundary
    env = peps_boundary(ℂ^2,ℂ^50);

    (T,Tz) = peps_ising(β);

    #optimize thoroughly to get magnetization
    (magns[i],env) = optimize_boundary(T,Tz,env,numiter=500);

    #approximate gradient with very few steps
    #costfun(β) = first(optimize_boundary(peps_ising(β)...,env,numiter=10));
    #derivs[i] = costfun'(β)
end

plot(βs,abs.(magns),seriestype=:scatter)

plot(βs,abs.(derivs),seriestype=:scatter,ylim=(-0.1,1))
