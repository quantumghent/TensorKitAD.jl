using Revise,TensorKitAD,TensorKit,MPSKit,Zygote

sz = TensorMap(ComplexF64[1 0;0 -1],ℂ^2,ℂ^2);
sx = TensorMap(ComplexF64[0 1;1 0],ℂ^2,ℂ^2);
@tensor ham[-1 -2;-3 -4]:=sz[-1 -3]*sz[-2 -4]+0.5*sx[-1 -3]*one(sx)[-2 -4]+0.5*one(sx)[-1 -3]*sx[-2 -4];

t1 = TensorMap(rand,ComplexF64,ℂ^6*ℂ^2,ℂ^5);
t2 = TensorMap(rand,ComplexF64,ℂ^5*ℂ^2,ℂ^7);
t3 = TensorMap(rand,ComplexF64,ℂ^7*ℂ^2,ℂ^6);

function tfun(x)
    ts = InfiniteMPS([x...]);;

    en = 0.0+0im;
    for pos in 1:3
        en += @tensor ts.AC[pos][1 2;3]*ts.AR[pos+1][3 4;5]*conj(ts.AC[pos][1 6;7])*conj(ts.AR[pos+1][7 8;5])*ham[6 8;2 4]
    end

    real(en)
end


using OptimKit;
cfun(x) = (tfun(x),tfun'(x))
my_retract(x,η,α) = (x .+ η.*α,η)
my_inner(x,η_1,η_2) = real(dot(η_1,η_2))
my_scale(v,α) = v.*α;
my_add(v,η,β) = v.+(η.*β);

((t1,t2,t3),_) = optimize(cfun, (t1,t2,t3),ConjugateGradient(verbosity=2,maxiter=500); inner = my_inner,retract = my_retract,scale! = my_scale, add! = my_add);

# check if the result agrees with MPSKit
ts = InfiniteMPS([t1,t2,t3]);
th = MPOHamiltonian(ham);
(ts,_) = find_groundstate(ts,th,VUMPS());
@show sum(expectation_value(ts,th))
