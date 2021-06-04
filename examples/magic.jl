#to run this file you'll need OMEinsum's master
using Zygote #the autodiff engine

using TensorKit, TensorKitAD
using OMEinsum # also exports an @tensor macro, but autodiffeable

a = TensorMap(rand,ComplexF64,ℂ^2*ℂ^5,ℂ^3);
ad = convert(Array,a);

function tfun1(x)
    res = real(@tensor x[1,2,3]*conj(x[1,2,3]))
end

function tfun2(x)
    res = real(sum(@ein t[] :=  x[1,2,3]*conj(x)[1,2,3]))
end

d1 = tfun1'(a)
d2 = tfun2'(ad)

@show norm(convert(Array,d1)-d2)
