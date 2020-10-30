#to run this file you'll need OMEinsum's master
using Zygote #the autodiff engine

using TensorKit, TensorKitAD
using OMEinsum # also exports an @tensor macro, but autodiffeable

a = TensorMap(rand,ComplexF64,ℂ^2*ℂ^5,ℂ^3);
ad = convert(Array,a);

function tfun1(x)
    @tensor res[-1,-2]:=x[-1,2,1]*conj(x[-2,2,1])
    norm(res)
end

function tfun2(x)
    @ein res[-1,-2]:=x[-1,2,1]*conj(x)[-2,2,1]
    norm(res)
end

d1 = tfun1'(a)
d2 = tfun2'(ad)

@show norm(convert(Array,d1)-d2)
