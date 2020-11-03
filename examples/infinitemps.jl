#werkt nog niet


using Zygote, TensorKit, TensorKitAD, KrylovKit

using MPSKit

let
    # our state will be this tensor - repeated
    st = TensorMap(rand,ComplexF64,ℂ^10*ℂ^3,ℂ^10);

    # the heisenberg ham
    (sx,sy,sz) = nonsym_spintensors(1);
    @tensor ham[-1 -2; -3 -4]:= sx[-1 -3]*sx[-2 -4]+sy[-1 -3]*sy[-2 -4]+sz[-1 -3]*sz[-2 -4]

    function calc_energy(st)
        (_,l,r) = fixpoints(st);
        en = @tensor l[1,2]*st[2,3,4]*st[4,5,6]*r[6,7]*ham[8,9,3,5]*conj(st[1,8,10])*conj(st[10,9,7])
        no = @tensor l[1,2]*st[2,3,4]*st[4,5,6]*r[6,7]*conj(st[1,3,10])*conj(st[10,5,7])
        real(en/no)
    end

    @show calc_energy'(st)
end

function fixpoints(st)
    l = TensorMap(rand,ComplexF64,space(st,1),space(st,1));
    r = TensorMap(rand,ComplexF64,space(st,3)',space(st,3)');
    El(x) = @tensor y[-1;-2]:=x[1,2]*st[2,3,-2]*conj(st[1,3,-1])
    Er(x) = @tensor y[-1;-2]:=st[-1,2,1]*x[1,3]*conj(st[-2,2,3])
    fixpoints(El,Er,l,r)
end

function fixpoints(El,Er,l0,r0)
    (vals,vecs) = eigsolve(El,l0,1,:LM,Arnoldi())
    l = first(vecs);
    (vals,vecs) = eigsolve(Er,r0,1,:LM,Arnoldi())
    r = first(vecs);
    vals[1],l,r
end
