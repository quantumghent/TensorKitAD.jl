using Zygote,TensorKit,TensorKitAD,MPSKit

let
    #create some random mps
    bonddims = [1,2,4,8,10,10,8,4,2,1]
    tensors = [TensorMap(rand,ComplexF64,ℂ^bonddims[i]*ℂ^2,ℂ^bonddims[i+1]) for i in 1:length(bonddims)-1];

    #the hamiltonian
    sz = TensorMap(ComplexF64[1 0;0 -1],ℂ^2,ℂ^2);
    sx = TensorMap(ComplexF64[0 1;1 0],ℂ^2,ℂ^2);
    @tensor ham[-1 -2;-3 -4]:=sz[-1 -2]*sz[-3 -4]+0.5*sx[-1 -2]*one(sx)[-3 -4]+0.5*one(sx)[-1 -2]*sx[-3 -4];

    function calculate_energy(state)

        l = isomorphism(Matrix{ComplexF64},space(state[1],1),space(state[1],1))
        r = isomorphism(Matrix{ComplexF64},space(state[end],3),space(state[end],3))

        @tensor lh[-1;-2]:=l[1,2]*state[1][2,3,-2]*conj(state[1][1,3,-1])
        lh*=0;

        for i in 2:length(state)
            @tensor lh[-1;-2] := lh[1,2]*state[i][2,3,-2]*conj(state[i][1,3,-1])

            #this works
            @tensor lh[-1;-2] := lh[-1;-2] + l[1,2]*state[i-1][2,3,4]*state[i][4,5,-2]*ham[6,3,7,5]*conj(state[i-1][1,6,8])*conj(state[i][8,7,-1])

            #this doesnt
            #@tensor lh[-1;-2]+= l[1,2]*state[i-1][2,3,4]*state[i][4,5,-2]*ham[6,3,7,5]*conj(state[i-1][1,6,8])*conj(state[i][8,7,-1])

            @tensor l[-1;-2] := l[1;2]*state[i-1][2,3,-2]*conj(state[i-1][1,3,-1])
        end

        en = @tensor lh[1,2]*r[1,2]
        norm = @tensor l[1,2]*state[end][2,3,4]*conj(state[end][1,3,4])

        real(en/norm)
    end

    for step in 1:1000
        tensors = tensors - 0.01*calculate_energy'(tensors)
        normalize!.(tensors)
        @show calculate_energy(tensors)
    end

end
