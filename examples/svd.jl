using Zygote, TensorKit, TensorKitAD, KrylovKit, LinearAlgebra

function cost(A)
    S, U, V = svdsolve(A)
    return real(dot(U[1], V[1]))
end

let
    A = randn(Complex{Float64}, (10,10))
    for i in 1:1000
        println(cost(A))
        A -=  0.1*cost'(A)
    end
end
