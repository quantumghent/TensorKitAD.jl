#to run this file you'll need OMEinsum's master
using Zygote #the autodiff engine

using TensorKit, TensorKitAD
using OMEinsum # also exports an @tensor macro, but autodiffeable
using KrylovKit
using LinearAlgebra


D = 4
K0 = Matrix{Float64}(I, D, D);
R0 = Matrix{Float64}(I, D, D);

function cMPSEnergy(k::Float64, μ::Float64, g::Float64, K::Array{}, R::Array{}, D::Int)
    Q = 1im*K - 1/2*(R1'*R1) - 1/2*(R2'*R2);
    U = Matrix{Float64}(I, D, D);

    lamda,r,info_eig = eigsolve(x -> Q*x + x*Q' + Re*x*Re' + Rp*x*Rp',Tr0,1,:LR,Arnoldi());
    R² = R*R;
    # Compute initial energy
    ekin = k*real(tr(QR*r*R'));
    epot = m*real(tr(R*r*R'));
    eint = g*real(tr(R²*r*R²))
    e = epot + ekin + eint;
    return e
end

@show cMPSEnergy'(1,1,10,K0,R0,D)
