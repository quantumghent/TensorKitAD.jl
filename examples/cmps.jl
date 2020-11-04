using Zygote #the autodiff engine
using TensorKit, TensorKitAD
using KrylovKit, LinearAlgebra


D = 4
K0 = Matrix{Float64}(I, D, D);
R0 = Matrix{Float64}(I, D, D);

function cMPSEnergy(tup)
    (k,m,g,K,R,D) = tup
    Q = 1im*K - 1/2*(R'*R);
    U = Matrix{Float64}(I, D, D);
    Tr0 = Matrix{Float64}(I, D, D);

    lamda,r,info_eig = eigsolve(x -> Q*x + x*Q' + R*x*R',Tr0,1,:LR,Arnoldi());
    r = r[1]/tr(r[1]) ;# Cancels complex phase of r and ensures the normalization condition tr(l*r) = 1, where l is the identity matrix
    r = (r + r')/2 ;# Ensure r is exactly Hermitian, which is necesary for eig to produce a unitary matrix Vr
    Dr,Vr = eigen(r)
    order = sortperm(real(Dr),rev=true);
    Dr = Dr[order];
    Vr = Vr[:,order];
    r = diagm(0 => Dr);
    K = Vr'*K*Vr;
    R = Vr'*R*Vr;


    Q = 1im*K - 1/2*(R'*R);
    QR = Q*R - R*Q
    R² = R*R;
    # Compute initial energy
    ekin = k*real(tr(QR*r*R'));
    epot = m*real(tr(R*r*R'));
    eint = g*real(tr(R²*r*R²'))
    e = epot + ekin + eint;
    return e
end

@show cMPSEnergy((1.,1.,10.,K0,R0,D))
@show cMPSEnergy'((1.,1.,10.,K0,R0,D))
