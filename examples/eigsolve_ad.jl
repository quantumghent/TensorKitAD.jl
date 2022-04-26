using LinearAlgebra, KrylovKit, ChainRulesCore, Test

function bieigsolve(A::AbstractMatrix, x₀; which=:LM)
    λᵣ, vᵣ, infoᵣ = eigsolve(A, x₀, 1, which, Arnoldi())
    λₗ, vₗ, infoₗ = eigsolve(A', x₀, 1, which, Arnoldi())

    infoₗ.converged > 0 && infoᵣ.converged > 0 || @warn "Eigensolvers not converged."

    λᵣ = first(λᵣ)
    λₗ = first(λₗ)
    conj(λₗ) ≈ λᵣ || @warn "Eigenvalues disagree: $λᵣ, $λₗ"

    vᵣ = first(vᵣ)
    vₗ = first(vₗ)

    return λᵣ, vₗ, vᵣ
end

function ChainRulesCore.rrule(
    ::typeof(bieigsolve),
    A::AbstractMatrix,
    x₀;
    kwargs...
)
    λ, vₗ, vᵣ = bieigsolve(A, x₀; kwargs...)

    function bieigsolve_pullback(X̄)
        λ̄, l̄, r̄ = X̄
        f = dot(vₗ,vᵣ);

        # account for normalization
        r̄ -= dot(vᵣ, r̄) * vᵣ
        l̄ -= dot(vₗ, l̄) * vₗ

        # (A - λI)ξₗ = (1 - vᵣvₗ')l̄
        # vₗ' ξₗ = 0
        if isa(l̄, typeof(ZeroTangent()))
            ξₗ = l̄
        else
            ξₗ, infoₗ = linsolve(A, l̄ - vᵣ * vₗ' * l̄/f, -λ)
            infoₗ.converged == 0 && @warn "Reverse left cotangent problem did not converge."
            ξₗ -=  vᵣ*dot(vₗ, ξₗ)/f
        end

        # (A - λI)'ξᵣ = (1 - vₗvᵣ')r̄
        # vᵣ' ξᵣ = 0
        if isa(r̄, typeof(ZeroTangent()))
            ξᵣ = r̄
        else
            ξᵣ, infoᵣ = linsolve(A', r̄ - vₗ * vᵣ' * r̄/f', -conj(λ))
            infoᵣ.converged == 0 && @warn "Reverse right cotangent problem did not converge."
            ξᵣ -=  vₗ*dot(vᵣ, ξᵣ)/f'
        end

        ∂A = λ̄ * vₗ * vᵣ'/f' - vₗ * ξₗ' - ξᵣ * vᵣ'
        return NoTangent(), ∂A, ZeroTangent()
    end
    return (λ, vₗ, vᵣ), bieigsolve_pullback
end

## Test with simple matrix, left and right dominant eigenvector the same.



A = [8.0 1.0 6.0; 3.0 5.0 7.0; 4.0 9.0 2.0]
x = rand(eltype(A), size(A, 2))

λ, l, r = bieigsolve(A, x)

# test defining relations of bieigsolve:
@test λ ≈ 15
# @test dot(l, r) ≈ 1.0
@test A * r ≈ λ * r
@test A' * l ≈ conj(λ) * l

function f(A)
    x₀ = rand(eltype(A), size(A, 2))
    λ, l, r = bieigsolve(A, x₀)
    return λ, l, r
end

function g(A)
    λᵣ, vᵣ = eigen(A, sortby=x->-abs(x))
    λₗ, vₗ = eigen(copy(A'), sortby=x->-abs(x))
    r = vᵣ[:,1]
    l = vₗ[:,1]
    return λᵣ[1], l, r
end

@test f(A)[1] ≈ g(A)[1]
for i = 2:3
    @test abs(dot(f(A)[i], g(A)[i])) ≈ 1
end

using Zygote
t = rand(eltype(A), size(A,2))
function costf(A)
    λ, l, r = f(A)
    a = abs(dot(t, l))
    b = abs(dot(t, r))
    return real(λ + a + b)
end
function costg(A)
    λ, l, r = g(A)
    a = abs(dot(t, l))
    b = abs(dot(t, r))
    return λ + a + b
end

@test costf(A) ≈ costg(A)
Zygote.refresh()
fₐ, ∂f = Zygote._pullback(costf, A)
gₐ, ∂g = Zygote._pullback(costg, A)

ȳ = 1.0
_, a, = ∂f(ȳ)
_, b, = ∂g(ȳ)
@test fₐ ≈ gₐ
@test ∂f(ȳ)[2] ≈ ∂g(ȳ)[2]

for i in 1:100
    j1 = jacobian(costf, A)[1];
    j2 = jacobian(costg, A)[1];
    @show norm(j1),norm(j2),j1./j2
    @show @test j1 ≈ j2
end

## Some harder test, left and right eigenvector different
T = ComplexF64
n = 2
A = rand(T, (n,n))

x = rand(eltype(A), size(A, 2))
λ, l, r = f(A)
a, b, c = g(A)

# test defining relations of bieigsolve:
# @test dot(l, r) ≈ 1.0
@test A * r ≈ λ * r
@test A' * l ≈ conj(λ) * l

# @test dot(b, c) ≈ 1
t = rand(eltype(A), size(A,2))

f1(A) = real(first(f(A)))
g1(A) = real(first(g(A)))
f2(A) = real(abs(dot(t, f(A)[2])))
g2(A) = real(abs(dot(t, g(A)[2])))
f3(A) = real(abs(dot(t, f(A)[3])))
g3(A) = real(abs(dot(t, g(A)[3])))

@test f1(A) ≈ g1(A)
@test f2(A) ≈ g2(A)
@test f3(A) ≈ g3(A)

@test jacobian(f1, A)[1] ≈ jacobian(g1, A)[1]
@test jacobian(f2, A)[1] ≈ jacobian(g2, A)[1]
@test jacobian(f3, A)[1] ≈ jacobian(g3, A)[1]
