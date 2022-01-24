using KrylovKit, TensorKitAD, ChainRulesCore, LinearAlgebra, Test, ChainRulesTestUtils, FiniteDifferences, Zygote

const n = 2
const N = 100
const T = Float64

struct MyRuleConfig <: RuleConfig{Union{}} end

Base.isapprox(x, y::ZeroTangent; atol::Real = 0) = norm(x - y) <= atol
@testset "CG problems" begin
    @testset for T in (Float32, Float64)

        A = rand(T, (n, n))
        A = sqrt(A * A')
        b = rand(T, n)
        x₀ = rand(T, n)
        ϵ = 10n * eps(real(T)) * norm(b)
        alg = CG(maxiter = 2n, tol = 10n * eps(real(T)) * norm(b))
        a₀ = rand(real(T)) + 1
        a₁ = rand(real(T))

        x, info = linsolve(A, b, x₀, alg, a₀, a₁)
        x2, pullback2 = rrule(linsolve, A, b, x₀, alg, a₀, a₁)



        @test x ≈ first(x2)     # check if result is correct

        # compute pullback with finite FiniteDifferences
        f(A, b, x₀, a₀, a₁) = first(linsolve(A, b, x₀, alg, a₀, a₁))
        pullback(x̄) = j′vp(central_fdm(10, 1; factor = ϵ / eps(real(T))), (A, b, x₀, a₀, a₁) -> first(linsolve(A, b, x₀, alg, a₀, a₁)), x̄, A, b, x₀, a₀, a₁)

        x̄ = rand(T, n)
        ∂A, ∂b, ∂x₀, ∂a₀, ∂a₁ = pullback(x̄)
        _, ∂A2, ∂b2, ∂x₀2, _, ∂a₀2, ∂a₁2 = pullback2((x̄, NoTangent()))

        @test isapprox(∂A, ∂A2; atol = 1e-4)
        @test ∂b ≈ ∂b2
        @test isapprox(∂x₀, ∂x₀2; atol = √eps(T))
        @test ∂a₀ ≈ ∂a₀2
        @test ∂a₁ ≈ ∂a₁2
    end
end

@testset "Lanczos - eigsolve full" begin
    A = rand(T, (n, n)) .- one(T) / 2
    A = (A + A') / 2
    v = rand(T, (n,))
    ϵ = 10 * n * eps(real(T))
    alg = Lanczos(; krylovdim = 2 * n, maxiter = 1, tol = ϵ)

    D1, V1, info = eigsolve(A, v, 1, :LR, alg)
    y, pullback2 = rrule(eigsolve, A, v, 1, :LR, alg)
    D2, V2, info2 = y
    @test D1 ≈ D2
    @test V1 ≈ V2

    function f(A)
        F = eigen(A)
        return F.values, F.vectors
    end
    y, pullback = Zygote.pullback(f, A)
    
    function g(A)
        vals, vecs, _ = eigsolve(A, v, 1, :LR, alg)
        return vals, vecs
    end
    y2, pullback2 = Zygote.pullback(g, A)
    
    @show y
    @show y2
    x̄ = ([1.0 1.0], [[1.0 1.0; 1.0 1.0]])
    @show pullback(x̄)
    x̄2 = ([1.0 1.0], [[1.0 1.0], [1.0 1.0]])
    @show pullback2(x̄)
    
    # pullback(x̄) = j′vp(forward_fdm(10, 1; factor = ϵ / eps(real(T))), f, x̄, A, v)

    # @show x̄ = (([1.0]), [[0.0, 0.0]])
    # @show _, ∂A2, ∂x2, _, _, _ = pullback2((x̄..., NoTangent()))
    # @show (x̄[1][1], x̄[2][1])
    # @show ∂A, ∂x = pullback((x̄[1][1], x̄[2][1]))
    # @show ∂A, ∂A2
    
    # @show jacobian(central_fdm(10, 1; factor = ϵ / eps(real(T))), f, A, v)
end