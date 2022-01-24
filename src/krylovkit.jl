function ChainRulesCore.rrule(
    ::typeof(KrylovKit.linsolve),
    A::AbstractMatrix,
    b::AbstractVector,
    x₀,
    algorithm,
    a₀,
    a₁
)
    (x, info) = KrylovKit.linsolve(A, b, x₀, algorithm, a₀, a₁)

    
    function linsolve_pullback(x̄)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        (∂b, _) = KrylovKit.linsolve(
            A', x̄[1], (zero(a₀) * zero(a₁)) * x̄[1], algorithm, a₀, a₁
        )
        ∂a₀ = -dot(x, ∂b)
        ∂A = -a₁ * ∂b * x'
        ∂a₁ = -x' * A' * ∂b 
        return ∂self, ∂A, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
    return (x, info), linsolve_pullback
end



function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(KrylovKit.linsolve),
    f,
    b,
    x₀,
    algorithm,
    a₀,
    a₁
)
    x, info = KrylovKit.linsolve(f, b, x₀, algorithm, a₀, a₁)

    # f defines a linear map => pullback defines action of the adjoint
    # TODO this is probably not necessary if self-adjoint, see kwargs.
    (y, f_pullback) = rrule_via_ad(config, f, x)
    fᵀ(xᵀ) = f_pullback(xᵀ)[2]

    function linsolve_pullback(x̄)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        (∂b, _) = KrylovKit.linsolve(
            fᵀ, x̄[1], (zero(a₀) * zero(a₁)) * x̄[1], algorithm, a₀, a₁
        )
        ∂a₀ = -x' * ∂b
        ∂f = a₁ * ∂a₀
        ∂a₁ = -y' * ∂b
        return ∂self, ∂f, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
    return (x, info), linsolve_pullback
end

function ChainRulesCore.rrule(::typeof(KrylovKit.eigsolve), A::AbstractMatrix, x₀, howmany, which, algorithm)
    @show vals, vecs, info = eigsolve(A, x₀, howmany, which, algorithm)
    
    function eigsolve_pullback(Q̄)
        v̄als, v̄ecs, _ = Q̄
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂which = NoTangent()
        ∂algorithm = NoTangent()
        ∂howmany = NoTangent()
    
        ∂A = map(1:howmany) do i
            α = vals[i]
            x = vecs[i]
            ᾱ = v̄als[i]
            x̄ = v̄ecs[i]
            # @show norm(x)
            @show λ₀ = _eigsolve_λ₀(A, α, x, x̄)
            # @show x' * λ₀
            # @show (ᾱ * x - λ₀)
            # @show x
            return -λ₀ * x' + ᾱ * x * x'
        end
        return ∂self, collect(∂A), ∂x₀, ∂howmany, ∂which, ∂algorithm
    end
    return (vals, vecs, info), eigsolve_pullback
end
using LinearAlgebra
function LinearAlgebra.norm(x::ZeroTangent)
    return 0
end

function _eigsolve_λ₀(A::AbstractMatrix, α, x, x̄::ZeroTangent)
    return x̄
end

function _eigsolve_λ₀(A::AbstractMatrix, α, x, x̄)
    RL = (I - x * x') * x̄
    #RL = RL - dot(x, RL) * RL
    
    f(y) = A * y - α * y
    λ₀, _ = linsolve(f, RL)
    @show dot(x, λ₀)
    return λ₀ - dot(x, λ₀) * λ₀
end