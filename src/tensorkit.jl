function ChainRulesCore.rrule(::typeof(dot),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(b * c')
        ∂b = @thunk(a * c)
        return (NoTangent(), ∂a, ∂b)
    end
    return dot(a,b),pullback
end

function ChainRulesCore.rrule(::typeof(+),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NoTangent(), c, c)
    return a+b,pullback
end

function ChainRulesCore.rrule(::typeof(-),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NoTangent(),c,-c)
    return a-b,pullback
end

function ChainRulesCore.rrule(::typeof(norm),a::AbstractTensorMap,p)
    p == 2 || throw(ArgumentError("DIY"))

    na = norm(a)
    function pullback(c)
        ∂a = @thunk(a*(c'+c)/(na*2))
        return (NoTangent(), ∂a)
    end
    return na,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(a'*c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::Number)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(dot(a,c))
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::Number,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(dot(b,c))
        ∂b = @thunk(a'*c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(isomorphism),args...)
    isomorphism(args...),x->(NoTangent(),[NoTangent() for a in args]...)
end

#we assume
function ChainRulesCore.rrule(::Type{<:TensorMap},f::Function,args...)
    function pullback(tm)
        if f in (ones,rand,randn,zeros,LinearAlgebra.I)
            ∂f = NoTangent()
        else
            throw(ArgumentError("derivative wrt to $(f) not implemented"))
        end

        (NoTangent(),∂f,[NoTangent() for a in args]...)
    end

    TensorMap(f,args...),pullback
end
