function ChainRulesCore.rrule(::typeof(dot),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(b * c')
        ∂b = @thunk(a * c)
        return (NO_FIELDS, ∂a, ∂b)
    end
    return dot(a,b),pullback
end

function ChainRulesCore.rrule(::typeof(+),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NO_FIELDS, c, c)
    return a+b,pullback
end

function ChainRulesCore.rrule(::typeof(-),a::AbstractTensorMap,b::AbstractTensorMap)
    pullback(c) = (NO_FIELDS,c,-c)
    return a-b,pullback
end

function ChainRulesCore.rrule(::typeof(norm),a::AbstractTensorMap,p)
    p == 2 || throw(ArgumentError("DIY"))

    na = norm(a)
    function pullback(c)
        ∂a = @thunk(a*(c'+c)/(na*2))
        return (NO_FIELDS, ∂a)
    end
    return na,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(a'*c)
        return (NO_FIELDS, ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::AbstractTensorMap,b::Number)
     function pullback(c)
        ∂a = @thunk(c*b')
        ∂b = @thunk(dot(a,c))
        return (NO_FIELDS, ∂a, ∂b)
    end
    return a*b,pullback
end

function ChainRulesCore.rrule(::typeof(*),a::Number,b::AbstractTensorMap)
     function pullback(c)
        ∂a = @thunk(dot(b,c))
        ∂b = @thunk(a'*c)
        return (NO_FIELDS, ∂a, ∂b)
    end
    return a*b,pullback
end
