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

#you cannot really define the pullback wrt a function, because you don't have an inner product
function ChainRulesCore.rrule(::Type{<:TensorMap},f::Function,args...)
    function pullback(tm)
        ∂f = NoTangent()
        (NoTangent(),∂f,[NoTangent() for a in args]...)
    end

    TensorMap(f,args...),pullback
end

function ChainRulesCore.rrule(::Type{<:TensorMap},d::DenseArray,args...)
    function pullback(tm)
        ∂d = @thunk(convert(Array,tm))
        (NoTangent(),∂d,[NoTangent() for a in args]...)
    end
    TensorMap(d,args...),pullback
end

#pullback rule based on tom's krylovkit rule
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), t::AbstractTensorMap;kwargs...)
    T = eltype(t);

    (U,S,V) = tsvd(t;kwargs...);

    F = similar(S);
    for (k,dst) in blocks(F)

        src = blocks(S)[k]

        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            dst[i,j] = (i == j) ? zero(eltype(S)) : 1/(src[j,j]^2-src[i,i]^2+1e-7)
        end
    end


    function pullback(v)
        dU,dS,dV = v

        dA = zero(t);
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            dA += U*dS*V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            J = _elementwise_mult(F,(U'*dU))
            dA += U*(J+J')*S*V
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            K = _elementwise_mult(F ,V*dV')
            dA += U*S*(K+K')*V
        end
        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.ZeroTangent() && T <: Complex
            L = Diagonal(VpdV)
            dA += U*inv(S)*(L' - L)*V
        end
        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U,S,V), pullback
end


function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end
