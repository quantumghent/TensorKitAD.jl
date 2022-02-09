module TensorKitAD
    using TensorKit, TensorOperations, ChainRulesCore, TupleTools, KrylovKit, LinearAlgebra
    using ChainRules

    include("tensorkit.jl")
    include("tensoroperations.jl")
    include("krylovkit.jl")
    #using BackwardsLinalg # only place I know that contains qr backwards derivative
    # however, it's outdated, so I just stole the code
    include("backwardslinalg.jl")
end # module
