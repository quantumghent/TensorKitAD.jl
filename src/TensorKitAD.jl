module TensorKitAD
    using TensorKit, TensorOperations, ChainRulesCore, TupleTools, KrylovKit

    include("linalg.jl")
    include("tensoroperations.jl")
    include("krylovkit.jl")

end # module
