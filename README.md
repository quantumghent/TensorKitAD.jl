# TensorKitAD.jl

Examples on how to derive backward rules : https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

How to define it in julia : https://github.com/JuliaDiff/ChainRules.jl

The autodiff engine I'm testing with : https://github.com/FluxML/Zygote.jl

## TensorOperations.jl

TensorOperations sometimes modifies memory in-place, which is forbidden for all autodiff engines. The way around this is to re-define the 'basic operations' to no longer work in-place. Things like

```
@tensor v[-1;-2] = ...
```

are automatically rewritten as
```
@tensor temp[-1;-2] := ...
v = temp
```

As long as your code does not explicitly depend on the memory location of v being unchanged, everything should work.

## TensorKit.jl

some basic operations are supported


## KrylovKit.jl

rules will move (or have moved) to KrylovKit itself
