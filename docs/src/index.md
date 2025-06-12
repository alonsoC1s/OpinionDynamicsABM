```@meta
CurrentModule = OpinionDynamicsABM
```

# OpinionDynamicsABM.jl: Create and simulate Opinion Dynamics Agent-Based models

This package provides tools to create and simulate the "Opinion Dynamics under the impact
of Influencers and Media", as published in [the original
paper](https://doi.org/10.1038/s41598-023-46187-9).

This implementation focuses on high-performance. It implements two ways of simulating the
opinion model:
- With a hand-implemented Euler--Maruyama scheme, or
- Via [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/), a cutting-edge
  library for solving differential equations, doing parameter analysis, etc...

```@index
```

```@autodocs
Modules = [OpinionDynamicsABM]
```
