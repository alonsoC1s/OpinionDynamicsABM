# OpinionDynamicsABM

[![Docs](https://computationalhumanities.pages.zib.de/opiniondynamicsabm.jl/)](https://amartine.gitlab.io/OpinionDynamicsABM.jl/dev)
[![Build Status](https://git.zib.de/computationalhumanities/opiniondynamicsabm.jl/badges/main/pipeline.svg)]()
[![Coverage](https://git.zib.de/computationalhumanities/opiniondynamicsabm.jl/badges/main/coverage.svg)]()

Re-implementation of the code used to simulate the model published in ["Modelling opinion
dynamics under the impact of influencer and media
strategies"](https://doi.org/10.1038/s41598-023-46187-9).

The re-implementation aims to provide a performant API to modify parameters and run
simulations without the need to modify the core algorithm by hand.

## Quick start

Create a new Opinion model problem with determinisitic params.

```julia
using OpinionDynamicsABM
deterministic_params = ModelParams(; σ=0.0, σ̂=0, σ̃=0)
problem = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0), p = deterministic_params)

sol_em = simulate!(copy(problem))
sol = simulate!(copy(problem), (0.0, 2.0))

# Comparing the 2 solutions
evolve_compare(sol, sol_em, "evolution.gif")
```
