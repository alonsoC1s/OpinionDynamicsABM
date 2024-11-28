module OpinionDynamicsABM

# Write your package code here.
# Imports
using LinearAlgebra, Plots, Random, Distributions, Statistics, OMEinsum,
      DifferentialEquations, DiffEqCallbacks

# Exports...
export _boolean_combinator,
       _orthantize,
       _place_influencers,
       OpinionModelProblem,
       ModelParams,
       AgAg_attraction,
       InfAg_attraction,
       MedAg_attraction,
       simulate!,
       evolution,
       frame,
       evolve_compare,
       time_rate_tensor,
       influencer_switch_rates,
       _ag_ag_echo_chamber,
       _media_network,
       snapshots

include("utils.jl")
include("opinion_model_problem.jl")
include("sde_functions.jl")
include("solvers.jl")
include("plotting.jl")

end
