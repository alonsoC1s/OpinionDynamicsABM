module OpinionDynamicsABM

# Write your package code here.
# Imports
using LinearAlgebra, Plots, Random, Distributions, Statistics, OMEinsum

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
       plot_evolution,
       plot_frame,
       time_rate_tensor,
       influencer_switch_rates,
       _ag_ag_echo_chamber,
       _media_network,
       plot_snapshot,
       drift,
       noise

include("utils.jl")
include("abm.jl")

theme(:ggplot2)

end
