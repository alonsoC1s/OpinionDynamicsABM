using Revise
using OpinionDynamicsABM
using Distributions
using Plots

theme(:ggplot2)
default(size = (335, 335))

# 3 Influencers, b = 0 eliminates media
n = 300
no_media_params = ModelParams(; L = 3, M = 2, b = 0, n = n)

function segment_agent(x::AbstractArray)
    if x[1] <= 0 && x[2] <= 0
        return [1 0 0]
    elseif x[1] >= 0 && x[2] <= 0
        return [0 1 0]
    elseif x[2] >= 0
        return [0 0 1]
    end
end


# Building model
dom = ((-2.0, 2.0), (-2.0, 2.0))
X = reduce(hcat, [rand(Uniform(t...), n) for t in dom]) # p.n × N matrix
# Hand-placed influencers
Z = [
    -1.0 -1.0;
    1.0 -1.0;
    0.0 1.0
    ]

# Assigning influencers by distance
# kdtree = KDTree(permutedims(X, (2, 1)))
C = reduce(vcat, map(segment_agent, eachrow(X))) |> BitMatrix


p = OpinionModelProblem{Float64, 2}(X, Z, C; p = no_media_params, dom = dom)
sol = simulate!(p)

evolution(sol, "tmp/img/christof.gif")
