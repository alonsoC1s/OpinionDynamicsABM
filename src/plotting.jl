# FIXME: Parametrize on dimension of the problem. If dim != 2, none of these should work

function plot_frame(X, Y, Z, B, C, t; title="Simulation",
                    mins=mapslices(minimum, X; dims=1),
                    maxs=mapslices(maximum, X; dims=1))
    colors = [:red, :green, :blue, :black]
    shapes = [:ltriangle, :rtriangle]

    c_idx = findfirst.(eachrow(C[:, :, t]))
    # s_idx = findfirst.(eachrow(B))

    p = scatter(eachcol(X[:, :, t])...;
                c=colors[c_idx],
                # m=shapes[s_idx],
                legend=:none,
                xlims=(mins[1], maxs[1]),
                ylims=(mins[2], maxs[2]))

    scatter!(p,
             eachcol(Z[:, :, t])...;
             m=:hexagon,
             ms=6,
             markerstrokecolor=:white,
             markerstrokewidth=3,
             c=colors,
             title="Step $(t)",)

    return p
end

"""
    frame(oms::ModelSimulation, t; title = "Simulation")

Plots a single point in time of the simulation `oms` as a scatterplot showing agents and
influencers coded by color.
"""
function frame(oms::ModelSimulation, t; title="Simulation", B::AbstractMatrix = nothing)
    colors = [:red, :green, :blue, :black]
    shapes = [:ltriangle, :rtriangle]
                c=colors[c_idx],
                m=markers,
                legend=:none,
                xlims=oms.dom[1],
                ylims=oms.dom[2])

    scatter!(p,
             eachcol(Z[:, :, t])...;
             m=:hexagon,
             ms=6,
             markerstrokecolor=:white,
             markerstrokewidth=3,
             c=colors,
             title="Step $(t)")

    return p
end

# function plot_evolution(X, Y, Z, B, C, filename; title="")
#     T = size(X, 3)
#     colwise_mins = mapslices(minimum, X; dims=1)
#     colwise_maxs = mapslices(maximum, X; dims=1)

#     anim = @animate for t in 1:T
#         plot_frame(X, Y, Z, B, C, t; title=title, mins=colwise_mins, maxs=colwise_maxs)
#     end

#     return gif(anim, filename; fps=15)
# end

"""
    evolution(oms::ModelSimulation, filename; [title])

Plots the entire simulation `oms` as a gif where each frame corresponds to a timestep
taken by the integration algorithm.
"""
function evolution(oms::ModelSimulation, filename; title="")
    anim = @animate for t in 1:length(oms)
        frame(oms, t; title=title)
    end

    return gif(anim, filename; fps=15)
end

"""
    evolve_compare(s1::ModelSimulation{DiffEqSolver}, s2::ModelSimulation{BespokeSolver}, filename)

Plots the evolution of simulations `s1` and `s2` side by side on a gif. The simulations
are compared at the exact same timepoints, which are the timepoints where `BespokeSolver`
acted.
"""
function evolve_compare(s1::A, s2::B, filename;
                        title="") where {T,D,A<:ModelSimulation{T,D,DiffEqSolver},
                                         B<:ModelSimulation{T,D,BespokeSolver}}
    interpolate!(s1, s2.solver.tstops)
    l = @layout [a b]

    anim = @animate for t in 1:length(s1)
        p_sde = frame(s1, t; title="DiffEqs sim.")
        p_ori = frame(s2, t; title="Bespoke sim.")

        plot(p_sde, p_ori; layout=l, plot_title=title)
    end

    return gif(anim, filename; fps=15)
end

function snapshots(oms::ModelSimulation; title = "Simulation", B::AbstractMatrix = nothing)
    start = 1
    finish = oms.nsteps
    middle = round(Int, (finish - start) / 2)

    frame_1st = frame(oms, start; B)
    frame_mid = frame(oms, middle; B)
    frame_end = frame(oms, finish; B)

    l = @layout [a b c]

    return plot(frame_1st, frame_mid, frame_end; layout=l, plot_title=title)
end