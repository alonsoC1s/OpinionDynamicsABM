"""
    frame(oms::ModelSimulation, t; title = "Simulation")

Plots a single point in time of the simulation `oms` as a scatterplot showing agents and
influencers coded by color.
"""
function frame(oms::ModelSimulation, t; title="Simulation $(t)",
               B::Union{AbstractMatrix,Nothing}=nothing,
               colors::AbstractVector=[:red, :green, :blue, :yellow])
    shapes = [:ltriangle, :rtriangle]
    X, _, Z, C, _ = oms

    c_idx = findfirst.(eachrow(C[:, :, t]))

    markers, markersize, infl_marker = if ~isnothing(B)
        s_idx = findfirst.(eachrow(B))
        (shapes[s_idx], 5, :circle)
    else
        (:none, 4, :star5)
    end

    # Plot agents
    p = scatter(eachcol(X[:, :, t])...;
                c=colors[c_idx],
                m=markers,
                ms=markersize,
                legend=:none,
                xlims=oms.dom[1],
                ylims=oms.dom[2],)

    # Plot influencers
    scatter!(p,
             eachcol(Z[:, :, t])...;
             m=infl_marker,
             ms=6,
             markerstrokecolor=:white,
             markerstrokewidth=3,
             c=colors,
             title=title,
             aspect_ratio=:equal)

    return p
end

"""
    evolution(oms::ModelSimulation, filename; [title])

Plots the entire simulation `oms` as a gif where each frame corresponds to a timestep
taken by the integration algorithm.
"""
function evolution(oms::ModelSimulation, filename; frame_title="Step ")
    anim = @animate for t in 1:length(oms)
        frame(oms, t; title=frame_title)
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

function snapshots(oms::ModelSimulation; title="Simulation",
                   B::Union{AbstractMatrix,Nothing}=nothing)
    start = 1
    finish = oms.nsteps
    middle = round(Int, (finish - start) / 2)

    frame_1st = frame(oms, start; B)
    frame_mid = frame(oms, middle; B)
    frame_end = frame(oms, finish; B)

    return plot(frame_1st, frame_mid, frame_end; layout=(1, 3), plot_title=title)
end
