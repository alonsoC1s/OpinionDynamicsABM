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
             title="$(title) at step $(t)",)

    return p
end

function frame(oms::ModelSimulation, t; title="Simulation")
    colors = [:red, :green, :blue, :black]
    shapes = [:ltriangle, :rtriangle]
    X, _, Z, C, _ = oms

    c_idx = findfirst.(eachrow(C[:, :, t]))

    p = scatter(eachcol(X[:, :, t])...;
                c=colors[c_idx],
                # m=shapes[s_idx],
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
             title="$(title) at step $(t)")

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

function evolution(oms::ModelSimulation, filename; title="")
    anim = @animate for t in 1:length(oms)
        frame(oms, t; title=title)
    end

    return gif(anim, filename; fps=15)
end

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

# function plot_snapshot(X, Y, Z, B, C, filename; title="")
#     T = size(X, 3)
#     colwise_mins = mapslices(minimum, X; dims=1)
#     colwise_maxs = mapslices(maximum, X; dims=1)

#     # First, last and middle indices
#     start = firstindex(X, 3)
#     finish = lastindex(X, 3)
#     middle = round(Int, (finish - start) / 2)

#     frame_1st = plot_frame(X, Y, Z, B, C, start; mins=colwise_mins,
#                            maxs=colwise_maxs)
#     frame_mid = plot_frame(X, Y, Z, B, C, middle; mins=colwise_mins,
#                            maxs=colwise_maxs)
#     frame_end = plot_frame(X, Y, Z, B, C, finish; mins=colwise_mins,
#                            maxs=colwise_maxs)

#     l = @layout [a b c]
#     plot(frame_1st, frame_mid, frame_end; layout=l, size=(1000, 618), plot_title=title)

#     return savefig(filename)
# end

# function plot_lambda_radius(X, Y, Z, B, C, t)
#     rates = influencer_switch_rates(X[:, :, t], Z[:, :, t], B, C[:, :, t] |> BitMatrix,
#                                     15.0)

#     subplots = [plot() for _ in 1:size(Z, 1)]

#     for (i, p) in pairs(subplots)
#         scatter!(p, eachcol(X[:, :, t])...; zcolor=rates[:, i], title="Influencer $(i)")
#         scatter!(p, [Z[i, 1, t]], [Z[i, 2, t]]; c=:green, m=:x)
#     end

#     return plot(subplots...; layout=(2, 2), legend=false)
# end

# function plot_switch_propensity(X, Y, Z, B, C, t)
#     rates = influencer_switch_rates(X[:, :, t], Z[:, :, t], B, C[:, :, t] |> BitMatrix,
#                                     15.0)

#     propensity = sum(rates; dims=2)

#     return scatter(eachcol(X[:, :, t])...;
#                    zcolor=propensity,
#                    title="Agent by switch propensity",
#                    legend=:none,
#                    colorbar=true,)
# end