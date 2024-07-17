
"""
    ModelParams

Wrapper for the Opinion Dynamics model parameters. The available parameters are:
- `L::Int`: number of influencers
- `M::Int`: number of media outlets
- `n::Int`: number of agents
- `η`: constant for the rate of influencer hopping for agents
- `a`: interaction strength between agents
- `b`: interaction strength between agents and media outlets
- `c`: interaction strength between agents and influencers
- `σ`: noise constant of the agent's stochastic movement
- `σ̂`: (\\sigma\\hat) noise constant for influencer movement
- `σ̃` (\\sigma\\tilde) noise constant for media movement
- `frictionI`: friction constant for influencers
- `frictionM`: friction constant for media outlets
"""
struct ModelParams{T<:Real} # FIXME: Parametrizing on T is unecessary
    L::Int
    M::Int
    n::Int
    η::T
    a::T
    b::T
    c::T
    σ::T
    σ̂::T
    σ̃::T
    frictionI::T
    frictionM::T
end

function ModelParams(L, M, n, η, a, b, c, σ, σ̂, σ̃, FI, FM)
    return ModelParams(L, M, n, promote(η, a, b, c, σ, σ̂, σ̃, FI, FM)...)
end

function ModelParams(;
                     L=4, M=2, n=250, η=15, a=1, b=2, c=4, σ=0.5, σ̂=0, σ̃=0, FI=10, FM=100)
    return ModelParams(L, M, n, η, a, b, c, σ, σ̂, σ̃, FI, FM)
end

"""
    OpinionModelProblem

encapsulates parameters and other properties of an Agent-Based model of Opinion Dynamics.
"""
struct OpinionModelProblem{T<:AbstractFloat}
    p::ModelParams{T} # Model parameters
    X::AbstractVecOrMat{T} # Array of Agents' positions
    M::AbstractVecOrMat{T} # Array of Media positions
    I::AbstractVecOrMat{T} # Array of Influencers' positions
    AgInfNet::BitMatrix # Adjacency matrix of Agents-Influencers
    AgAgNet::BitMatrix # Adjacency matrix of Agent-Agent interactions
    AgMedNet::BitMatrix # Agent-media correspondence vector
end

function Base.show(io::IO, omp::OpinionModelProblem{T}) where {T}
    return print("""
                 $(size(omp.X, 2))-dimensional Agent Based Opinion Model with:
                 - $(omp.p.n) agents
                 - $(omp.p.L) influencers
                 - $(omp.p.M) media outlets
                 """)
end

function OpinionModelProblem(dom::Vararg{Tuple{Real,Real},D}; p=ModelParams(),
                             seed=MersenneTwister(),
                             AgAgNetF::Function=I -> trues(p.n, p.n),) where {D<:Real}
    @info "Promoting elements of domain tuples"
    throw(ErrorException("Mixed-type tuples are not yet supported"))
    # return OpinionModelProblem(promote(dom...), seed, AgAgNetF)
end

function OpinionModelProblem(dom::Vararg{Tuple{T,T},D}; p=ModelParams(),
                             seed=MersenneTwister(),
                             AgAgNetF::Function=I -> trues(p.n, p.n),) where {D,T<:Real}

    # We divide the domain into orthants, and each orthant has 1 influencer
    p.L != 2^D && throw(ArgumentError("Number of influencers has to be 2^dim"))

    # Seeding the RNG
    Random.seed!(seed)

    # Place agents uniformly distributed across the domain
    X = reduce(hcat, [rand(Uniform(t...), p.n) for t in dom]) # p.n × N matrix

    # Create Agent-Influence network (n × L) by grouping individuals into quadrants
    # i,j-th entry is true if i-th agent follows the j-th influencer
    AgInfNet = _orthantize(X) |> BitMatrix

    # Placing the influencers as the barycenter of agents per orthant
    I = _place_influencers(X, AgInfNet)

    if D == 1
        X = vec(X)
        M = vec(M)
    end

    return OpinionModelProblem(X, I; p=p, AgAgNetF=AgAgNetF)
end

function OpinionModelProblem(agents_init::AbstractVecOrMat{T},
                             influencers_init::AbstractVecOrMat{T};
                             p=ModelParams(; L=size(influencers_init, 1),
                                           n=size(agents_init, 1)),
                             AgAgNetF::Function=I -> trues(p.n, p.n)) where {T<:AbstractFloat}
    p.L != size(influencers_init, 1) &&
        throw(ArgumentError("`influencers_init` defined more influencers than contemplated" *
                            "in the parameters $(p)"))

    p.n != size(agents_init, 1) &&
        throw(ArgumentError("`agents_init` defined more agents than contemplated in the" *
                            "parameters $(p)"))

    # Create Agent-Influence network (n × L) by grouping individuals into quadrants
    # i,j-th entry is true if i-th agent follows the j-th influencer
    AgInfNet = _orthantize(agents_init) |> BitMatrix

    # Defining the Agent-Agent interaction matrix as a function of the Agent-Influencer
    # matrix. In the default case, the matrix represents a fully connected network. In other
    # cases, the adjacency is computed with the adjacency to influencers.
    AgAgNet = AgAgNetF(AgInfNet)

    # Assign agents to media outlet randomly s.t. every agent is connected to 1 and only 1 media.
    AgMedNet = _media_network(p.n, p.M)

    # We consider just 2 media outlets at the "corners"
    D = size(agents_init, 2)
    M = vcat(fill(-one(T), (1, D)), fill(one(T), (1, D)))

    return OpinionModelProblem(p, agents_init, M, influencers_init, AgInfNet, AgAgNet,
                               AgMedNet)
end

function get_values(omp::OpinionModelProblem)
    return omp.X, omp.M, omp.I, omp.AgAgNet, omp.AgMedNet, omp.AgInfNet
end

# This could perhaps be optimized even further. One of the possibly good ideas would be to
# allocate the forces vector once at the E-M solver level and pass a view to the
# attraction functions. This would reduce memory usage by making everything in-place.
function AgAg_attraction(X::AbstractVecOrMat{T}, A::BitMatrix; φ=x -> exp(-x)) where {T}
    # Pre allocating outputs
    I, D = size(X, 1), size(X, 2)
    J = size(X, 1) # Cheating. This only works for fully connected A

    force = similar(X)
    Dijd = similar(X, I, J, D)
    Wij = similar(X, I, J)

    for i in axes(force, 1) # Iterate over agents
        agent = view(X, i, :)
        neighbors = findall(@view A[i, :])
        normalization_constant = zero(T)

        if isempty(neighbors)
            view(force, i, :) .= zeros(T, 1, D)
            continue
        end

        # Distance between agent and neighbor
        for j in neighbors # |neighbors| <= |J| so no index-out-of-bounds
            neighbor = view(X, j, :)
            Dij = view(Dijd, i, j, :)
            Dij .= neighbor .- agent

            # Filling Wij in the same loop
            w = φ(norm(Dij))
            view(Wij, i, j) .= w
            normalization_constant += w
        end

        # row-normalize W to get 1/sum(W[i, j] for j)
        view(Wij, i, :) .= view(Wij, i, :) ./ normalization_constant
    end

    # Calculate the attraction force per dimension with Einstein sum notation
    force .= ein"ijd,ij -> id"(Dijd, Wij)
    return force
end

"""
    AgAg_attraction(omp::OpinionModelProblem, φ = x -> exp(-x))

Calculate the force of attraction on agents exerted by other agents they are connected to,
with φ the scaling function.
"""
function AgAg_attraction(omp::OpinionModelProblem{T}; φ=x -> exp(-x)) where {T}
    X, A = omp.X, omp.AgAgNet
    return AgAg_attraction(X, A)
end

function MedAg_attraction(X::T, M::T, B::BitMatrix) where {T<:AbstractVecOrMat}
    force = similar(X)
    # FIXME: Can be written even more compactly

    # Detect early if an agent is not connected to any Media Outlets
    if !(all(any(B; dims=2)))
        throw(ErrorException("Model violation detected: An agent is disconnected from " *
                             "all media outlets."))
    end

    for i in axes(X, 1)
        media_idx = findfirst(B[i, :])
        force[i, :] = M[media_idx, :] - X[i, :]
    end

    return force
end

"""
    MedAg_attraction(omp::OpinionModelProblem)

Calculates the Media-Agent attraction force for all agents.
"""
function MedAg_attraction(omp::OpinionModelProblem)
    return MedAg_attraction(omp.X, omp.M, omp.AgMedNet)
end

function InfAg_attraction(X::T, Z::T, C::BitMatrix) where {T<:AbstractVecOrMat}
    force = similar(X)

    # Detect early if an agent doesn't follow any influencers
    if !(all(any(C; dims=2)))
        throw(ErrorException("Model violation detected: An Agent doesn't follow any  " *
                             "influencers"))
    end

    for i in axes(X, 1)
        # force[i, :] = sum(C[i, m] * (Z[m, :] - X[i, :]) for m = axes(C, 2)) # ./ count(C[i, :])
        influencer_idx = findfirst(C[i, :])
        force[i, :] = Z[influencer_idx, :] - X[i, :]
    end

    return force
end

"""
    InfAg_attraction(omp::OpinionModelProblem)

Calculates the Influencer-Agent attraction force for all agents.
"""
function InfAg_attraction(omp::OpinionModelProblem)
    X, Z, C = omp.X, omp.I, omp.AgInfNet
    return InfAg_attraction(X, Z, C)
end

"""
    follower_average(X, Network::BitMatrix)

Calculates the center of mass of the agents connected to the same media or influencer as
determined by the adjacency matrix `Network`.
"""
function follower_average(X::AbstractVecOrMat, Network::BitMatrix)
    mass_centers = zeros(size(Network, 2), size(X, 2))

    # Detect early if one outlet/influencer has lost all followers i.e a some column is empty
    lonely_outlets = Int[]
    if !(all(any(Network; dims=1)))
        # Exclude this index of the calculations and set zeros manually to the results
        v = (collect ∘ vec)(any(Network; dims=1)) # Hack to force v into a Vector{bool}
        append!(lonely_outlets, findall(!, v))
    end

    # Calculate centers of mass, excluding the outlets left alone to avoid div by zero
    for m in setdiff(axes(Network, 2), lonely_outlets)
        # Get the index of all the followers of m-th medium
        ms_followers = findall(Network[:, m])
        # Store the col-wise average for the subset of X that contains the followers
        mass_centers[m, :] = mean(X[ms_followers, :]; dims=1)
    end

    # Set center of mass as missing for lonely actors so drift can deal with it differently
    # FIXME: Eliminar y solamente crear array de missings en vez de esto
    if length(lonely_outlets) > 0
        mass_centers[lonely_outlets] = fill(missing, size(X, 2))
    end

    return mass_centers
end

"""
    agent_drift(X, M, I, A, B, C, p)

Calculates the drift force acting on agents, which is the weighted sum of the Agent-Agent,
Media-Agent and Influencer-Agent forces of attraction.
"""
function agent_drift(X::T, Y::T, Z::T, A::Bm, B::Bm, C::Bm,
                     p::ModelParams) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    a, b, c = p.a, p.b, p.c
    return a * AgAg_attraction(X, A) +
           b * MedAg_attraction(X, Y, B) +
           c * InfAg_attraction(X, Z, C)
end

"""
    media_drift(X, Y, B; f = identity)

Calculates the drift force acting on media outlets, as described in eq. (4).
"""
function media_drift(X::T, Y::T, B::Bm;
                     f=identity) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    f_full(x) = ismissing(x) ? 0 : f(x)
    force = similar(Y)
    x_tilde = follower_average(X, B)
    force = f_full.(x_tilde .- Y)

    return force
end

"""
    influencer_drift(X, Z, C, g = identity)

Calculates the drift force action on influencers as described in eq. (5).
"""
function influencer_drift(X::T, Z::T, C::Bm;
                          g=identity) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    g_full(x) = ismissing(x) ? 0 : g(x)
    force = similar(Z)
    x_hat = follower_average(X, C)
    force = g_full.(x_hat .- Z)

    return force
end

"""
    followership_ratings(B, C)

Calculates the rate of individuals that follow both the `m`-th media outlet and the `l`-th
influencer. The `m,l`-th entry of the output corresponds to the proportion of agents that
follows `m` and `l`.

# Arguments:
- `B::BitMatrix`: Adjacency matrix of the agents to the media outlets
- `C::BitMatrix`: Adjacency matrix of the agents to the influencers
"""
function followership_ratings(B::BitMatrix, C::BitMatrix)
    n, M = size(B)
    L = size(C, 2)

    R = zeros(M, L)
    for m in 1:M
        audience_m = findall(B[:, m])
        R[m, :] = count(C[audience_m, :]; dims=1) ./ n
    end

    return R
end

"""
    influencer_switch_rates(X, Z, B, C, η; ψ = x -> exp(-x), r = relu)

Returns an n × L matrix where the `j,l`-th entry contains the rate λ of the Poisson point
process modeling how agent `j` switches influencers to `l`. Note that this is not the
same as ``Λ_{m}^{→l}``.
"""
function influencer_switch_rates(X::T, Z::T, B::Bm, C::Bm, η; ψ=x -> exp(-x),
                                 r=relu) where {T<:AbstractVecOrMat,Bm<:BitMatrix}

    # Compute the followership rate for media and influencers
    rate_m_l = followership_ratings(B, C)
    # attractiveness is the total proportion of followers per influencer
    # Divide each col of rates over sum(n_(m, l) for m = 1:M)
    struct_similarity = rate_m_l ./ sum(rate_m_l; dims=1)

    # Computing distances of each individual to the influencers
    D = zeros(size(X, 1), size(Z, 1))
    for (i, agent_i) in pairs(eachrow(X))
        for (l, influencer_l) in pairs(eachrow(Z))
            D[i, l] = ψ(norm(agent_i - influencer_l))
        end
    end

    # Calculating switching rate based on eq. (6)
    R = zeros(size(X, 1), size(Z, 1))
    for (j, agentj_media) in pairs(eachrow(B))
        m = findfirst(agentj_media)
        R[j, :] = η * D[j, :] .* r.(struct_similarity[m, :])
    end

    return R
end

"""
    switch_influencer(C, X, Z, B, η, dt)

Simulates the Poisson point process that determines how agents change influencers based on
the calculated switching rates. The keyword argument `method` determines how the process
is simulated. If `method` == :other, the process is simulated with the rates calculated
via [`influencer_switch_rates`], and if `method` == :luzie, the process is simulated with
the legacy approach that was used in the paper preprint.

See also [`influencer_switch_rates`]
"""
function switch_influencer(C::Bm, X::T, Z::T, rates::T,
                           dt) where {Bm<:BitMatrix,T<:AbstractVecOrMat}
    L, n = size(Z, 1), size(X, 1)

    # rates = influencer_switch_rates(X, Z, B, C, η)
    RC = copy(C)

    ## Trying it Luzie's way
    for j in 1:n
        r = rand()
        lambda = sum(rates[j, :])
        if r < 1 - exp(-lambda * dt)
            p = rates[j, :] / lambda
            r2 = rand()
            k = 1
            while sum(p[1:k]) < r2
                k += 1
            end

            RC[j, :] = zeros(L)
            RC[j, k] = 1
        end
    end

    return RC
end

"""
    simulate!(omp::OpinionModelProblem; Nt=100, dt=0.01, method=:other)

Simulates the evolution of the Opinion Dynamics problem `omp` by solving the associated
SDE via Euler--Maruyama with `Nt` time steps and resolution `dt`.

The kwarg `method` is used to determine the influencer switching method. See
[`influencer_switch_rates`] for more information.
"""
function simulate!(omp::OpinionModelProblem{T};
                   Nt=200,
                   dt=0.01,
                   seed=MersenneTwister(),
                   echo_chamber::Bool=false,) where {T}
    X, Y, Z, A, B, C = get_values(omp)
    σ, n, Γ, γ, = omp.p.σ, omp.p.n, omp.p.frictionM, omp.p.frictionI
    M, L = omp.p.M, omp.p.L
    d = size(X, 2)
    σ̂, σ̃ = omp.p.σ̂, omp.p.σ̃
    η = omp.p.η

    # Seeding the RNG
    Random.seed!(seed)

    # Allocating solutions & setting initial conditions
    rX = zeros(T, n, d, Nt)
    rY = zeros(T, M, d, Nt)
    rZ = zeros(T, L, d, Nt)
    rC = BitArray{3}(undef, n, L, Nt)
    # Jump rates can be left uninitialzied. Not defined for the last time step
    # rR = Array{T,3}(undef, n, L, Nt-1)
    rR = Array{T,3}(undef, n, L, Nt - 1)

    rX[:, :, begin] = X
    rY[:, :, begin] = Y
    rZ[:, :, begin] = Z
    rC[:, :, begin] = C

    # Solve with Euler-Maruyama
    for i in 1:(Nt - 1)
        X = view(rX, :, :, i)
        Y = view(rY, :, :, i)
        Z = view(rZ, :, :, i)
        C = view(rC, :, :, i) |> BitMatrix

        # FIXME: Try using the dotted operators to fuse vectorized operations
        # Agents movement
        FA = agent_drift(X, Y, Z, A, B, C, omp.p)
        rX[:, :, i + 1] .= X + dt * FA + σ * sqrt(dt) * randn(n, d)

        # Media movements
        FM = media_drift(X, Y, B)
        rY[:, :, i + 1] .= Y + (dt / Γ) * FM + (σ̃ / Γ) * sqrt(dt) * randn(M, d)

        # Influencer movements
        FI = influencer_drift(X, Z, C)
        rZ[:, :, i + 1] .= Z + (dt / γ) * FI + (σ̂ / γ) * sqrt(dt) * randn(L, d)

        # Change influencers
        rates = influencer_switch_rates(X, Z, B, C, η)
        rR[:, :, i] .= rates
        R = view(rR, :, :, i)
        view(rC, :, :, i + 1) .= switch_influencer(C, X, Z, R, dt)

        if echo_chamber
            # Modify Agent-Agent interaction network
            A .= _ag_ag_echo_chamber(BitMatrix(rC[:, :, i + 1]))
        end
    end

    return rX, rY, rZ, rC, rR
end

function drift(du, u, p, t)
    # Defining the indices for readability
    # FIXME: Broken offsets. Do it properly
    agents = CartesianIndices((firstindex(u):p.n, axes(u, 2)))
    influencers = CartesianIndices((p.n+1:p.n+p.L, axes(u,2)))
    media = CartesianIndices((p.n+p.L+1:size(u, 1), axes(u, 2)))

    # Assigning variable names to vector of solutions for readability
    X = @view u[agents]
    Y = @view u[media]
    Z = @view u[influencers]

    # Agents SDE
    du[agents] .= agent_drift(X, Y, Z, p.A, p.B, p.C, p.p)

    # Influencer SDE
    du[influencers] .= influencer_drift(X, Z, p.C)

    # Media drift
    du[media] .= media_drift(X, Y, p.B)
    
end

function noise(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):p.n, axes(du, 2)))
    influencers = CartesianIndices((p.n+1:p.L, axes(du,2)))
    media = CartesianIndices((p.L+1:p.M, axes(du, 2)))
    
    # Additive noise
    du[agents] .= p.σ
    du[influencers] .= p.σ̂
    du[media] .= p.σ̃
end


# FIXME: Parametrize on dimension of the problem. If dim != 2, none of these should work


function plot_frame(X, Y, Z, B, C, t; title="Simulation",
                    mins=mapslices(minimum, X; dims=1),
                    maxs=mapslices(maximum, X; dims=1))
    colors = [:red, :green, :blue, :black]
    shapes = [:ltriangle, :rtriangle]

    c_idx = findfirst.(eachrow(C[:, :, t]))
    s_idx = findfirst.(eachrow(B))

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

function plot_evolution(X, Y, Z, B, C, filename; title="",)
    T = size(X, 3)
    colwise_mins = mapslices(minimum, X; dims=1)
    colwise_maxs = mapslices(maximum, X; dims=1)

    anim = @animate for t in 1:T
        plot_frame(X, Y, Z, B, C, t; title=title, mins=colwise_mins, maxs=colwise_maxs)
    end

    return gif(anim, filename; fps=15)
end

function plot_snapshot(X, Y, Z, B, C, filename; title="")
    T = size(X, 3)
    colwise_mins = mapslices(minimum, X; dims=1)
    colwise_maxs = mapslices(maximum, X; dims=1)

    # First, last and middle indices
    start = firstindex(X, 3)
    finish = lastindex(X, 3)
    middle = round(Int, (finish - start) / 2)

    frame_1st = plot_frame(X, Y, Z, B, C, start; mins=colwise_mins,
                           maxs=colwise_maxs)
    frame_mid = plot_frame(X, Y, Z, B, C, middle; mins=colwise_mins,
                           maxs=colwise_maxs)
    frame_end = plot_frame(X, Y, Z, B, C, finish; mins=colwise_mins,
                           maxs=colwise_maxs)

    l = @layout [a b c]
    plot(frame_1st, frame_mid, frame_end; layout=l, size=(1000, 618), plot_title=title)

    return savefig(filename)
end

function plot_lambda_radius(X, Y, Z, B, C, t)
    rates = influencer_switch_rates(X[:, :, t], Z[:, :, t], B, C[:, :, t] |> BitMatrix,
                                    15.0)

    subplots = [plot() for _ in 1:size(Z, 1)]

    for (i, p) in pairs(subplots)
        scatter!(p, eachcol(X[:, :, t])...; zcolor=rates[:, i], title="Influencer $(i)")
        scatter!(p, [Z[i, 1, t]], [Z[i, 2, t]]; c=:green, m=:x)
    end

    return plot(subplots...; layout=(2, 2), legend=false)
end

function plot_switch_propensity(X, Y, Z, B, C, t)
    rates = influencer_switch_rates(X[:, :, t], Z[:, :, t], B, C[:, :, t] |> BitMatrix,
                                    15.0)

    propensity = sum(rates; dims=2)

    return scatter(eachcol(X[:, :, t])...;
                   zcolor=propensity,
                   title="Agent by switch propensity",
                   legend=:none,
                   colorbar=true,)
end
