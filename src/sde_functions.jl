
# TODO: This could perhaps be optimized even further. One of the possibly good ideas would
# be to allocate the forces vector once at the E-M solver level and pass a view to the
# attraction functions. This would reduce memory usage by making everything in-place.
function AgAg_attraction(X::AbstractVecOrMat{T}, A::BitMatrix; φ=x -> exp(-x)) where {T}
    I, D = size(X, 1), size(X, 2)
    J = size(X, 1) # Cheating. This only works for fully connected A

    # FIXME: These can be pre-allocated all the way up in the integrator. Perhaps even
    # reusing the same array over and over.
    # Pre allocating outputs
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
    agent_drift(X, M, I, A, B, C, a, b, c)

Calculates the drift force acting on agents, which is the weighted sum of the Agent-Agent,
Media-Agent and Influencer-Agent forces of attraction.
"""
function agent_drift(X::T, Y::T, Z::T, A::Bm, B::Bm, C::Bm,
                     a, b, c) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    return a * AgAg_attraction(X, A) +
           b * MedAg_attraction(X, Y, B) +
           c * InfAg_attraction(X, Z, C)
end

"""
    media_drift(X, Y, B, Γ; f = identity)

Calculates the drift force acting on media outlets, as described in eq. (4).
"""
function media_drift(X::T, Y::T, B::Bm, Γ;
                     f=identity) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    f_full(x) = ismissing(x) ? 0 : f(x)
    force = similar(Y)
    x_tilde = follower_average(X, B)
    force = f_full.(x_tilde .- Y)

    return (1 / Γ) .* force
end

"""
    influencer_drift(X, Z, C, γ, g = identity)

Calculates the drift force action on influencers as described in eq. (5).
"""
function influencer_drift(X::T, Z::T, C::Bm, γ;
                          g=identity) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    g_full(x) = ismissing(x) ? 0 : g(x)
    force = similar(Z)
    x_hat = follower_average(X, C)
    force = g_full.(x_hat .- Z)

    return (1 / γ) .* force
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
function influencer_switch_rates(X::T, Z::T, B::Bm, C::Bm, η::Float64; ψ=x -> exp(-x),
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
        R[j, :] = η .* D[j, :] .* r.(struct_similarity[m, :])
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
function switch_influencer(C::Bm, X::T, Z::T, rates::U,
                           dt) where {Bm<:BitMatrix,T,U<:AbstractVecOrMat}
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
