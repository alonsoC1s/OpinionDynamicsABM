import Base.Math: exp_fast

# TODO: This could perhaps be optimized even further. One of the possibly good ideas would
# be to allocate the forces vector once at the E-M solver level and pass a view to the
# attraction functions. This would reduce memory usage by making everything in-place.
@doc raw"""
    AgAg_attraction(X, A; φ = x -> exp(-x))

Computes the force collectively exerted on agents by other agents they are connected to.
Returns the results as a matrix where the ``i,j-th`` entry represents the commulative
force felt by agent ``i`` along dimension ``j``.

# Mathematical definition
Mathematically, the function computes,
```math
\frac{1}{\sum_{j^\prime} w_{ij^{\prime}}} \sum_{j=1}^{N} w_{ij} \, (x_j - x_i),
```
where the interaction weights are:
```math
w_{ij} = A_{ij} \, \varphi(\| x_j - x_t \|).
```
The resulting row vectors are stored as rows of the matrix returned.
"""
function AgAg_attraction(X::AbstractArray{T}, A::AdjMatrix{T}; φ=x -> exp(-x)) where {T}
    I, D = size(X)
    J = size(X, 1) # Cheating. This only works for fully connected A

    # Pre allocating outputs. I can get away with not initializing force, not so with D, W.
    force = similar(X)
    Dijd = zeros(I, J, D)
    Wij = zeros(I, J)

    AgAg_attraction!(force, Dijd, Wij, X, A; φ=φ)

    return force
end

# function AgAg_attraction!(force, Dijd, Wij, X::AbstractArray{T},
#                           A::SubArray{Bool,D,BitArray{3},I,L};
#                           φ=x -> exp(-x)) where {T,D,I,L}
function AgAg_attraction!(force, Dijd, Wij, X::AbstractArray{T}, A;
                          φ=x -> exp_fast(-x)) where {T}
    # Resetting buffers. Force can be left as-is, every entry is guaranteed to be overwritten.
    fill!(Dijd, zero(T))
    fill!(Wij, zero(T))

    @inbounds for i in axes(force, 1) # Iterate over agents
        agent = view(X, i, :)
        neighbors = findall(view(A, i, :)) # FIXME: `neighbors` is being allcoated every loop
        normalization_constant = zero(T)

        if isempty(neighbors)
            fill!(view(force, i, :), zero(T))
            continue
        end

        # Distance between agent and neighbor
        @inbounds for j in neighbors # |neighbors| <= |J| so no index-out-of-bounds
            norm_accumulator = zero(T)
            neighbor = view(X, j, :)
            @inbounds for d in axes(Dijd, 3)
                dist_d = neighbor[d] - agent[d]
                view(Dijd, i, j, d) .= dist_d
                norm_accumulator += dist_d^2
            end

            # Filling Wij in the same loop
            w = φ(sqrt(norm_accumulator))
            view(Wij, i, j) .= w
            normalization_constant += w # FIXME: I haven't accounted for this in the theory
        end

        # row-normalize W to get 1/sum(W[i, j] for j)
        view(Wij, i, :) .= view(Wij, i, :) ./ normalization_constant
    end

    # Calculate the attraction force per dimension with Einstein sum notation
    # force .= ein"ijd,ij -> id"(Dijd, Wij)
    @tullio force[i, d] = Dijd[i, j, d] * Wij[i, j]
    return nothing
end

# Version specialized on abstract array version of the adjacency matrix
function AgAg_attraction!(force, Dijd, Wij, X::AbstractArray{T}, A::SparseOrViewMatrix{T};
                          φ=x -> exp_fast(-x)) where {T}
    # Resetting buffers. fill! is calls an optimized version for sparse  arrays like these
    fill!(force, zero(T)) # Resetting to 0 is crucial
    fill!(Dijd, zero(T))
    fill!(Wij, zero(T))

    # Look at the connection weight of all agents connected to agent_idx
    rows = rowvals(A)
    vals = nonzeros(A)
    w_i = zeros(T, size(Wij, 1))

    @inbounds for j in axes(force, 1) # Iterate over agents
        # Entries of Fid of disconnected agents are 0 by default
        agent = view(X, j, :)
        neighbors = nzrange(A, j)

        if length(neighbors) == 0
            # If (col) j had no neighbors, the j-th row of Wij will be zeros
            view(w_i, j) .= one(T)
        end

        # Exploit CSC sparse structure to efficiently explore the network
        @inbounds for ii in neighbors
            i = rows[ii]
            norm_accumulator = zero(T)
            neighbor = view(X, i, :)
            @inbounds @simd for d in axes(Dijd, 3)
                dist_d = agent[d] - neighbor[d]
                view(Dijd, i, j, d) .= dist_d
                norm_accumulator += dist_d^2
            end

            # Filling Wij in the same loop
            w = φ(sqrt(norm_accumulator))
            view(Wij, i, j) .= w
            view(w_i, i) .= w_i[i] + w # FIXME: I haven't accounted for this in the theory
        end
    end
    # row-normalize W to get 1/sum(W[i, j] for j)
    view(Wij, :, :) .= view(Wij, :, :) ./ w_i
    # Calculate the attraction force per dimension with Einstein sum notation
    # force .= ein"ijd,ij -> id"(Dijd, Wij)
    @tullio force[i, d] = Dijd[i, j, d] * Wij[i, j]
    return nothing
end

function AgAg_attraction(omp::OpinionModelProblem{T}; φ=x -> exp(-x)) where {T}
    return AgAg_attraction(omp.X, omp.A; φ=φ)
end

@doc raw"""
    MedAg_attraction(X, M, B)

Computes the force exerted on an agent by the media outlet it follows as determined by
``B``. Output has the same shape as in [`AgAg_attraction`](@ref).

# Mathematical definition
Mathematically, the function computes,
```math
\sum_{m=1}^{M} B_{im} \, (y_m - x_i)
```
"""
function MedAg_attraction(X, M, B)
    force = similar(X)
    MedAg_attraction!(force, X, M, B)

    return force
end

function MedAg_attraction(omp::OpinionModelProblem)
    return MedAg_attraction(omp.X, omp.Y, omp.B)
end

function MedAg_attraction!(Force, X, M, B)
    for i in axes(X, 1)
        media_idx = findfirst(B[i, :])
        view(Force, i, :) .= view(M, media_idx, :) .- view(X, i, :)
    end
end

@doc raw"""
    InfAg_attraction(X, Z, C)

Computes the force exerted on each agent by the influencer they follow, determined by
``C``. Output has the same shape as in [`AgAg_attraction`](@ref).

# Mathematical definition
Mathematically, the function computes,
```math
\sum_{\ell=1}^{L} C_{i\ell} \, (z_m - x_i)
```
"""
function InfAg_attraction(X, Z, C)
    force = similar(X)
    InfAg_attraction!(force, X, Z, C)

    return force
end

function InfAg_attraction(omp::OpinionModelProblem)
    X, Z, C = omp.X, omp.Z, omp.C
    return InfAg_attraction(X, Z, C)
end

function InfAg_attraction!(Force, X, Z, C)
    for i in axes(X, 1)
        influencer_idx = findfirst(C[i, :])
        view(Force, i, :) .= view(Z, influencer_idx, :) .- view(X, i, :)
    end
end

"""
    follower_average(X, Network::BitMatrix)

Calculates the center of mass of the agents connected to the same media or influencer as
determined by the adjacency matrix `Network`.
"""
function follower_average(X::AbstractArray, Network)
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

@doc raw"""
    agent_drift(X, M, I, A, B, C, a, b, c)

Calculates the drift force acting on agents, which is the weighted sum of the Agent-Agent,
Media-Agent and Influencer-Agent forces of attraction.

# Mathematical definition
The function corresponds to the drift function ``F_i`` of the SDE
```math
dx_i(t) = F_i (x, y, z, t) \, dt + \sigma \, dW_i (t).
```
"""
function agent_drift(X, Y, Z, A, B, C, a, b, c)
    return a * AgAg_attraction(X, A) +
           b * MedAg_attraction(X, Y, B) +
           c * InfAg_attraction(X, Z, C)
end

function agent_drift!(Force, Ftmp, Dijd, Wij, X, Y, Z, A, B, C, a, b, c)
    # Aggregate results into `Force` reusing `Ftmp` as buffer
    AgAg_attraction!(Ftmp, Dijd, Wij, X, A)
    Force .= a .* Ftmp

    MedAg_attraction!(Ftmp, X, Y, B)
    Force .+= b .* Ftmp

    InfAg_attraction!(Ftmp, X, Z, C)
    return Force .+= c .* Ftmp

    # Force .= a * Force + b * MedAg_attraction(X, Y, B) + c * InfAg_attraction(X, Z, C)
end

@doc raw"""
    media_drift(X, Y, B, Γ)

Calculates the drift force acting on media outlets. In other words, the drift function of
the following SDE:
```math
\Gamma dy_m(t) = (\widetilde{x_m}(t) - y_m (t)) \, dt + \widetilde{\sigma} \, d
\widetilde{W}_m (t),
```
where ``\widetilde{x_m}`` is the average opinion of the media outlet ``m`` followers.
"""
function media_drift(X::T, Y::T, B, Γ; f=identity) where {T<:AbstractArray}
    f_full(x) = ismissing(x) ? 0 : f(x) # FIXME: Use missings.jl
    force = similar(Y)
    x_tilde = follower_average(X, B)
    force = f_full.(x_tilde .- Y)

    return (1 / Γ) .* force
end

@doc raw"""
    influencer_drift(X, Z, C, γ)

Calculates the drift force action on influencers, i.e. the drift function of the SDE:
```math
\gamma dz_{\ell} (t) = (\widehat{x_{\ell}} (t) - z_{\ell}(t)) \, dt + \widehat{\sigma} d
\widetilde{W_\ell},
```
where ``\widehat{x_{\ell}}`` is the average opinion of the ``\ell`` influencer's
followers.
"""
function influencer_drift(X::T, Z::T, C, γ;
                          g=identity) where {T<:AbstractArray}
    g_full(x) = ismissing(x) ? 0 : g(x) # FIXME: Use missings.jl
    force = similar(Z)
    x_hat = follower_average(X, C)
    force = g_full.(x_hat .- Z)

    return (1 / γ) .* force
end

@doc raw"""
    followership_ratings(B, C)

Calculates the rate of individuals that follow both the ``m``-th media outlet and the
``\ell``-th influencer. The `m,l`-th entry of the output corresponds to the proportion of
agents that follows ``m`` and ``l``.

# Arguments:
- `B::BitMatrix`: Adjacency matrix of the agents to the media outlets
- `C::BitMatrix`: Adjacency matrix of the agents to the influencers
"""
function followership_ratings(B::BitMatrix, C)
    n, M = size(B)
    L = size(C, 2)

    R = zeros(Float64, M, L)
    for m in 1:M
        audience_m = findall(B[:, m])
        R[m, :] = count(C[audience_m, :]; dims=1) ./ n
    end

    return R
end

function influencer_switch_rates(X::A, Z::A, B, C, η::Float64;
                                 ψ=x -> exp_fast(-x), r=relu) where {T,A<:AbstractArray{T}}
    Rates = zeros(T, size(X, 1), size(Z, 1))
    influencer_switch_rates!(Rates, X, Z, B, C, η; ψ, r)
    return Rates
end

@doc raw"""
    influencer_switch_rates(X, Z, B, C, η; ψ = x -> exp(-x), r = relu)

Returns an n × L matrix where the `j,l`-th entry contains the rate λ of the Poisson point
process modeling how agent `j` switches influencers to `l`. Note that this is not the
same as ``Λ_{m}^{→l}``, defined as:
```math
\Lambda_{m}^{\to \ell} (x, t) = \eta \, \psi(\| z_{\ell} - x \|) \, r \left(
\frac{n_{m,\ell}(t)}{\sum_{m^\prime = 1}^{M} n_{m^\prime, \ell} (t) } \right).
```
"""
function influencer_switch_rates!(Ril, X::A, Z::A, B, C, η::Float64;
                                  ψ=x -> exp_fast(-x), r=relu) where {T,A<:AbstractArray{T}}

    # Compute the followership rate for media and influencers
    rate_m_l = followership_ratings(B, C)
    # attractiveness is the total proportion of followers per influencer
    # Divide each col of rates over sum(n_(m, l) for m = 1:M)
    @assert all(sum(rate_m_l; dims=1) .> 1e-5)
    struct_similarity = rate_m_l ./ sum(rate_m_l; dims=1)

    # Computing distances of each individual to the influencers
    D = zeros(T, size(X, 1), size(Z, 1))
    for (i, agent_i) in pairs(eachrow(X))
        for (l, influencer_l) in pairs(eachrow(Z))
            D[i, l] = ψ(norm(agent_i - influencer_l)) # FIXME: Eliminate broadcast
        end
    end

    # Calculating switching rate based on eq. (6)
    for (j, agentj_media) in pairs(eachrow(B))
        m = findfirst(agentj_media)
        # row_r = view(Ril, j, :)
        # row_d = view(D, j, :)
        # row_s = view(struct_similarity, m, :)
        # Ril[j, :] = η .* D[j, :] .* r.(struct_similarity[m, :]) # FIXME: Problematic line.

        for i in eachindex(Ril[j, :])
            # temp1 = η * D[j, i]
            # temp2 = temp1 * r(struct_similarity[m, i])
            # Ril[j, i] = temp1 * temp2
            temp1 = η * D[j, i]
            s_mi = struct_similarity[m, i]
            temp2 = r(s_mi)
            Ril[j, i] = temp1 * temp2
        end
    end

    return nothing
end

"""
    switch_influencer(C, X, Z, B, η, dt)

Simulates the Poisson point process that determines how agents change influencers based on
the calculated switching rates via a Tau-Leaping-like approach.

See also [`influencer_switch_rates`](@ref)
"""
function switch_influencer(C, X::T, Z::T, rates::U,
                           dt) where {T,U<:AbstractVecOrMat}
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
