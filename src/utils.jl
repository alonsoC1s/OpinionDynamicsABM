## Maybe Fix:
# _boolean_combinator gives the orthant's signatures in a "weird order". Instead of
# numbering counter clokwise, the numbering starts at the all-positive quadrant, then moves
# Down, then Up-Left, then down. In 3 dimensions:
# If z >= 0     If z < 0
#      |            |
#   5 | 1         6 | 2
# ---------     --------- 
#   7 | 3         8 | 4
#     |             |
"""
    _boolean_combinator(n::Integer)::BitMatrix

Given `n` boolean variables, generates all the possible logical combinations for those
variables. e.g for A, B, it generates (true, true), (true, false), (false, true), (false,
false).

Code inspired by TruthTables.jl (macroutils.jl)
https://github.com/eliascarv/TruthTables.jl/blob/main/src/macroutils.jl#L44
"""
function _boolean_combinator(n::Integer)::BitMatrix
    bools = [true, false]
    outers = [2^x for x in 0:(n - 1)]
    inners = reverse(outers)
    return reduce(hcat,
                  [repeat(bools; inner, outer) for (inner, outer) in zip(inners, outers)])
end

"""
    _orthantize(X :: AbstractArray{Float64, N})

Classifies the `n` agents in X into the 2^`N` orthants (or quadrants) by returning an
adjacency matrix of size n × 2^`N` where the i-th row has a single true entry representing
which orthant the i-th agent is in.

FIXME: Might be overcomplicated. Perhaps achieved with `A .> [0, 0]` (simplified idea)
"""
function _orthantize(X)
    N = size(X, 2) # Dimensions of the problem (i.e N if R^N is the problem space)
    orth_chart = similar(X, Bool)
    # For each row of X (i.e agent), we check if it's on the >=0 semispace for each variable
    for (agent_coord, coord_idx) in zip(eachcol(X), axes(orth_chart, 2))
        orth_chart[:, coord_idx] = agent_coord .>= 0
    end

    # We compare the "orth_chart" with the sequence of booleans that characterize an
    # orthant.  e.g The first quadrant (in R^2) can be characterized as (true, true),
    # since x and y are >= 0. The first octant is then (true, true, true) and so on.

    # The "signature" of an orthant is a sequence of booleans s.t the i-th entry is true
    # if the i-th coordinate of points within is >= 0.
    orth_signatures = _boolean_combinator(N)

    orth_class = falses(size(X, 1), 2^N)

    for orthant_id in axes(orth_signatures, 1) # FIXME: Iter over rows, not cols
        signature = orth_signatures[orthant_id, :]
        for agent_id in axes(orth_chart, 1)
            agent = orth_chart[agent_id, :]

            orth_class[agent_id, orthant_id] = all(agent .== signature)
        end
    end

    return orth_class
end

"""
    _ag_ag_echo_chamber(AgInfNet::BitArray)

Constructs a fully echo-chamber Agent-Agent adjacency network. In other words, constructs
a network of Agents where each Agent is only connected to, and thus only influence by,
Agents that follow the same influencer.
"""
function _ag_ag_echo_chamber(AgInfNet::BitArray)
    n = size(AgInfNet, 1)
    AgAgNet = falses(n, n)
    clicque_subnet = falses(n, n)

    # For each influencer we find the followers and mutually connect all of them.
    for clicque in eachcol(AgInfNet)
        clicque_peers = findall(clicque) # Indices of agents following the current influencer.

        # We compute all possible pairs of agents withing the clicque (Cartesian product).
        all_pairs = Iterators.product(clicque_peers, clicque_peers)
        foreach(pair -> clicque_subnet[pair...] = true, all_pairs)

        # And-ing into AgAgNet to connect individuals that are on the same orthant.
        AgAgNet = AgAgNet .|| clicque_subnet
        # Reseting the current influencer's subnetwork for the next iteration
        fill!(clicque_subnet, false)
    end

    return AgAgNet
end

"""
    _place_influencers(X::AbstractArray{Float64, N}, AgInfNet::BitMatrix)

Returns the positions of the influencers in the problem space calculated as the barycenter
of the agents in each orthant
"""
function _place_influencers(X, AgInfNet)
    L = size(AgInfNet, 2) # Number of influencers
    N = size(X, 2) # Dimension of the problem space

    I = similar(X, L, N)
    for (infl_number, orthant_mask) in enumerate(eachcol(AgInfNet)) # FIXME: Use `pairs` instead of enumerate
        orthant_members_idx = findall(orthant_mask)
        I[infl_number, :] = mean(X[orthant_members_idx, :]; dims=1)
    end

    return I
end

"""
    _media_network(n::Int, M::Int)

Returns the adjacency matrix of `n` agents to `M` media outlets such that each agent is
connected to exactly one media outlet.
"""
function _media_network(n, M)::BitMatrix
    # Fill a vector with `n` powers of 2
    powers_of_2 = rand([2^i for i in 0:(M - 1)], n)
    C = BitMatrix(undef, (n, M))

    for (pow, c_row) in zip(powers_of_2, eachrow(C))
        # Get the binary representation of 2^i and store it in a row of C
        digits!(c_row, pow; base=2)
    end

    return C
end

function relu(x)
    return max(0.1, -1 + 2 * x)
end

# FIXME: This is faster for full Arrays, but BitArrays do findfirst smarter so this is
# potentially slower. The advantage on sparse arrays is not clear
function fragment_network(C::BitArray)
    # Thanks to the problem properties we know there are exactly n non-zeros in C
    n, L = size(C)

    # Sorted indices of agents such that member of the l-th clique are in positions c_(l-1):c_l
    agent_ids = Vector{Int}(undef, n)
    # Range limits c_l for agents in the cliques
    clique_limits = Vector{Int}(undef, L + 1)
    clique_limits[begin] = 0
    clique_limits[end] = n

    a, l = 1, 1 # Current index of agent_ids & clique_limits

    # Since we iterate by columns, cliques end when we move to the next col.
    for (cartInd, val) in pairs(IndexCartesian(), C)
        if val
            i, j = Tuple(cartInd)
            @inbounds agent_ids[a] = i
            # Is this true entry on the same column as last?  If not, a clique ends in the
            # (a-1)th entry of agent_ids
            l != j && (l += 1; @inbounds clique_limits[l] = a - 1)
            a += 1
        end
    end

    return agent_ids, ntuple(i -> (clique_limits[i] + 1):clique_limits[i + 1], L)
end

function time_rate_tensor(R::AbstractArray{U,3}, C::BitArray{3}) where {U<:Real}
    n, L, T = size(R)

    @assert size(R, 3) == size(C, 3)
    # Λ = Array{T, 3}(undef, n, n, T)
    Λ = similar(R, L, L, T)

    # for (R_t, C_t) = zip(eachslice(R; dims=3), eachslice(C; dims=3))
    for t in 1:T
        C_t, R_t = view(C, :, :, t), view(R, :, :, t)
        # Set "staying" rates to zero
        leaving_rates = .!C_t .* R_t
        # use fragmented network to sum leaving rate for whole clique at once
        clique_ids, clique_bounds = fragment_network(BitMatrix(C_t))
        for (l, range) in pairs(clique_bounds)
            clique_rates = leaving_rates[clique_ids[range], :]
            λ = vec(sum(clique_rates; dims=1))
            @inbounds Λ[:, l, t] = λ
            @inbounds Λ[l, l, t] = -sum(λ)
        end
    end
    return Λ
end

# TODO: Test legacy functions

function legacy_rates(B, x, FolInfNet, inf, eta)
    n, L = size(x, 1), size(inf, 1)

    state = replace(findfirst.(eachrow(B)) .== 2, 0 => -1)
    theta = 0.1 # threshold for r-function

    fraction = zeros(L)
    for i in 1:L
        fraction[i] = sum(FolInfNet[:, i] .* state) / sum(FolInfNet[:, i])
    end

    # compute distance of followers to influencers
    dist = zeros(n, L)
    for i in 1:L
        for j in 1:n
            d = x[j, :] - inf[i, :]
            dist[j, i] = exp(-sqrt(d[1]^2 + d[2]^2))
        end
    end

    # compute attractiveness of influencer for followers
    attractive = zeros(n, L)
    for j in 1:n
        for i in 1:L
            g2 = state[j] * fraction[i]
            # The `if` emulates relu(x) = max(0.1, -1 + 2*x)
            if g2 < theta
                g2 = theta
            end
            attractive[j, i] = eta * dist[j, i] * g2
        end
    end

    return attractive
end

function legacy_media_drift(FolInfNet, xold, inf; dt=0.01)
    masscenter = zeros(L, 2)

    for i in 1:L
        if sum(FolInfNet[:, i]) > 0
            masscenter[i, :] = sum(FolInfNet[:, i] .* xold; dims=1) / sum(FolInfNet[:, i])
            inf[i, :] = inf[i, :] +
                        dt / frictionI * (masscenter[i, :] - inf[i, :]) +
                        1 / frictionI * sqrt(dt) * sigmahat * randn(2, 1)
        else
            # FIXME: Should `infold` be on the rhs? Or, shouldn't inf[i+1, :] be?
            inf[i, :] = inf[i, :] + 1 / frictionI * sqrt(dt) * sigmahat * randn(2, 1)
        end
    end
end

function legacy_changeinfluencer(B, x, FolInfNet, inf, eta, dt=0.01)
    n, L = size(x, 1), size(inf, 1)

    state = replace(findfirst.(eachrow(B)) .== 2, 0 => -1)
    theta = 0.1 # threshold for r-function
    fraction = zeros(L)

    for i in 1:L
        fraction[i] = sum(FolInfNet[:, i] .* state) / sum(FolInfNet[:, i])
    end

    # compute distance of followers to influencers
    dist = zeros(n, L)
    for i in 1:L
        for j in 1:n
            d = x[j, :] - inf[i, :]
            dist[j, i] = exp(-sqrt(d[1]^2 + d[2]^2))
        end
    end

    # compute attractiveness of influencer for followers
    attractive = zeros(n, L)
    for j in 1:n
        for i in 1:L
            g2 = state[j] * fraction[i]
            if g2 < theta
                g2 = theta
            end
            attractive[j, i] = eta * dist[j, i] * g2
        end

        r = rand()
        lambda = sum(attractive[j, :])
        if r < 1 - exp(-lambda * dt)
            p = attractive[j, :] / lambda
            r2 = rand()
            k = 1
            while sum(p[1:k]) < r2
                k = k + 1
            end
            FolInfNet[j, :] = zeros(L)
            FolInfNet[j, k] = 1
        end
    end

    return FolInfNet
end

function legacy_influence(x, media, inf, FolInfNet, state, (p, q))
    (; n, b, c, L) = q
    force1 = zeros(size(x))
    force2 = zeros(size(x))
    for j in 1:n
        if state[j] == 1
            force1[j, :] = media[2, :] - x[j, :]
        else
            force1[j, :] = media[1, :] - x[j, :]
        end

        for k in 1:L
            if FolInfNet[j, k] == 1
                force2[j, :] = inf[k, :] - x[j, :]
            end
        end
    end

    force = c * force1 + b * force2
    return force
end

function legacy_attraction(x, IndNet)
    n = size(IndNet, 1)
    force = zeros(n, 2)
    for j in 1:n
        Neighb = findall(x -> x == 1, IndNet[j, :]) # findall doesn't need a predicate. If the matrix is bool you can just get the indices directly without the predicate x -> x == 1
        if isempty(Neighb)
            force[j, :] = [0 0]
        else
            fi = [0 0]
            wsum = 0
            for i in eachindex(Neighb) # For each neighbor of j-th agent
                d = x[Neighb[i], :] - x[j, :]
                w = exp(-sqrt(d[1]^2 + d[2]^2))
                fi = fi + w * d'
                wsum = wsum + w
            end
            force[j, :] = fi / wsum
        end
    end
    return force
end
