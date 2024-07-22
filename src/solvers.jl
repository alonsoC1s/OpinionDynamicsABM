
"""
    simulate!(omp::OpinionModelProblem; Nt=100, dt=0.01, method=:other)

Simulates the evolution of the Opinion Dynamics problem `omp` by solving the associated
SDE via Euler--Maruyama with `Nt` time steps and resolution `dt`.
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

    # return rX, rY, rZ, rC, rR
    return OpinionModelSimulation(omp.p, rX, rY, rZ, rC, rR)
end

function drift(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    influencers = CartesianIndices(((p.n + 1):(p.n + p.L), axes(u, 2)))
    media = CartesianIndices(((p.n + p.L + 1):size(u, 1), axes(u, 2)))

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
    return nothing
end

function noise(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(du, 2)))
    influencers = CartesianIndices(((p.n + 1):(p.L), axes(du, 2)))
    media = CartesianIndices(((p.L + 1):(p.M), axes(du, 2)))

    # Additive noise
    du[agents] .= p.σ
    du[influencers] .= p.σ̂
    du[media] .= p.σ̃
    return nothing
end

function influencer_switch_affect!(integrator)
    dt = get_proposed_dt(integrator)
    u = integrator.u
    p = integrator.p
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    influencers = CartesianIndices(((p.n + 1):(p.n + p.L), axes(u, 2)))

    # Assigning variable names to vector of solutions for readability
    X = @view u[agents]
    Z = @view u[influencers]

    # 
    rates = influencer_switch_rates(X, Z, p.B, p.C, p.η)
    return integrator.p.C .= switch_influencer(p.C, X, Z, rates, dt)
end

true_condition = function (u, t, integrator)
    true
end

influencer_switching_callback = DiscreteCallback(true_condition, influencer_switch_affect!)

function build_sdeproblem(omp::OpinionModelProblem{T}, time::Tuple{T, T}) where {T}
    p = omp.p
    # Stack all important parameters to be fed to the integrator
    P = (L=p.L, M=p.M, n=p.n, η=p.η, a=p.a, b=p.b, c=p.c, σ=p.σ, σ̂=p.σ̂, σ̃=p.σ̃,
         A=omp.AgAgNet, B=omp.AgMedNet, C=omp.AgInfNet, p=p)

    u₀ = vcat(omp.X, omp.I, omp.M)

    return SDEProblem(drift, noise, u₀, time, P)
end

function simulate!(omp::OpinionModelProblem{T}, time::Tuple{T,T};
                   seed=MersenneTwister()) where {T}
    # Seeding the RNG
    Random.seed!(seed)

    problem = build_sdeproblem(omp, time)

    return solve(problem, SRIW1(); callback=influencer_switching_callback)
end

function simulate!(omp::SDEProblem; seed = MersenneTwister())
    # Seeding the RNG
    Random.seed!(seed)

    return solve(omp, SRIW1(); callback=influencer_switching_callback)
end
