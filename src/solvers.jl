
"""
    simulate!(omp::OpinionModelProblem; Nt=200, dt=0.01, [seed])

Simulates the evolution of the Opinion Dynamics model `omp` by solving the defining SDE
via a hand-implemented Euler--Maruyama integrator with `Nt` steps of size `dt`.
"""
function simulate!(omp::OpinionModelProblem{T};
                   Nt=200,
                   dt=0.01,
                   seed=MersenneTwister(),
                   echo_chamber::Bool=false)::OpinionModelSimulation{T} where {T}
    X, Y, Z, A, B, C = omp
    L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ = omp.p
    d = size(X, 2)

    # Seeding the RNG
    Random.seed!(seed)

    # Allocating solutions & setting initial conditions
    rX = zeros(T, n, d, Nt)
    rY = zeros(T, M, d, Nt)
    rZ = zeros(T, L, d, Nt)
    rC = BitArray{3}(undef, n, L, Nt)
    # Jump rates can be left uninitialzied. Not defined for the last time step
    rR = Array{T,3}(undef, n, L, Nt - 1)

    rX[:, :, begin] = X
    rY[:, :, begin] = Y
    rZ[:, :, begin] = Z
    rC[:, :, begin] = C

    # Solve with Euler-Maruyama
    t_points = 1:(Nt - 1)
    for i in t_points
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

    solver_meta = BespokeSolver(collect(range(zero(T), step = dt, length = Nt)))

    # return rX, rY, rZ, rC, rR
    return OpinionModelSimulation{T,BespokeSolver}(omp.p, Nt, solver_meta, rX, rY, rZ, rC,
                                                   rR)
end

function drift(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    media = CartesianIndices(((p.n +1):(p.n + p.M), axes(u, 2)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(u ,2)))

    # Assigning variable names to vector of solutions for readability
    X = @view u[agents]
    Y = @view u[media]
    Z = @view u[influencers]

    # Agents SDE
    du[agents] .= agent_drift(X, Y, Z, p.A, p.B, p.C, p.p)
    # Media drift
    du[media] .= media_drift(X, Y, p.B)
    # Influencer SDE
    du[influencers] .= influencer_drift(X, Z, p.C)

    return nothing
end

function noise(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    media = CartesianIndices(((p.n +1):(p.n + p.M), axes(u, 2)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(u ,2)))

    # Additive noise
    du[agents] .= p.σ
    du[media] .= p.σ̃
    du[influencers] .= p.σ̂

    return nothing
end

# SDE Callbacks

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

    rates = influencer_switch_rates(X, Z, p.B, p.C, p.η)
    new_C = switch_influencer(integrator.p.C, X, Z, rates, dt)
    integrator.p.C .= new_C
    return nothing
end

true_condition = function (u, t, integrator)
    return true
end

influencer_switching_callback = DiscreteCallback(true_condition, influencer_switch_affect!;
                                                 save_positions=(true, true))

# save_B(u, t, integrator) = integrator.p.C
save_B = function (u, t, integrator)
    return integrator.p.C
end

function build_sdeproblem(omp::OpinionModelProblem{T}, time::Tuple{T,T}) where {T}
    mp = omp.p
    # Stack all important parameters to be fed to the integrator
    P = (L=mp.L, M=mp.M, n=mp.n, η=mp.η, a=mp.a, b=mp.b, c=mp.c, σ=mp.σ, σ̂=mp.σ̂, σ̃=mp.σ̃,
         A=omp.AgAgNet, B=omp.AgMedNet, C=omp.AgInfNet, p=mp)

    u₀ = vcat(omp.X, omp.M, omp.I)

    return SDEProblem(drift, noise, u₀, time, P)
end

# Constructor to go directly from "native" problem def. to diffeq
function simulate!(omp::OpinionModelProblem{T}, time::Tuple{T,T};
                   seed=MersenneTwister())::OpinionModelSimulation where {T}
    # Seeding the RNG
    Random.seed!(seed)

    # Defining the callbacks
    B_cache = SavedValues(Float64, BitMatrix)
    saving_callback = SavingCallback(save_B, B_cache; save_everystep=true)
    cbs = CallbackSet(influencer_switching_callback, saving_callback)

    diffeq_prob = build_sdeproblem(omp, time)
    diffeq_sol = solve(diffeq_prob, SRIW1(); callback=cbs, alg_hints=:additive)

    # TODO: Maybe use the retcode from diffeq to warn here.

    return OpinionModelSimulation{T,DiffEqSolver}(diffeq_sol, B_cache, diffeq_prob.p.p)
end

function simulate!(sde_omp::SDEProblem; seed=MersenneTwister())
    # Seeding the RNG
    Random.seed!(seed)

    # Defining the callbacks
    B_cache = SavedValues(Float64, BitMatrix)
    saving_callback = SavingCallback(save_B, B_cache; save_everystep=true, save_start=true)

    cbs = CallbackSet(influencer_switching_callback, saving_callback)

    diffeq_sol = solve(sde_omp, SRIW1(); callback=cbs, alg_hints=:additive)
    return OpinionModelSimulation{Float64,DiffEqSolver}(diffeq_sol, B_cache, sde_omp.p.p)
end
