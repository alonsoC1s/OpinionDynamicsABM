
"""
    simulate!(omp::OpinionModelProblem; Nt=200, dt=0.01, [seed])

Simulates the evolution of the Opinion Dynamics model `omp` by solving the defining SDE
via a hand-implemented Euler--Maruyama integrator with `Nt` steps of size `dt`. Returns a
`ModelSimulation{BespokeSolver}`
"""
function simulate!(omp::OpinionModelProblem{T,D};
                   Nt=200,
                   dt=0.01,
                   seed=MersenneTwister(),
                   control::Bool=false) where {T,D}
    X, Y, Z, A, B, C = omp
    L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ = omp.p

    # Detect early if an agent is not connected to any Media Outlets
    if !(all(any(B; dims=2)))
        throw(ErrorException("Model violation detected: An agent is disconnected from " *
                             "all media outlets."))
    end

    # Seeding the RNG
    Random.seed!(seed)

    # Allocating solutions & setting initial conditions
    rX = zeros(T, n, D, Nt)
    rY = zeros(T, M, D, Nt)
    rZ = zeros(T, L, D, Nt)
    rA = similar(A, n, n, Nt)
    rC = similar(C, n, L, Nt)
    # Jump rates can be left uninitialized. Not defined for the last time step
    rR = similar(X, n, L, Nt - 1)

    rX[:, :, begin] = X
    rY[:, :, begin] = Y
    rZ[:, :, begin] = Z
    rA[:, :, begin] = A
    rC[:, :, begin] = C

    # Reusable arrays for forces, distances and weights
    # FIXME: Reusable arrays will be sparse whenever A is
    FA = similar(X)
    Ftmp = similar(FA)
    Dijd = similar(X, n, n, D)
    Wij = similar(X, n, n)

    # Solve with Euler-Maruyama
    t_points = 1:(Nt - 1)
    @inbounds for i in t_points
        X = view(rX, :, :, i)
        Y = view(rY, :, :, i)
        Z = view(rZ, :, :, i)
        C = view(rC, :, :, i) # |> BitMatrix
        A = view(rA, :, :, i)

        ## Check network consistency
        # Detect early if an agent doesn't follow any influencers
        if !(all(any(C; dims=2)))
            throw(ErrorException("Model violation detected: An Agent doesn't follow any  " *
                                 "influencers"))
        end

        # FIXME: Try using the dotted operators to fuse vectorized operations
        # Agents movement
        agent_drift!(FA, Ftmp, Dijd, Wij, X, Y, Z, A, B, C, a, b, c)
        # FA was mutated by previous line
        rX[:, :, i + 1] .= X + dt * FA + σ * sqrt(dt) * randn(n, D)

        # Media movements
        FM = media_drift(X, Y, B, Γ)
        rY[:, :, i + 1] .= Y + dt * FM + (σ̃ / Γ) * sqrt(dt) * randn(M, D)

        # Influencer movements
        FI = influencer_drift(X, Z, C, γ)
        rZ[:, :, i + 1] .= Z + dt * FI + (σ̂ / γ) * sqrt(dt) * randn(L, D)

        # Change influencers
        rates = influencer_switch_rates(X, Z, B, C, η)
        rR[:, :, i] .= rates
        R = view(rR, :, :, i)
        view(rC, :, :, i + 1) .= switch_influencer(C, X, Z, R, dt)

        if control && i >= 10
            # Modify Agent-Agent interaction network
            # A .= _ag_ag_echo_chamber(BitMatrix(rC[:, :, i + 1]))
            fill!(A, zero(T))
            # _antidiagonal!(A)
        end

        # Record changes to Agent-Agent adj matrix
        rA[:, :, i + 1] .= A
    end

    solver_meta = BespokeSolver(collect(range(zero(T); step=dt, length=Nt)))

    # return rX, rY, rZ, rC, rR
    return ModelSimulation{T,D,BespokeSolver}(omp.p, omp.domain, Nt, solver_meta, rX,
                                              rY, rZ, rC, rR)
end

function drift(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    media = CartesianIndices(((p.n + 1):(p.n + p.M), axes(u, 2)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(u, 2)))

    # Assigning variable names to vector of solutions for readability
    X = @view u[agents]
    Y = @view u[media]
    Z = @view u[influencers]

    # Agents SDE
    du[agents] .= agent_drift(X, Y, Z, p.A, p.B, p.C, p.a, p.b, p.c)
    # Media drift
    du[media] .= media_drift(X, Y, p.B, p.Γ)
    # Influencer SDE
    du[influencers] .= influencer_drift(X, Z, p.C, p.γ)

    return nothing
end

function noise(du, u, p, t)
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(du):(p.n), axes(du, 2)))
    media = CartesianIndices(((p.n + 1):(p.n + p.M), axes(du, 2)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(du, 2)))

    # Additive noise
    du[agents] .= p.σ
    du[media] .= p.σ̃ / p.Γ
    du[influencers] .= p.σ̂ / p.γ

    return nothing
end

# SDE Callbacks

# FIXME: Make this a single, saving callback if possible
function influencer_switch_affect!(integrator)
    dt = get_proposed_dt(integrator)
    u = integrator.u
    p = integrator.p
    # Defining the indices for readability
    agents = CartesianIndices((firstindex(u):(p.n), axes(u, 2)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(u, 2)))

    # Assigning variable names to vector of solutions for readability
    X = @view u[agents]
    Z = @view u[influencers]

    rates = influencer_switch_rates(X, Z, integrator.p.B, integrator.p.C, p.η)
    new_C = switch_influencer(integrator.p.C, X, Z, rates, dt)
    integrator.p.C .= new_C
    return nothing
end

true_condition = function (u, t, integrator)
    return true
end

influencer_switching_callback = DiscreteCallback(true_condition, influencer_switch_affect!;
                                                 save_positions=(true, false))

# save_B(u, t, integrator) = integrator.p.C
save_C = function (u, t, integrator)
    return integrator.p.C
end

"""
    build_sdeproblem(omp::OpinionModelProblem, tspan)

Returns an `SDEProblem` suitable to be used with the SciML DifferentialEquations.jl
ecosystem.
"""
function build_sdeproblem(omp::OpinionModelProblem{T,D}, tspan::Tuple{T,T}) where {T,D}
    mp = omp.p
    X, Y, Z, A, B, C = omp
    L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ = omp.p
    # Stack all important parameters to be fed to the integrator
    P = (L=L, M=M, n=n, η=η, a=a, b=b, c=c, σ=σ, σ̂=σ̂, σ̃=σ̃, γ=γ, Γ=Γ, A=A, B=B, C=C,
         p=mp)

    u₀ = vcat(X, Y, Z)

    return SDEProblem(drift, noise, u₀, tspan, P)
end

"""
    simulate!(omp::OpinionModelProblem, tspan; dt=0.01, [seed])

Simulates the evolution of the Opinion Dynamics model `omp` by solving the defining SDE
via a DifferentialEquations.jl with the `SRIW1` algorithm and the appropriate callbacks.
Produces a `ModelSimulation{DiffEqSolver}`.
"""
function simulate!(omp::OpinionModelProblem{T,D}, tspan::Tuple{T,T}; dt::T=0.01,
                   seed=MersenneTwister())::ModelSimulation where {T,D}
    # Seeding the RNG
    Random.seed!(seed)

    # Defining the callbacks
    C_cache = SavedValues(Float64, BitMatrix)
    saving_callback = SavingCallback(save_C, C_cache; saveat=dt, save_end=false,
                                     save_start=true)
    influencer_switching_timed_cb = PeriodicCallback(influencer_switch_affect!, dt)
    # cbs = CallbackSet(influencer_switching_callback, saving_callback)
    cbs = CallbackSet(influencer_switching_timed_cb, saving_callback)

    diffeq_prob = build_sdeproblem(omp, tspan)
    diffeq_sol = solve(diffeq_prob, SRIW1(); callback=cbs, alg_hints=:additive,
                       save_everystep=true)

    # TODO: Maybe use the retcode from diffeq to warn here.

    return ModelSimulation{T,D,DiffEqSolver}(diffeq_sol, omp.domain, C_cache,
                                             diffeq_prob.p.p)
end

"""
    simulate!(sde_omp::SDEProblem; dt=0.01, [seed])

Simulates the opinion model problem defined in `sde_problem` by applying the appropriate
callbacks and calling `solve` on the SDE. Produces a `ModelSimulation{DiffEqSolver}`.
"""
function simulate!(sde_omp::SDEProblem; dt::T=0.01, seed=MersenneTwister()) where {T}
    # Seeding the RNG
    Random.seed!(seed)

    # Defining the callbacks
    B_cache = SavedValues(Float64, BitMatrix)
    saving_callback = SavingCallback(save_B, B_cache; saveat=dt, save_end=false,
                                     save_start=true)
    cbs = CallbackSet(influencer_switching_callback, saving_callback)

    diffeq_sol = solve(sde_omp, SRIW1(); callback=cbs, alg_hints=:additive,
                       save_everystep=true)

    domain = _array_bounds(sde_omp.u0) # Domain is the bounding box of initial opinions

    return ModelSimulation{Float64,2,DiffEqSolver}(diffeq_sol, domain, B_cache,
                                                   sde_omp.p.p)
end
