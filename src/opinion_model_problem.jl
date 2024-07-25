
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

# iteration for destructuring into components
Base.iterate(p::ModelParams) = (p.L, Val(:M))
Base.iterate(p::ModelParams, ::Val{:M}) = (p.M, Val(:n))
Base.iterate(p::ModelParams, ::Val{:n}) = (p.n, Val(:η))
Base.iterate(p::ModelParams, ::Val{:η}) = (p.η, Val(:a))
Base.iterate(p::ModelParams, ::Val{:a}) = (p.a, Val(:b))
Base.iterate(p::ModelParams, ::Val{:b}) = (p.b, Val(:c))
Base.iterate(p::ModelParams, ::Val{:c}) = (p.c, Val(:σ))
Base.iterate(p::ModelParams, ::Val{:σ}) = (p.σ, Val(:σ̂))
Base.iterate(p::ModelParams, ::Val{:σ̂}) = (p.σ̂, Val(:σ̃))
Base.iterate(p::ModelParams, ::Val{:σ̃}) = (p.σ̃, Val(:γ))
Base.iterate(p::ModelParams, ::Val{:γ}) = (p.frictionI, Val(:Γ))
Base.iterate(p::ModelParams, ::Val{:Γ}) = (p.frictionM, Val(:done))
Base.iterate(p::ModelParams, ::Val{:done}) = nothing

"""
    OpinionModelProblem{T, D}

encapsulates parameters and other properties of an Agent-Based model of Opinion Dynamics.
"""
struct OpinionModelProblem{T<:AbstractFloat,D}
    p::ModelParams{T} # Model parameters
    domain::NTuple{D,Tuple{T,T}} # Bounds of opinion space. Exactly D 2-ples of (min, max)
    X::AbstractVecOrMat{T} # Array of Agents' positions
    M::AbstractVecOrMat{T} # Array of Media positions
    I::AbstractVecOrMat{T} # Array of Influencers' positions
    AgInfNet::BitMatrix # Adjacency matrix of Agents-Influencers
    AgAgNet::BitMatrix # Adjacency matrix of Agent-Agent interactions
    AgMedNet::BitMatrix # Agent-media correspondence vector
end

function Base.show(io::IO, omp::OpinionModelProblem{T,D}) where {T,D}
    return print("""
                 $(D)-dimensional Agent Based Opinion Model with:
                 - $(omp.p.n) agents
                 - $(omp.p.M) media outlets
                 - $(omp.p.L) influencers
                 """)
end

# function OpinionModelProblem(dom::Vararg{Tuple{Real,Real},D}; p=ModelParams(),
#                              seed=MersenneTwister(),
#                              AgAgNetF::Function=I -> trues(p.n, p.n),) where {D<:Real}
#     @info "Promoting elements of domain tuples"
#     throw(ErrorException("Mixed-type tuples are not yet supported"))
#     # return OpinionModelProblem(promote(dom...), seed, AgAgNetF)
# end

function OpinionModelProblem(dom::Vararg{Tuple{T,T},D}; p=ModelParams(),
                             seed=MersenneTwister(),
                             AgAgNetF::Function=I -> trues(p.n, p.n)) where {D,
                                                                              T<:AbstractFloat}
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

    return OpinionModelProblem{T,D}(X, I; p=p, dom=dom, AgAgNetF=AgAgNetF)
end

function OpinionModelProblem{T,D}(agents_init::AbstractArray{T},
                                  influencers_init::AbstractArray{T};
                                  p=ModelParams(; L=size(influencers_init, 1),
                                                n=size(agents_init, 1)),
                                  dom::NTuple{D,Tuple{T,T}}=_array_bounds(agents_init),
                                  AgAgNetF::Function=I -> trues(p.n, p.n)) where {D,
                                                                                  T<:AbstractFloat}
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
    M = vcat(fill(-one(T), (1, D)), fill(one(T), (1, D)))

    return OpinionModelProblem{T,D}(p, dom, agents_init, M, influencers_init, AgInfNet,
                                    AgAgNet,
                                    AgMedNet)
end

# iteration for destructuring into components
Base.iterate(omp::OpinionModelProblem) = (omp.X, Val(:M))
Base.iterate(omp::OpinionModelProblem, ::Val{:M}) = (omp.M, Val(:I))
Base.iterate(omp::OpinionModelProblem, ::Val{:I}) = (omp.I, Val(:AgAgNet))
Base.iterate(omp::OpinionModelProblem, ::Val{:AgAgNet}) = (omp.AgAgNet, Val(:AgMedNet))
Base.iterate(omp::OpinionModelProblem, ::Val{:AgMedNet}) = (omp.AgMedNet, Val(:AgInfNet))
Base.iterate(omp::OpinionModelProblem, ::Val{:AgInfNet}) = (omp.AgInfNet, Val(:done))
Base.iterate(omp::OpinionModelProblem, ::Val{:done}) = nothing

# Supporting structs for the SciML DiffEq based solver

abstract type AbstractSolver end
struct BespokeSolver <: AbstractSolver
    tstops::AbstractVector{Float64}
end
struct DiffEqSolver <: AbstractSolver
    sol::SciMLBase.RODESolution
    # abstol::AbstractFloat # tolerance value of the diff.eq. solver. Used to compare
    # TODO: Check if I can put the `sol` from DiffEq here to get access to the solution
    # interpolator to make comparisons at the exact same timepoints.
end

struct OpinionModelSimulation{T<:AbstractFloat,D,S<:AbstractSolver}
    p::ModelParams{T} # Model parameters
    dom::NTuple{D,Tuple{T,T}} # Domain of Opinion Space
    nsteps::Integer # Number of steps the solver used
    solver::S
    X::AbstractArray{T,3} # Array of Agents' positions
    Y::AbstractArray{T,3} # Array of Media positions
    Z::AbstractArray{T,3} # Array of Influencers' positions
    C::BitArray{3} # Adjacency matrix of Agents-Influencers
    R::AbstractArray{T,3} # Computed influencer switching rates for Agents
end

# Implement destructuring via iteration
Base.iterate(oms::OpinionModelSimulation) = (oms.X, Val(:Y))
Base.iterate(oms::OpinionModelSimulation, ::Val{:Y}) = (oms.Y, Val(:Z))
Base.iterate(oms::OpinionModelSimulation, ::Val{:Z}) = (oms.Z, Val(:C))
Base.iterate(oms::OpinionModelSimulation, ::Val{:C}) = (oms.C, Val(:R))
Base.iterate(oms::OpinionModelSimulation, ::Val{:R}) = (oms.R, Val(:done))
Base.iterate(oms::OpinionModelSimulation, ::Val{:done}) = nothing

Base.length(oms::OpinionModelSimulation) = size(oms.X, 3)
Base.eltype(oms::OpinionModelSimulation{T,D,S}) where {T,D,S} = T
solvtype(oms::OpinionModelSimulation{T,D,S}) where {T,D,S} = S

function OpinionModelSimulation{T,D,DiffEqSolver}(sol::S,
                                                  dom::NTuple{D,Tuple{T,T}},
                                                  cache::IntCache,
                                                  p::ModelParams{T}) where {T,D,
                                                                            S<:SciMLBase.AbstractODESolution,
                                                                            IntCache<:DiffEqCallbacks.SavedValues}
    U = reshape(sol, p.n + p.L + p.M, :, length(sol)) # FIXME: Maybe use `stack`

    # FIXME: I could hard code this, or use the smart version published in
    # https://julialang.org/blog/2016/02/iteration/.
    agents = CartesianIndices((firstindex(U):(p.n), axes(U, 2), axes(U, 3)))
    media = CartesianIndices(((p.n + 1):(p.n + p.M), axes(U, 2), axes(U, 3)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(U, 2),
                                    axes(U, 3)))

    # Assigning variable names to vector of solutions for readability
    X = @view U[agents]
    Y = @view U[media]
    Z = @view U[influencers]

    C = stack(cache.saveval)
    # FIXME: Not saving the rates for now for convenience. But leaving the possibility open
    R = zeros(Float64, p.n, p.L, length(sol))

    solver_meta = DiffEqSolver(sol)

    return OpinionModelSimulation{T,D,DiffEqSolver}(p, dom, length(sol.t), solver_meta, X,
                                                    Y, Z, C, R)
end

function Base.show(io::IO, oms::OpinionModelSimulation)
    return print("""
                 Simulation of the ABM Opinion Model with:
                 - $(oms.p.n) agents
                 - $(oms.nsteps) time steps
                 - Solved with: $(solvtype(oms))
                 """)
end

# Functions for comparing solutions

function Base.:-(s1::SBe,
                 s2::SD) where {T,D,SBe<:OpinionModelSimulation{T,D,BespokeSolver},
                                SD<:OpinionModelSimulation{T,D,DiffEqSolver}}
    interpolated_sol = s2.solver.sol.(s1.solver.tstops) |> stack
    stacked_sol = vcat(s1.X, s1.Y, s1.Z)

    return stacked_sol .- interpolated_sol
end

function Base.:-(s1::SD,
                 s2::SBe) where {T,D,SBe<:OpinionModelSimulation{T,D,BespokeSolver},
                                 SD<:OpinionModelSimulation{T,D,DiffEqSolver}}
    interpolated_sol = s1.solver.sol.(s2.solver.tstops) |> stack
    stacked_sol = vcat(s2.X, s2.Y, s2.Z)

    return interpolated_sol .- stacked_sol
end

# FIXME: Make these operators commutative, if not already so.

function Base.isapprox(s1::Sim, s2::Sim; rtol::Real=atol > 0 ? 0 : √eps(T),
                       atol::Real=0) where {T,D,
                                            Sim<:OpinionModelSimulation{T,D,BespokeSolver}}
    # Check maximum elementwise differences are below the tolerance
    ΔX = s1.X .- s2.X
    ΔY = s1.Y .- s2.Y
    ΔZ = s1.Z .- s2.Z

    return arrays_areapprox(ΔX, ΔY, ΔZ, atol, rtol)
end

function Base.isapprox(s1::SBe, s2::SD; rtol::Real=atol > 0 ? 0 : √eps(T),
                       atol::Real=0) where {T,D,
                                            SBe<:OpinionModelSimulation{T,D,BespokeSolver},
                                            SD<:OpinionModelSimulation{T,D,DiffEqSolver}}
    return Δ_isapprox(s1 - s2, atol, rtol)
end