
"""
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
- `γ`: friction constant for influencers
- `Γ`: friction constant for media outlets
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
    γ::T
    Γ::T
end

function ModelParams(L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ)
    return ModelParams(L, M, n, promote(η, a, b, c, σ, σ̂, σ̃, γ, Γ)...)
end

"""
    ModelParams(L=4, M=2, n=250, η=15, a=1, b=2, c=4, σ=0.5, σ̂=0, σ̃=0, γ=10, Γ=100)

Creates a `ModelParams` instance with the default values shown in the signature. Any
parameter can be modified by passing it as a keyword argument.

# Examples
```julia-repl
julia> deterministic_params = ModelParams(; σ=0.0, σ̂=0, σ̃=0)
```
"""
function ModelParams(;
                     L=4, M=2, n=250, η=15, a=1, b=2, c=4, σ=0.5, σ̂=0, σ̃=0, γ=10, Γ=100)
    return ModelParams(L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ)
end

function Base.show(::IO, params::ModelParams)
    L, M, n, η, a, b, c, σ, σ̂, σ̃, γ, Γ = params
    return print("Opinion Model parameters: η=$(η), a=$(a), b=$(b), c=$(c), σ=$(σ), " *
                 "σ_hat=$(σ̂), σ_tilde=$(σ̃), γ=$(γ), Γ=$(Γ)")
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
Base.iterate(p::ModelParams, ::Val{:γ}) = (p.γ, Val(:Γ))
Base.iterate(p::ModelParams, ::Val{:Γ}) = (p.Γ, Val(:done))
Base.iterate(p::ModelParams, ::Val{:done}) = nothing

# Implementing networks as either bit arrays or sparse arrays, or views of either
const BitOrViewMatrix = Union{BitMatrix, SubArray{Bool,2,BitMatrix, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}
const SparseOrViewMatrix{T} = Union{SparseMatrixCSC{T}, SubArray{T, 2, <:AbstractSparseMatrix, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, false}}
const AdjMatrix{T} = Union{BitOrViewMatrix, SparseOrViewMatrix{T}}

"""
Represents a `D`-dimensional opinion dynamics problem with specific `ModelParams`. See
[`ModelParams`](@ref).

## Fields
- `p`: Instance of `ModelParams` defining problem-wide parameters.
- `domain`: `D` 2-tuples representing the bounds of each dimenion of the opinion space.
- `X`: Matrix of shape (`p.n` × `D`) containing the coordinates of agents in the model.
    Agents are represented as rows.
- `Y`: Matrix storing the coordinates of media agents of shape similar to `X`
- `Z`: Matrix storing the coordinates of influencers.
- `A`: Agent-Agent adjacency matrix
- `B`: Agent-Media adjacency matrix
- `C`: Agent-Influencer adjacency matrix

## Constructors
There are several ways of creating an instance of `OpinionModelProblem` depending on the
initial data at hand:

-   `OpinionModelProblem(domain..., p=ModelParams(), seed=MersenneTwister(), [AgAgNetF])`:
    Creates a problem with the default parameters by specifying bounds of the
    `D`-dimensional, rectangular opinion space (i.e. the space where the simulation will
    take place). The bounds are given as a succesion of tuples. Agents are distributed
    uniformly along the opinion space, setting a seed manually will affect the initial
    positions. This is intended to be the main constructor

!!! warning
    Specifying `domain` with mixed-type tuples will fail (and with a cryptic error).
    Instead of passing `(-2.0, 2)`, pass `(-2.0, 2.0)`. See examples below.

- `OpinionModelProblem(X₀ Z₀; p=ModelParams(), domain::NTuple{D,Tuple{T,T})`: Creates a
    problem by providing the initial positions of agents and influencers. The model
    parameters and domain of opinion space can optionally be given manually. If ommited,
    the `ModelParams` are the default parameters (see [`ModelParams`](@ref)) and the
    `domain` is inferred as the minima and maxima of `X₀ ∪ Z₀`.

-  `OpinionModelProblem(p::ModelParams, domain::NTuple{D, Tuple{T, T}}, X, Y, Z, A, B, C)`:
    creates an opinion problem by explicitly filling every field. The constructor is not
    intended to be used directly.

## Examples

```julia
deterministic_params = ModelParams(;σ=0.0, σ̂=0, σ̃=0)
# Create a problem over `[-2, 2] × [-2, 2]` with stochastic noise turned off
omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); p=deterministic_params)
# Will throw an error because of the mixed-type tuples
OpinionModelProblem((-2.0, 2), (-2, 2))
```
"""
struct OpinionModelProblem{T<:AbstractFloat,D}
    p::ModelParams{T} # Model parameters
    domain::NTuple{D,Tuple{T,T}} # Bounds of opinion space. Exactly D 2-ples of (min, max)
    X::AbstractVecOrMat{T} # Agent coordinates
    Y::AbstractVecOrMat{T} # Media coordinates
    Z::AbstractVecOrMat{T} # Influencer coordinates
    A::AdjMatrix{T} # Agent-Agent adjacency matrix
    B::AdjMatrix{T} # Agent-Media adjacency matrix
    C::AdjMatrix{T} # Agent-Influencer adjaceny matrix

    # FIXME: Implement internal constructor to enforce invariants like symmetry
    function OpinionModelProblem{T,D}(p, domain, X, Y, Z, A, B, C) where {T,D}
        any([A[i, i] for i in axes(A, 1)]) &&
            throw(ArgumentError("Agents have self-connections"))

        issymmetric(A) || throw(ArgumentError("Matrix A is not symmetric"))

        # X = sortslices(X; dims=1)
        return new{T,D}(p, domain, X, Y, Z, A, B, C)
    end
end

function Base.show(::IO, omp::OpinionModelProblem{T,D}) where {T,D}
    return print("""
                 $(D)-dimensional Agent Based Opinion Model with:
                 - $(omp.p.n) agents
                 - $(omp.p.M) media outlets
                 - $(omp.p.L) influencers
                 """)
end

# iteration for destructuring into components
Base.iterate(omp::OpinionModelProblem) = (omp.X, Val(:M))
Base.iterate(omp::OpinionModelProblem, ::Val{:M}) = (omp.Y, Val(:I))
Base.iterate(omp::OpinionModelProblem, ::Val{:I}) = (omp.Z, Val(:A))
Base.iterate(omp::OpinionModelProblem, ::Val{:A}) = (omp.A, Val(:B))
Base.iterate(omp::OpinionModelProblem, ::Val{:B}) = (omp.B, Val(:C))
Base.iterate(omp::OpinionModelProblem, ::Val{:C}) = (omp.C, Val(:done))
Base.iterate(::OpinionModelProblem, ::Val{:done}) = nothing

# function OpinionModelProblem(dom::Vararg{Tuple{Real,Real},D}; p=ModelParams(),
#                              seed=MersenneTwister(),
#                              AgAgNetF::Function=I -> trues(p.n, p.n),) where {D<:Real}
#     @info "Promoting elements of domain tuples"
#     throw(ErrorException("Mixed-type tuples are not yet supported"))
#     # return OpinionModelProblem(promote(dom...), seed, AgAgNetF)
# end

function OpinionModelProblem(dom::Vararg{Tuple{T,T},D}; p=ModelParams(),
                             AgAgNetF::Function=I -> fullyconnected_network(p.n),
                             seed=MersenneTwister()) where {D,T<:AbstractFloat}
    # We divide the domain into orthants, and each orthant has 1 influencer
    p.L != 2^D && throw(ArgumentError("Number of influencers has to be 2^dim"))

    # Seeding the RNG
    Random.seed!(seed)

    # Place agents uniformly distributed across the domain
    X = reduce(hcat, [rand(Uniform(t...), p.n) for t in dom]) # p.n × N matrix

    # Create Agent-Influence network (n × L) by grouping individuals into quadrants
    # i,j-th entry is true if i-th agent follows the j-th influencer
    C = _orthantize(X) |> BitMatrix

    # Placing the influencers as the barycenter of agents per orthant
    I = _place_influencers(X, C)

    if D == 1
        X = vec(X)
        I = vec(I)
    end

    return OpinionModelProblem{T,D}(X, I, C; p=p, dom=dom, AgAgNetF=AgAgNetF)
end

function OpinionModelProblem{T,D}(X₀::AbstractArray{T},
                                  Z₀::AbstractArray{T},
                                  C₀::AdjMatrix{T};
                                  p=ModelParams(; L=size(Z₀, 1), n=size(X₀, 1)),
                                  AgAgNetF::Function=I -> fullyconnected_network(p.n),
                                  dom::NTuple{D,Tuple{T,T}}=_array_bounds(X₀)) where {D,
                                                                                      T<:AbstractFloat}
    p.L != size(Z₀, 1) &&
        throw(ArgumentError("`influencers_init` defined more influencers than contemplated" *
                            "in the parameters $(p)"))

    p.n != size(X₀, 1) &&
        throw(ArgumentError("`agents_init` defined more agents than contemplated in the" *
                            "parameters $(p)"))

    # Create Agent-Influence network (n × L) by grouping individuals into quadrants
    # i,j-th entry is true if i-th agent follows the j-th influencer
    # C = _orthantize(X₀) |> BitMatrix

    # Defining the Agent-Agent interaction matrix as a function of the Agent-Influencer
    # matrix. In the default case, the matrix represents a fully connected network. In other
    # cases, the adjacency is computed with the adjacency to influencers.
    A = AgAgNetF(C₀)

    # Assign agents to media outlet randomly s.t. every agent is connected to 1 and only 1 media.
    B = _media_network(p.n, p.M)

    # We consider just 2 media outlets at the "corners"
    Y = vcat(fill(-one(T), (1, D)), fill(one(T), (1, D)))

    return OpinionModelProblem{T,D}(p, dom, X₀, Y, Z₀, A, B, C₀)
end

# Supporting structs for the SciML DiffEq based solver

abstract type AbstractSolver end

"""
Defines a Type that parametrizes [`ModelSimulation`](@ref) and signifies the simulation
was computed the the "bespoke" (i.e. hand-implemented, modified) Euler--Maruyama. It has a
single property: the timesteps used in the E-M integration procedure.
"""
struct BespokeSolver <: AbstractSolver
    tstops::AbstractVector{Float64}
end

"""
Defines a Type that parametrizes [`ModelSimulation`](@ref) and signifies the opinion
problem was simulated by integrating the SDEs defining the model via
DifferentialEquations.jl. It has a single property: `sol`, the solution of the SDE as a
DifferentialEquations.jl `SciMLBase.RODESolution`.
"""
struct DiffEqSolver <: AbstractSolver
    sol::SciMLBase.RODESolution
    # abstol::AbstractFloat # tolerance value of the diff.eq. solver. Used to compare
    # TODO: Check if I can put the `sol` from DiffEq here to get access to the solution
    # interpolator to make comparisons at the exact same timepoints.
end

"""
Defines a full simulation of a `D`-dimensional OpinionDynamicsProblem simulated with
either [`BespokeSolver`](@ref) or [`DiffEqSolver`](@ref).

## Fields
- `p`: The [`ModelParams`](@ref) of the simulated model.
- `dom`: The domain of opinion space expressed as tuples of bounds, one per dimension.
- `nsteps`: The total number of steps the simulation took.
- `X`: A tensor containing the coordinates of agents at each time step.
- `Y`: Tensor with media coordinates over time.
- `Z`: Tensor with influencer coordinates over time.
- `C`: Tensor of Agent-Influencer adjacency matrices stacked.
- `R`: Computed jumping rates used for influencer switching
"""
mutable struct ModelSimulation{T<:AbstractFloat,D,S<:AbstractSolver}
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

# TODO: Make some Type aliases to make code less obnoxious (e.g. isless)

# Implement destructuring via iteration
Base.iterate(oms::ModelSimulation) = (oms.X, Val(:Y))
Base.iterate(oms::ModelSimulation, ::Val{:Y}) = (oms.Y, Val(:Z))
Base.iterate(oms::ModelSimulation, ::Val{:Z}) = (oms.Z, Val(:C))
Base.iterate(oms::ModelSimulation, ::Val{:C}) = (oms.C, Val(:R))
Base.iterate(oms::ModelSimulation, ::Val{:R}) = (oms.R, Val(:done))
Base.iterate(::ModelSimulation, ::Val{:done}) = nothing

Base.length(oms::ModelSimulation) = size(oms.X, 3)
Base.eltype(::ModelSimulation{T,D,S}) where {T,D,S} = T
solvtype(::ModelSimulation{T,D,S}) where {T,D,S} = S

function ModelSimulation{T,D,DiffEqSolver}(sol::S,
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

    return ModelSimulation{T,D,DiffEqSolver}(p, dom, length(sol.t), solver_meta, X,
                                             Y, Z, C, R)
end

"""
    interpolate!(oms::ModelSimulation{T, D, DiffEqSolver}, tstops) where {T, D}

Mutates the simulation `oms` to re-sample the timeseries at the explcitly given `tstops`
instead of the points where the DiffEq integrator saved by exploiting the enclosed SciML
ODESolution.
"""
function interpolate!(oms::ModelSimulation{T,D,DiffEqSolver}, tstops) where {T,D}
    U = oms.solver.sol.(tstops) |> stack
    p = oms.p

    agents = CartesianIndices((firstindex(U):(p.n), axes(U, 2), axes(U, 3)))
    media = CartesianIndices(((p.n + 1):(p.n + p.M), axes(U, 2), axes(U, 3)))
    influencers = CartesianIndices(((p.n + p.M + 1):(p.n + p.M + p.L), axes(U, 2),
                                    axes(U, 3)))

    oms.X = U[agents]
    oms.Y = U[media]
    oms.Z = U[influencers]

    @assert length(oms) == length(tstops)
end

function Base.show(io::IO, oms::ModelSimulation)
    return print("""
                 Simulation of the ABM Opinion Model with:
                 - $(oms.p.n) agents
                 - $(oms.nsteps) time steps
                 - Solved with: $(solvtype(oms))
                 """)
end

# Functions for comparing solutions

function Base.:-(s1::SBe,
                 s2::SD) where {T,D,SBe<:ModelSimulation{T,D,BespokeSolver},
                                SD<:ModelSimulation{T,D,DiffEqSolver}}
    # FIXME: Use `interpolate!`
    interpolated_sol = s2.solver.sol.(s1.solver.tstops) |> stack
    stacked_sol = vcat(s1.X, s1.Y, s1.Z)

    return stacked_sol .- interpolated_sol
end

function Base.:-(s1::SD,
                 s2::SBe) where {T,D,SBe<:ModelSimulation{T,D,BespokeSolver},
                                 SD<:ModelSimulation{T,D,DiffEqSolver}}
    interpolated_sol = s1.solver.sol.(s2.solver.tstops) |> stack
    stacked_sol = vcat(s2.X, s2.Y, s2.Z)

    return interpolated_sol .- stacked_sol
end

# FIXME: Make these operators commutative, if not already so.

function Base.isapprox(s1::Sim,
                       s2::Sim;
                       rtol::Real=atol > 0 ? 0 : √eps(T),
                       atol::Real=0) where {T,D,Sim<:ModelSimulation{T,D,BespokeSolver}}
    # Check maximum elementwise differences are below the tolerance
    ΔX = s1.X .- s2.X
    ΔY = s1.Y .- s2.Y
    ΔZ = s1.Z .- s2.Z

    return arrays_areapprox(ΔX, ΔY, ΔZ, atol, rtol)
end

function Base.isapprox(s1::SBe,
                       s2::SD;
                       rtol::Real=atol > 0 ? 0 : √eps(T),
                       atol::Real=0) where {T,D,SBe<:ModelSimulation{T,D,BespokeSolver},
                                            SD<:ModelSimulation{T,D,DiffEqSolver}}
    return Δ_isapprox(s1 - s2, atol, rtol)
end
