using OpinionDynamicsABM
using DifferentialEquations

omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
p = omp.p

P = (L=p.L, M=p.M, n=p.n, η=p.η, a=p.a, b=p.b, c=p.c, σ=p.σ, σ̂=p.σ̂, σ̃=p.σ̃,
     A=omp.AgAgNet, B=omp.AgMedNet, C=omp.AgInfNet, p=p)

u₀ = vcat(omp.X, omp.I, omp.M)

# Simple SDE problem
problem = SDEProblem(drift, noise, u₀, (0.0, 2.0), P)
# Introducing the callback
condition(u, t, integrator) = true
function affect!(integrator)
    dt = get_proposed_dt(integrator)
    res = influencer_callback(integrator.u, integrator.p, dt)
    @show res
    integrator.p.C .= influencer_callback(integrator.u, integrator.p, dt)
end
cb = DiscreteCallback(condition, affect!)

ssol = solve(problem, SRIW1(), callback = cb, seed = seed)

# # Ensemble simulations
# ensembleprob = EnsembleProblem(problem)
# esol = solve(ensembleprob, SRIW1(), EnsembleThreads(); trajectories=100)