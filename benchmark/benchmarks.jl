using BenchmarkTools, OpinionDynamicsABM
const SUITE = BenchmarkGroup()

# Create hierarchy of benchmarks:
SUITE["abm_simulation"] = BenchmarkGroup()

omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
prob = OpinionDynamicsABM.build_sdeproblem(omp, (0.0, 2.0))

SUITE["abm_simulation"]["agent-agent attraction"] = @benchmarkable begin
    setup = (X = copy(omp.X); A = copy(omp.A))
    AgAg_attraction(X, A)
end

SUITE["amb_simulation"]["influencer switch rates"] = @benchmarkable begin
    setup = (Ril = similar(omp.X, omp.p.n, omp.p.L);
             X = copy(omp.X);
             Z = copy(omp.Z);
             B = copy(omp.B);
             C = copy(omp.C);
             η = omp.p.η)
    OpinionDynamicsABM.influencer_switch_rates!(Ril, X, Z, B, C, η)
end

SUITE["abm_simulation"]["full simulation"] = @benchmarkable begin
    setup = (abm = deepcopy(omp))
    simulate!(abm)
end

# SUITE["abm_simulation"]["diffeq_integrator"] = @benchmarkable(
#     simulate!($prob)
# )
