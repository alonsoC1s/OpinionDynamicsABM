using BenchmarkTools, OpinionDynamicsABM
const SUITE = BenchmarkGroup()

# Create hierarchy of benchmarks:
SUITE["abm_simulation"] = BenchmarkGroup()

omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
prob = OpinionDynamicsABM.build_sdeproblem(omp, (0.0, 2.0))
# X, M, I, A, Z, Y = OpinionDynamicsABM.get_values(omp)
# X, M, I, A, Z, Y = omp
SUITE["abm_simulation"]["agent-agent attraction"] = @benchmarkable AgAg_attraction(X, A) setup = (X=copy(omp.X);
                                                                                                  A=copy(omp.A))

SUITE["abm_simulation"]["full simulation"] = @benchmarkable simulate!(abm) setup = (abm = deepcopy(omp))

# SUITE["abm_simulation"]["diffeq_integrator"] = @benchmarkable(
#     simulate!($prob)
# )
