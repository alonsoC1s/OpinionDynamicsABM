module TestUtils
using Test, OpinionDynamicsABM, ReferenceTests

@testset "Legacy force functions" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)

    force = OpinionDynamicsABM.legacy_attraction(omp.X, omp.A)
end

@testset "Force & Drift functions" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)

    # Testing the Agent-Agent attraction
    force = OpinionDynamicsABM.AgAg_attraction(omp.X, omp.A)
    @test_reference "reftest-files/ag_ag_forces.npz" force by = isapprox
end

end
