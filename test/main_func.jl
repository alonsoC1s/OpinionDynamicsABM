module TestOpinionDynamics
using Test, ReferenceTests, OpinionDynamicsABM

@testset "Params and Problem initializers" begin
    N = 250 # Number of agents
    @testset "1-dimensional problem initializers" begin
        X = rand(N)
        M = [-1.0, 1.0]
        # Creating from initial positions of agents and media
        @test_nowarn OpinionModelProblem(X, M)
        # Creating from tuples with the same number types
        @test_nowarn OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
        # Creating from tuples with mixed number types
        @test_throws MethodError OpinionModelProblem((-2, 2.0), (-2.0, 2.0))

        # TODO: Test the other keyword arguments of the constructors

    end
end

@testset "Test simulations" begin
    fixed_seed = 210624

    comp(d1, d2) = keys(d1) == keys(d2) &&
                   all([v1 ≈ v2 for (v1, v2) in zip(values(d1), values(d2))])

    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)

    # Testing the Agent-Agent attraction
    force = OpinionDynamicsABM.AgAg_attraction(omp.X, omp.AgAgNet)
    @test_reference "reftest-files/ag_ag_forces.npz" force by=isapprox


    # Testing the full simulation
    X, Y, Z, _, R = simulate!(omp; seed=fixed_seed)

    # Testing agent's positions
    @test_reference "reftest-files/X.npz" X by=isapprox
    @test_reference "reftest-files/Y.npz" Y by=isapprox
    @test_reference "reftest-files/Z.npz" Z by=isapprox
    @test_reference "reftest-files/R.npz" R by=isapprox
end

end # TestOpinionDynamics
