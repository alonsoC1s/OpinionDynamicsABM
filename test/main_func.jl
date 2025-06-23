module TestOpinionDynamics
using Test, ReferenceTests, OpinionDynamicsABM

@testset "Params and Problem initializers" begin
    N = 250 # Number of agents
    # @testset "1-dimensional problem initializers" begin
    #     X = rand(N)
    #     M = [-1.0, 1.0]
    #     # Creating from initial positions of agents and media
    #     @test_nowarn OpinionModelProblem(X, M)
    #     # Creating from tuples with the same number types
    #     @test_nowarn OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
    #     # Creating from tuples with mixed number types
    #     @test_throws MethodError OpinionModelProblem((-2, 2.0), (-2.0, 2.0))

    #     # TODO: Test the other keyword arguments of the constructors

    # end
end

@testset "Test simulations" begin
    @testset "Default parameters" begin
        fixed_seed = 210624
        deterministic_params = ModelParams(; σ=0.0, σ̂=0.0, σ̃=0.0)
        omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed,
                                  p=deterministic_params)

        # Testing the full simulation
        X, Y, Z, C, R = simulate!(omp; seed=fixed_seed)

        # Testing agent's positions
        @test_reference "reftest-files/sim/X.npz" X by = isapprox
        @test_reference "reftest-files/sim/Y.npz" Y by = isapprox
        @test_reference "reftest-files/sim/Z.npz" Z by = isapprox
        @test_reference "reftest-files/sim/C.npz" C by = isapprox
        @test_reference "reftest-files/sim/R.npz" R by = isapprox
    end

    @testset "No media" begin
        fixed_seed = 210624
        no_media_p = ModelParams(; b=0, σ=0.0, σ̂=0.0, σ̃=0.0)
        omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed,
                                  p=no_media_p)

        # Testing the full simulation
        X, Y, Z, C, R = simulate!(omp; seed=fixed_seed)

        # Testing agent's positions
        @test_reference "reftest-files/no_media/X.npz" X by = isapprox
        @test_reference "reftest-files/no_media/Y.npz" Y by = isapprox
        @test_reference "reftest-files/no_media/Z.npz" Z by = isapprox
        @test_reference "reftest-files/no_media/C.npz" C by = isapprox
        @test_reference "reftest-files/no_media/R.npz" R by = isapprox
    end
end

end # TestOpinionDynamics
