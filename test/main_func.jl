module TestOpinionDynamics
using Test, OpinionDynamicsABM

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
        @test_throws ErrorException OpinionModelProblem((-2, 2.0),(-2.0, 2.0))

    end
end

@testset "Attraction functions" begin end

@testset "Test simulations" begin end

end # TestOpinionDynamics