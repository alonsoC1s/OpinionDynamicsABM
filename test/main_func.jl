module TestOpinionDynamics
using Test, JLD2, ReferenceTests, OpinionDynamicsABM

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
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0), seed = fixed_seed)
    X, Y, Z, _, R = simulate!(omp, seed = fixed_seed)

    comp(d1, d2) = keys(d1)==keys(d2) && all([ v1≈v2 for (v1,v2) in zip(values(d1), values(d2))])

    rsp = Dict("X" => X, "Y" => Y, "Z" => Z, "R" => R)

    # Testing agent's positions
    @test_reference "reftest-files/resp.jld2" rsp by=comp
end

end # TestOpinionDynamics