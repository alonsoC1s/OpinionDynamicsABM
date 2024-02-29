module TestUtils

using OpinionDynamicsABM, Test, Random

# include("../../Numerics4/src/abm.jl")

import
    OpinionDynamicsABM._orthantize,
    OpinionDynamicsABM._place_influencers,
    OpinionDynamicsABM._media_network,
    OpinionDynamicsABM.influencer_switch_rates,
    OpinionDynamicsABM.legacy_rates,
    OpinionDynamicsABM.legacy_changeinfluencer,
    OpinionDynamicsABM.legacy_attraction,
    OpinionDynamicsABM.legacy_influence,
    OpinionDynamicsABM.switch_influencer

@testset "Tests for auxiliary functions" begin
    ## 1-d test
    ## Agent orthontization
    X = [-1, 1]
    S_expected = [false true; true false]
    S = _orthantize(X)

    @test S == S_expected

    ## Correct influencer placement
    I_expected = [-1; 1]
    I = _place_influencers(X, S)

    # @test I == I_expected

    ## 2-d test
    ## 1 agent per quadrant in ascending clockwise numbering
    X = [
        1 1;
        -1 1;
        -1 -1;
        1 -1
    ]

    S_expected = [
        # Follows the numbering sequence 1-3-4-2 (clockwise)
        true false false false;
        false false true false;
        false false false true;
        false true false false
    ]
    
    S = _orthantize(X)

    @test S == S_expected

    ## Correct influencer placing
    I_expected = [
        1 1;
        1 -1;
        -1 1
        -1 -1;
    ]

    I = _place_influencers(X, S)

    @test I == I_expected

    # 3-d test
    # 1 agent per octant to reproduce sequence 1:8
    X = [
        1 1 1;
        1 1 -1;
        1 -1 1;
        1 -1 -1;
        -1 1 1;
        -1 1 -1;
        -1 -1 1;
        -1 -1 -1;
    ]

    S_expected = Bool[
        1 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0;
        0 0 0 1 0 0 0 0;
        0 0 0 0 1 0 0 0;
        0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 1;
    ]

    S = _orthantize(X)

    @test S == S_expected

    ## Correct placement of influencers
    I_expected = [
        1 1 1;
        1 1 -1;
        1 -1 1;
        1 -1 -1;
        -1 1 1;
        -1 1 -1;
        -1 -1 1;
        -1 -1 -1;
    ]

    I = _place_influencers(X, S)

    @test I == I_expected
    # TODO: Add one more randomized test

    # Media network
    # Check if every single row has at least 1 true by summing along rows, and converting to bools
    # FIXME: Use `findall` & check size instead of relying on all(sum(⋅))
    M = _media_network(1000, 2)
    @test all(sum(M; dims=2) |> BitMatrix)

    M = _media_network(1000, 4)
    @test all(sum(M; dims=2) |> BitMatrix)
end

# @testset "Main functionality" begin
#    o = OpinionModelProblem((-2, 2), (-2, 2))
#    a, b, c, n, L = o.p.a, o.p.b, o.p.c, o.p.n, o.p.L
#    X, Y, Z = o.X, o.M, o.I
#    A, B, C = o.AgAgNet, o.AgMedNet, o.AgInfNet

#    # Testing model violation detections
#    invalid_network = [true false; false true; false false] |> BitMatrix
#    @test_throws ErrorException MedAg_attraction(X, Y, invalid_network)
#    @test_throws ErrorException InfAg_attraction(X, Z, invalid_network)

#    # Testing agent-agent attraction vs. Luzie's version
#    @test legacy_attraction(o.X, o.AgAgNet) == AgAg_attraction(o.X, o.AgAgNet)

#    # Convert media adj-matrix to a {-1, 1} vector representation expected by legacy code
#    state = (findfirst.(eachrow(B)) .== 2) |> Vector
#    @test legacy_influence(X, Y, Z, C, state, (0, (n=n, b=b, c=c, L=L))) == c * MedAg_attraction(X, Y, B) +
#         b * InfAg_attraction(X, Z, C)

#     # Testing influencer switch rates
#     @test influencer_switch_rates(X, Z, B, C, 15) ≈ legacy_rates(B, X, C, Z, 15)

#     # Testing agent swtiching, re-seeding the RNG
#     Random.seed!(200923)
#     U = switch_influencer(C, X, Z, B, 15.0, 0.01, method = :other)
#     Random.seed!(200923)
#     U_l = switch_influencer(C, X, Z, B, 15.0, 0.01, method = :luzie)
#     Random.seed!(200923)
#     V = legacy_changeinfluencer(B, X, C, Z, 15.0, 0.01)

#     # Congruency between two versions in the same method
#     @test U == U_l
#     # Testing vs. full Luzie code
#     @test U_l == V

#     # Testing agent switching after 2 steps
#     Random.seed!(200923)
#     U = switch_influencer(C, X, Z, B, 15.0, 0.01, method = :other)
#     U2 = switch_influencer(U, X, Z, B, 15.0, 0.01, method = :other)
#     Random.seed!(200923)
#     U_l = switch_influencer(C, X, Z, B, 15.0, 0.01, method = :luzie)
#     U2_l = switch_influencer(U_l, X, Z, B, 15.0, 0.01, method = :luzie)
#     Random.seed!(200923)
#     V = legacy_changeinfluencer(B, X, C, Z, 15.0, 0.01)
#     V2 = legacy_changeinfluencer(B, X, V, Z, 15.0, 0.01)

#     # Congruency between two versions in the same method
#     @test U2 == U2_l
#     # Testing vs. full Luzie code
#     @test U2_l == V2
# end
end