using OpinionDynamicsABM
using Test

module TestUtils

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
end # TestUtils