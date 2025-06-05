module TestSDEFunctions
using Test, OpinionDynamicsABM, ReferenceTests
using SparseArrays

@testset "Force & Drift functions" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)
    X, Y, Z, A, B, C = omp
    a, b, c = omp.p.a, omp.p.b, omp.p.c

    # Testing the Agent-Agent attraction
    agent_force = OpinionDynamicsABM.AgAg_attraction(X, A)
    @test_reference "reftest-files/ag_ag_forces.npz" agent_force by = isapprox

    agent_force_sparse = AgAg_attraction(X, sparse(Float64.(A)))
    @test_reference "reftest-files/ag_ag_forces.npz" agent_force_sparse by = isapprox

    media_force = MedAg_attraction(X, Y, B)
    @test_reference "reftest-files/me_ag_forces.npz" media_force by = isapprox

    influ_force = InfAg_attraction(X, Z, C)
    @test_reference "reftest-files/if_ag_forces.npz" influ_force by = isapprox

    agent_drift = OpinionDynamicsABM.agent_drift(X, Y, Z, A, B, C, a, b, c)
    @test_reference "reftest-files/agent_drift.npz" agent_drift by = isapprox

    media_drift = OpinionDynamicsABM.media_drift(X, Y, B, omp.p.Γ)
    @test_reference "reftest-files/media_drift.npz" media_drift by = isapprox

    infl_drift = OpinionDynamicsABM.influencer_drift(X, Z, C, omp.p.γ)
    @test_reference "reftest-files/infl_drift.npz" infl_drift by = isapprox

    follower_rates = OpinionDynamicsABM.influencer_switch_rates(X, Z, B, C, omp.p.η)
    @test_reference "reftest-files/follw_rates.npz" follower_rates by = isapprox
end

@testset "Legacy force functions" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)
    X, Y, Z, A, B, C = omp
    a, b, c = omp.p.a, omp.p.b, omp.p.c

    # Test against reference in modern implementation
    agent_force = OpinionDynamicsABM.legacy_attraction(X, A)
    @test_reference "reftest-files/ag_ag_forces.npz" agent_force by = isapprox

    follower_rates = OpinionDynamicsABM.legacy_rates(B, X, C, Z, omp.p.η)
    @test_reference "reftest-files/follw_rates.npz" follower_rates by = isapprox
end

@testset "Force functions on non-full networks" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed,
                              AgAgNetF=OpinionDynamicsABM._ag_ag_echo_chamber)
    X, Y, Z, A, B, C = omp
    a, b, c = omp.p.a, omp.p.b, omp.p.c

    # Only testing things that depend on the non-full network
    agent_force = AgAg_attraction(X, A)
    @test_reference "reftest-files/nonfull/ag_ag_forces.npz" agent_force by = isapprox

    agent_drift = OpinionDynamicsABM.agent_drift(X, Y, Z, A, B, C, a, b, c)
    @test_reference "reftest-files/nonfull/agent_drift.npz" agent_drift by = isapprox

    # Testing agreement with legacy implementations (not really crucial)
    old_agent_force = OpinionDynamicsABM.legacy_attraction(X, A)
    @test_reference "reftest-files/nonfull/ag_ag_forces.npz" old_agent_force by = isapprox

    @testset "Hand-computed example on block network" begin
        X = [-1 1.0; 0.0 1.0; 1.0 1.0; -1 -1; 0 -1; 1 -1]
        A = [trues(3, 3) falses(3, 3); falses(3, 3) trues(3, 3)]
        # To be used when testing the in-place implementation
        ref_Dijd = zeros(6, 6, 2)
        dist_block = [0 1 2; -1 0 1; -2 -1 0]
        ref_Dijd[1:3, 1:3, 1] .= dist_block
        ref_Dijd[4:6, 4:6, 1] .= dist_block
        ref_Wij = zeros(6, 6)
        w_block = [1 exp(-1) exp(-2); exp(-1) 1 exp(-1); exp(-2) exp(-1) 1]
        normalizers = [1+exp(-1)+exp(-2); 1+2*exp(-1); 1+exp(-1)+exp(-2)]
        w_block .= w_block ./ normalizers
        ref_Wij[1:3, 1:3] .= w_block
        ref_Wij[4:6, 4:6] .= w_block
        ref_Force = zeros(6, 2)
        c = inv(1 + exp(-1) + exp(-2))
        k = c * exp(-1) + 2 * c * exp(-2)
        ref_Force[:, 1] = [k, 0, -k, k, 0, -k]

        ##
        I, D = size(X)
        J = size(X, 1)
        Fid = similar(X)
        Dijd = zeros(I, J, D)
        Wij = zeros(I, J)

        AgAg_attraction!(Fid, Dijd, Wij, X, A)
        @test Fid == ref_Force
        @test Dijd == ref_Dijd
        @test Wij == ref_Wij


        ## Testing with networks as sparse arrays
        sA = blockdiag(sparse(ones(3, 3)), sparse(ones(3,3)))
        Fid = similar(X)
        Dijd = zeros(I, J, D)
        Wij = zeros(I, J)
        AgAg_attraction!(Fid, Dijd, Wij, X, sA)
        @test Fid == ref_Force
        @test Dijd == ref_Dijd
        @test Wij == ref_Wij
    end

    @testset "Networks with lonely agents" begin
        # FIXME: Add tests
    end
end
end
