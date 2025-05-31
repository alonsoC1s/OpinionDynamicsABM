module TestSDEFunctions
using Test, OpinionDynamicsABM, ReferenceTests

@testset "Force & Drift functions" begin
    fixed_seed = 210624
    omp = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0); seed=fixed_seed)
    X, Y, Z, A, B, C = omp
    a, b, c = omp.p.a, omp.p.b, omp.p.c

    # Testing the Agent-Agent attraction
    agent_force = OpinionDynamicsABM.AgAg_attraction(X, A)
    @test_reference "reftest-files/ag_ag_forces.npz" agent_force by = isapprox

    media_force = OpinionDynamicsABM.MedAg_attraction(X, Y, B)
    @test_reference "reftest-files/me_ag_forces.npz" media_force by = isapprox

    influ_force = OpinionDynamicsABM.InfAg_attraction(X, Z, C)
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
    agent_force = OpinionDynamicsABM.AgAg_attraction(X, A)
    @test_reference "reftest-files/nonfull/ag_ag_forces.npz" agent_force by = isapprox

    agent_drift = OpinionDynamicsABM.agent_drift(X, Y, Z, A, B, C, a, b, c)
    @test_reference "reftest-files/nonfull/agent_drift.npz" agent_drift by = isapprox

    # Testing agreement with legacy implementations (not really crucial)
    old_agent_force = OpinionDynamicsABM.legacy_attraction(X, A)
    @test_reference "reftest-files/nonfull/ag_ag_forces.npz" old_agent_force by = isapprox
end
end
