using OpinionDynamicsABM, Test

my_tests = ["utils.jl", "main_func.jl", "sde_functions.jl"]
# my_tests = ["sde_functions.jl"]

@info "Running tests"

for test_file in my_tests
    include(test_file)
end
