using OpinionDynamicsABM, Test

my_tests = ["utils.jl", "main_func.jl"]

@info "Running tests"

for test_file in my_tests
    include(test_file)
end
