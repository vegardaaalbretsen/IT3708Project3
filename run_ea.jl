using IT3708Project3
using Random

function usage()
    println("Usage: julia --project=. run_ea.jl <dataset-key|triangle> [iterations] [epsilon] [seed] [initial-index]")
    println("")
    println("Examples:")
    println("  julia --project=. run_ea.jl breast-w")
    println("  julia --project=. run_ea.jl breast-w 10000 0.01")
    println("  julia --project=. run_ea.jl triangle 5000 0.0 42 0")
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "breast-w"
iterations = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
epsilon = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
rng = length(ARGS) >= 4 ? MersenneTwister(parse(Int, ARGS[4])) : Random.default_rng()
initial_index = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing

landscape = load_landscape_key(dataset_key)
result = run_standard_ea(
    landscape;
    iterations=iterations,
    epsilon=epsilon,
    rng=rng,
    initial_index=initial_index,
)

println("Standard EA on `$(landscape.name)`")
println("Iterations: $(result.iterations)")
println("Epsilon: $(result.epsilon)")
println("Accepted moves: $(result.accepted_moves)")
println("Initial: index=$(result.initial_index), features=$(result.initial_num_selected), accuracy=$(result.initial_accuracy), penalized=$(result.initial_penalized_fitness)")
println("Final:   index=$(result.final_index), features=$(result.final_num_selected), accuracy=$(result.final_accuracy), penalized=$(result.final_penalized_fitness)")
println("Best:    index=$(result.best_index), features=$(result.best_num_selected), accuracy=$(result.best_accuracy), penalized=$(result.best_penalized_fitness)")
