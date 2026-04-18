using IT3708Project3
using Random

function usage()
    println("Usage: julia --project=. plot_ea.jl <dataset-key|triangle> [iterations] [epsilon] [seed] [initial-index] [output-path]")
    println("")
    println("Examples:")
    println("  julia --project=. plot_ea.jl breast-w")
    println("  julia --project=. plot_ea.jl breast-w 10000 0.01 42")
    println("  julia --project=. plot_ea.jl triangle 5000 0.1 42 0")
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "breast-w"
iterations = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
epsilon = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
initial_index = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing

default_name = if epsilon == 0
    "$(dataset_key)_ea_trace"
else
    epsilon_tag = replace(string(epsilon), "." => "p")
    "$(dataset_key)_ea_trace_e$(epsilon_tag)"
end

output_path = length(ARGS) >= 6 ? ARGS[6] : default_ea_plot_path(default_name)
rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

landscape = load_landscape_key(dataset_key)
result = run_standard_ea(
    landscape;
    iterations=iterations,
    epsilon=epsilon,
    rng=rng,
    initial_index=initial_index,
    keep_history=true,
)

fitness_label = epsilon == 0 ? "Fitness" : "Penalized fitness"
title = epsilon == 0 ? "$(landscape.name) standard EA trace" : "$(landscape.name) standard EA trace (epsilon=$(epsilon))"
saved_path = save_ea_trace_plot(result, output_path; title=title, fitness_label=fitness_label)

println("Saved EA trace plot for `$(landscape.name)` to `$saved_path`.")
