using IT3708Project3

function usage()
    println("Usage: julia --project=. plot_feature_count.jl <dataset-key|triangle> [epsilon] [output-path]")
    println("")
    println("Examples:")
    println("  julia --project=. plot_feature_count.jl breast-w")
    println("  julia --project=. plot_feature_count.jl breast-w 0.1")
    println("  julia --project=. plot_feature_count.jl triangle")
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "breast-w"
epsilon = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing
metadata = dataset_plot_metadata(dataset_key)

landscape = load_landscape_key(dataset_key)
default_name = if isnothing(epsilon)
    "$(metadata.slug)_feature_count"
else
    epsilon_tag = replace(string(epsilon), "." => "p")
    "$(metadata.slug)_feature_count_e$(epsilon_tag)"
end

output_path = length(ARGS) >= 3 ? ARGS[3] : default_feature_count_plot_path(default_name; dataset_key=dataset_key)
values = isnothing(epsilon) ? fitness_values(landscape) : penalized_fitness_values(landscape, epsilon)
fitness_label = isnothing(epsilon) ? "Fitness" : "Penalized fitness"
title = isnothing(epsilon) ? "$(metadata.label) fitness by feature count" : "$(metadata.label) fitness by feature count (epsilon=$(epsilon))"

saved_path = save_fitness_by_feature_count_plot(
    landscape,
    output_path;
    values=values,
    title=title,
    fitness_label=fitness_label,
)

println("Saved feature-count plot for `$(metadata.label)` to `$saved_path`.")
