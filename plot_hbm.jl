using IT3708Project3

function usage()
    println("Usage: julia --project=. plot_hbm.jl <dataset-key> [epsilon] [output-path]")
    println("")
    println("Examples:")
    println("  julia --project=. plot_hbm.jl breast-w")
    println("  julia --project=. plot_hbm.jl breast-w 0.1")
    println("  julia --project=. plot_hbm.jl breast-w 0.1 exports/plots/hbm/breast-w_penalized_e0p1_hbm.png")
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "breast-w"
haskey(DATASETS, dataset_key) || error("Unknown dataset key: $dataset_key")

epsilon = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing

default_name = if isnothing(epsilon)
    dataset_key
else
    epsilon_tag = replace(string(epsilon), "." => "p")
    "$(dataset_key)_penalized_e$(epsilon_tag)"
end

output_path = length(ARGS) >= 3 ? ARGS[3] : default_hbm_plot_path(default_name)

dataset = DATASETS[dataset_key]
csv_path = default_output_path(dataset_key)

if !isfile(csv_path)
    parsed = parse_dataset(dataset.path, dataset.num_features)
    write_csv(parsed, csv_path)
end

landscape = load_landscape(csv_path)
fitness_values = if isnothing(epsilon)
    landscape.mean_accuracy
else
    apply_penalty(landscape, epsilon).fitness
end

fitness_label = isnothing(epsilon) ? "Accuracy" : "Penalized fitness"
title = isnothing(epsilon) ? "$(dataset_key) HBM" : "$(dataset_key) HBM (epsilon=$(epsilon))"

nodes = build_hbm(landscape.indices, fitness_values, dataset.num_features)
saved_path = save_hbm_plot(nodes, dataset.num_features, output_path; title=title, fitness_label=fitness_label)

println("Saved HBM plot for `$dataset_key` to `$saved_path`.")
