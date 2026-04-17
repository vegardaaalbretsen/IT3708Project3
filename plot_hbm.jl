using IT3708Project3

function usage()
    println("Usage: julia --project=. plot_hbm.jl <dataset-key|triangle> [epsilon] [output-path]")
    println("")
    println("Examples:")
    println("  julia --project=. plot_hbm.jl breast-w")
    println("  julia --project=. plot_hbm.jl breast-w 0.1")
    println("  julia --project=. plot_hbm.jl triangle")
    println("  julia --project=. plot_hbm.jl breast-w 0.1 exports/plots/hbm/breast-w_penalized_e0p1_hbm.png")
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "breast-w"
epsilon = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing

default_name = if isnothing(epsilon)
    dataset_key
else
    epsilon_tag = replace(string(epsilon), "." => "p")
    "$(dataset_key)_penalized_e$(epsilon_tag)"
end

output_path = length(ARGS) >= 3 ? ARGS[3] : default_hbm_plot_path(default_name)

landscape = load_landscape_key(dataset_key)
values = if isnothing(epsilon)
    fitness_values(landscape)
else
    penalized_fitness_values(landscape, epsilon)
end

fitness_label = isnothing(epsilon) ? "Accuracy" : "Penalized fitness"
title = isnothing(epsilon) ? "$(landscape.name) HBM" : "$(landscape.name) HBM (epsilon=$(epsilon))"

saved_path = save_hbm_plot(landscape, output_path; values=values, title=title, fitness_label=fitness_label)

println("Saved HBM plot for `$(landscape.name)` to `$saved_path`.")
