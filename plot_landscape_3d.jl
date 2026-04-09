using IT3708Project3
using Plots

function print_usage()
    println("Usage: julia --project=. plot_landscape_3d.jl [dataset_key] [score_type] [output]")
    println("")
    println("Defaults:")
    println("  dataset_key = credit-a")
    println("  score_type = penalized")
    println("  output = <dataset_key>_<score_type>_3d.png")
    println("")
    println("score_type options: raw, penalized")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "credit-a"
    score_type = length(args) >= 2 ? args[2] : "penalized"
    output = length(args) >= 3 ? args[3] : "$(dataset_key)_$(score_type)_3d.png"

    return dataset_key, score_type, output
end

function score_config(landscape, score_type::AbstractString)
    if score_type == "raw"
        return (
            zvalues = landscape.raw_accuracy_table,
            zlabel = "Raw accuracy",
            best = best_raw_subset(landscape),
            best_value = best_raw_subset(landscape).raw_accuracy,
        )
    elseif score_type == "penalized"
        return (
            zvalues = landscape.penalized_table,
            zlabel = "Penalized fitness",
            best = best_penalized_subset(landscape),
            best_value = best_penalized_subset(landscape).penalized_fitness,
        )
    end

    error("Unknown score_type: $score_type")
end

function plot_landscape_3d(dataset_key::AbstractString, score_type::AbstractString, output::AbstractString)
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")

    spec = DATASET_SPECS[dataset_key]
    landscape = read_feature_selection_landscape(spec.path, spec.n_features)
    subset_sizes = count_ones.(landscape.subset_indices)
    config = score_config(landscape, score_type)

    default(size = (1050, 850), legend = :topright)

    p = scatter3d(
        subset_sizes,
        landscape.raw_time_table,
        config.zvalues;
        marker_z = config.zvalues,
        markersize = 3,
        markerstrokewidth = 0,
        alpha = 0.45,
        colorbar_title = config.zlabel,
        xlabel = "Active features",
        ylabel = "Mean training time",
        zlabel = config.zlabel,
        title = "3D feature-selection landscape: $dataset_key",
        label = "all subsets",
        camera = (45, 25),
    )

    scatter3d!(
        p,
        [config.best.n_active],
        [config.best.mean_time],
        [config.best_value];
        markersize = 9,
        marker = :diamond,
        color = :red,
        markerstrokewidth = 0,
        label = "best subset $(config.best.subset_index)",
    )

    savefig(p, output)

    println("Saved 3D plot to $(abspath(output))")
    println("Highlighted subset: $(config.best)")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, score_type, output = parsed
    plot_landscape_3d(dataset_key, score_type, output)
end
