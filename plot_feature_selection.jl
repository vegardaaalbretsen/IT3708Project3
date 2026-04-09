using IT3708Project3
using Plots
using Statistics

function print_usage()
    println("Usage: julia --project=. plot_feature_selection.jl [dataset_key] [output]")
    println("")
    println("Defaults:")
    println("  dataset_key = credit-a")
    println("  output = <dataset_key>_landscape.png")
    println("")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "credit-a"
    output = length(args) >= 2 ? args[2] : "$(dataset_key)_landscape.png"

    return dataset_key, output
end

function aggregate_by_size(values, subset_sizes, n_features; reducer = maximum)
    aggregated = Float64[]
    for size in 1:n_features
        mask = subset_sizes .== size
        push!(aggregated, reducer(values[mask]))
    end
    return aggregated
end

function count_by_size(subset_sizes, n_features)
    return [count(==(size), subset_sizes) for size in 1:n_features]
end

function plot_feature_selection(dataset_key::AbstractString, output::AbstractString)
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")

    spec = DATASET_SPECS[dataset_key]
    landscape = read_feature_selection_landscape(spec.path, spec.n_features)
    subset_sizes = count_ones.(landscape.subset_indices)
    sizes = collect(1:spec.n_features)

    best_raw_by_size = aggregate_by_size(landscape.raw_accuracy_table, subset_sizes, spec.n_features)
    best_penalized_by_size = aggregate_by_size(landscape.penalized_table, subset_sizes, spec.n_features)
    mean_time_by_size = aggregate_by_size(landscape.raw_time_table, subset_sizes, spec.n_features; reducer = mean)
    subset_count_by_size = count_by_size(subset_sizes, spec.n_features)

    default(size = (1100, 850), legend = :best)

    p1 = scatter(
        subset_sizes,
        landscape.raw_accuracy_table;
        alpha = 0.15,
        markersize = 3,
        color = :steelblue,
        xlabel = "Active features",
        ylabel = "Raw accuracy",
        title = "Raw accuracy by subset size",
        label = "all subsets",
    )
    plot!(p1, sizes, best_raw_by_size; linewidth = 3, color = :navy, label = "best by size")

    p2 = scatter(
        subset_sizes,
        landscape.penalized_table;
        alpha = 0.15,
        markersize = 3,
        color = :darkgreen,
        xlabel = "Active features",
        ylabel = "Penalized fitness",
        title = "Penalized fitness by subset size",
        label = "all subsets",
    )
    plot!(p2, sizes, best_penalized_by_size; linewidth = 3, color = :forestgreen, label = "best by size")

    p3 = scatter(
        subset_sizes,
        landscape.raw_time_table;
        alpha = 0.15,
        markersize = 3,
        color = :darkorange,
        xlabel = "Active features",
        ylabel = "Mean training time",
        title = "Training time by subset size",
        label = "all subsets",
    )
    plot!(p3, sizes, mean_time_by_size; linewidth = 3, color = :red3, label = "mean by size")

    p4 = bar(
        sizes,
        subset_count_by_size;
        alpha = 0.85,
        color = :purple,
        xlabel = "Active features",
        ylabel = "Number of subsets",
        title = "How many subsets exist at each size",
        label = "subset count",
    )

    combined = plot(p1, p2, p3, p4; layout = (2, 2), plot_title = "Feature-selection landscape: $dataset_key")
    savefig(combined, output)

    println("Saved plot to $(abspath(output))")
    println("Best raw subset: $(best_raw_subset(landscape))")
    println("Best penalized subset: $(best_penalized_subset(landscape))")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, output = parsed
    plot_feature_selection(dataset_key, output)
end
