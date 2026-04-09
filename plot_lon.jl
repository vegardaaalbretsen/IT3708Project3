using IT3708Project3
using Plots

function print_usage()
    println("Usage: julia --project=. plot_lon.jl [dataset_key] [score_type] [top_edges] [output]")
    println("")
    println("Defaults:")
    println("  dataset_key = breast-w")
    println("  score_type = raw")
    println("  top_edges = 200")
    println("  output = <dataset_key>_<score_type>_lon.png")
    println("")
    println("score_type options: raw, penalized")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "breast-w"
    score_type = length(args) >= 2 ? args[2] : "raw"
    top_edges = length(args) >= 3 ? parse(Int, args[3]) : 200
    output = length(args) >= 4 ? args[4] : "$(dataset_key)_$(score_type)_lon.png"

    return dataset_key, score_type, top_edges, output
end

function score_symbol(score_type::AbstractString)
    if score_type == "raw"
        return :raw, "Raw accuracy"
    elseif score_type == "penalized"
        return :penalized, "Penalized fitness"
    end

    error("Unknown score_type: $score_type")
end

function stable_node_positions(node_subset_indices, node_active_counts, node_fitness)
    phase = 2pi .* mod.(node_subset_indices .* 0.61803398875, 1.0)
    x = node_active_counts .+ 0.22 .* sin.(phase)
    y = node_fitness .+ 0.015 .* cos.(phase)
    return x, y
end

function plot_lon(dataset_key::AbstractString, score_type::AbstractString, top_edges::Integer, output::AbstractString)
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")
    top_edges > 0 || error("top_edges must be positive")

    spec = DATASET_SPECS[dataset_key]
    score, ylabel = score_symbol(score_type)
    lon = feature_selection_local_optima_network(spec.path, spec.n_features; score = score)

    x, y = stable_node_positions(lon.node_subset_indices, lon.node_active_counts, lon.node_fitness)
    node_sizes = 5 .+ 6 .* log10.(lon.basin_sizes .+ 1)
    best_node = argmax(lon.node_fitness)
    fitness_gradient = cgrad([:deeppink, :lightpink, :khaki, :yellowgreen, :forestgreen])

    n_edges = min(top_edges, length(lon.edges))
    edges_to_plot = lon.edges[1:n_edges]
    max_count = isempty(edges_to_plot) ? 1 : maximum(edge.count for edge in edges_to_plot)
    max_probability = isempty(edges_to_plot) ? 1.0 : maximum(edge.probability for edge in edges_to_plot)

    default(size = (1450, 750))

    p_network = plot(
        xlabel = "Active features in local optimum",
        ylabel = ylabel,
        title = "Local Optima Network: $dataset_key ($score_type)",
        legend = false,
        colorbar = :right,
    )

    for edge in edges_to_plot
        alpha = 0.08 + 0.45 * (edge.probability / max_probability)
        width = 0.3 + 2.7 * (edge.count / max_count)
        plot!(
            p_network,
            [x[edge.source], x[edge.target]],
            [y[edge.source], y[edge.target]];
            color = :gray40,
            alpha = alpha,
            linewidth = width,
            label = false,
        )
    end

    scatter!(
        p_network,
        x,
        y;
        marker_z = lon.node_fitness,
        markersize = node_sizes,
        markerstrokewidth = 0,
        color = fitness_gradient,
        colorbar_title = ylabel,
        label = false,
    )

    scatter!(
        p_network,
        [x[best_node]],
        [y[best_node]];
        markersize = [node_sizes[best_node] + 4],
        markercolor = RGBA(0, 0, 0, 0),
        markerstrokecolor = :red,
        markerstrokewidth = 2.5,
        label = false,
    )

    top_nodes = sortperm(lon.basin_sizes; rev = true)[1:min(12, length(lon.basin_sizes))]
    for node in top_nodes
        annotate!(p_network, x[node], y[node], text(string(lon.node_subset_indices[node]), 8, :black))
    end

    top_basin_sizes = lon.basin_sizes[top_nodes]
    top_labels = string.(lon.node_subset_indices[top_nodes])
    p_basins = bar(
        top_labels,
        top_basin_sizes;
        xlabel = "Local optimum subset index",
        ylabel = "Basin size",
        title = "Largest basins",
        legend = false,
        color = :slateblue,
        xrotation = 45,
    )

    combined = plot(p_network, p_basins; layout = (1, 2))
    savefig(combined, output)

    println("Saved LON plot to $(abspath(output))")
    println("Local optima: $(length(lon.node_subset_indices))")
    println("Directed inter-basin edges: $(length(lon.edges))")
    println("Plotted strongest edges: $n_edges")
    println("Largest basin subset: $(lon.node_subset_indices[top_nodes[1]]) with basin size $(lon.basin_sizes[top_nodes[1]])")
    println("Interpretation:")
    println("  greener nodes are better local optima")
    println("  pinker nodes are worse local optima")
    println("  larger nodes have larger basins")
    println("  red outline marks the best local optimum")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, score_type, top_edges, output = parsed
    plot_lon(dataset_key, score_type, top_edges, output)
end
