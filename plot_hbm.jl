using IT3708Project3
using Plots

function print_usage()
    println("Usage: julia --project=. plot_hbm.jl [dataset_key] [score_type] [output]")
    println("")
    println("Defaults:")
    println("  dataset_key = breast-w")
    println("  score_type = penalized")
    println("  output = <dataset_key>_<score_type>_hbm.png")
    println("")
    println("score_type options: raw, penalized")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "breast-w"
    score_type = length(args) >= 2 ? args[2] : "penalized"
    output = length(args) >= 3 ? args[3] : "$(dataset_key)_$(score_type)_hbm.png"

    return dataset_key, score_type, output
end

function score_symbol(score_type::AbstractString)
    if score_type == "raw"
        return :raw, "Raw accuracy"
    elseif score_type == "penalized"
        return :penalized, "Penalized fitness"
    end

    error("Unknown score_type: $score_type")
end

function normalize_marker_sizes(values)
    lo = minimum(values)
    hi = maximum(values)
    if hi <= lo + eps()
        return fill(5.0, length(values))
    end

    return 3 .+ 5 .* ((values .- lo) ./ (hi - lo))
end

function hbm_grid_lines(n_features::Integer)
    split = cld(n_features, 2)
    xlines = [2^k for k in 0:(split - 1)]
    ybits = n_features - split
    ylines = [2^k for k in 0:(max(ybits - 1, -1)) if ybits > 0]
    return xlines, ylines
end

function plot_hbm(dataset_key::AbstractString, score_type::AbstractString, output::AbstractString)
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")

    spec = DATASET_SPECS[dataset_key]
    score, clabel = score_symbol(score_type)
    landscape = read_feature_selection_landscape(spec.path, spec.n_features)
    lon = feature_selection_local_optima_network(spec.path, spec.n_features; score = score)

    values = score == :raw ? landscape.raw_accuracy_table : landscape.penalized_table
    best = score == :raw ? best_raw_subset(landscape) : best_penalized_subset(landscape)

    coords = [hinged_bitstring_coordinates(index, spec.n_features) for index in landscape.subset_indices]
    x = first.(coords)
    y = last.(coords)
    sizes = normalize_marker_sizes(values)
    local_optima = Set(lon.node_subset_indices)
    local_positions = findall(index -> index in local_optima, landscape.subset_indices)
    xlines, ylines = hbm_grid_lines(spec.n_features)

    default(size = (1100, 850), legend = false)

    p = scatter(
        x,
        y;
        marker_z = values,
        markersize = sizes,
        markerstrokewidth = 0,
        color = cgrad([:deeppink, :lightpink, :khaki, :yellowgreen, :forestgreen]),
        colorbar_title = clabel,
        xlabel = "First half as decimal",
        ylabel = "Second half as decimal",
        title = "HBM: $dataset_key ($score_type)",
        aspect_ratio = :equal,
    )

    for xline in xlines
        vline!(p, [xline]; color = :white, alpha = 0.15, linestyle = :dash, linewidth = 1)
    end
    for yline in ylines
        hline!(p, [yline]; color = :white, alpha = 0.15, linestyle = :dash, linewidth = 1)
    end

    if !isempty(local_positions)
        scatter!(
            p,
            x[local_positions],
            y[local_positions];
            markersize = sizes[local_positions] .+ 2,
            markercolor = RGBA(0, 0, 0, 0),
            markerstrokecolor = :dodgerblue,
            markerstrokewidth = 1.8,
        )
    end

    best_xy = hinged_bitstring_coordinates(best.subset_index, spec.n_features)
    scatter!(
        p,
        [best_xy[1]],
        [best_xy[2]];
        markersize = [maximum(sizes) + 4],
        markercolor = RGBA(0, 0, 0, 0),
        markerstrokecolor = :red,
        markerstrokewidth = 2.5,
    )

    annotate!(
        p,
        best_xy[1],
        best_xy[2],
        text("best=$(best.subset_index)", 9, :red, :left),
    )

    savefig(p, output)

    println("Saved HBM plot to $(abspath(output))")
    println("Local optima: $(length(lon.node_subset_indices))")
    println("Best subset: $(best.subset_index) -> columns $(best.active_columns)")
    println("Interpretation:")
    println("  each point is one subset")
    println("  x = first half of the bitstring as decimal")
    println("  y = second half of the bitstring as decimal")
    println("  greener and larger points are better")
    println("  blue outlines are local optima")
    println("  red outline is the global optimum under the chosen score")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, score_type, output = parsed
    plot_hbm(dataset_key, score_type, output)
end
