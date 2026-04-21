using IT3708Project3
using Plots

const DEFAULT_SNAPSHOT_CSV = joinpath("exports", "csv", "experiments", "population_snapshots.csv")
const DEFAULT_OUTPUT_DIR = joinpath("exports", "plots", "experiments", "snapshots")
const ALLOWED_PLOT_KINDS = Set(["both", "feature-count", "hbm"])

struct SnapshotRow
    algorithm::String
    landscape::String
    seed::Int
    epsilon::Float64
    snapshot_order::Int
    snapshot_label::String
    generation::Int
    index::Int
    bitstring::String
    count::Int
    num_selected::Int
    accuracy::Float64
    time::Float64
    penalized_fitness::Float64
end

function usage()
    println("Usage: julia --project=. plot_population_snapshots.jl --landscape KEY --algorithm ALG --seed N [options]")
    println("")
    println("Required:")
    println("  --landscape KEY         Landscape key/name, e.g. breast-w")
    println("  --algorithm ALG         ga, nsga2, or swarm")
    println("  --seed N                Seed/run to plot")
    println("")
    println("Options:")
    println("  --epsilon E             Epsilon to plot; defaults to the first matching epsilon")
    println("  --snapshot-csv PATH     Snapshot CSV, default $(DEFAULT_SNAPSHOT_CSV)")
    println("  --output-dir PATH       Output directory, default $(DEFAULT_OUTPUT_DIR)")
    println("  --plot both|feature-count|hbm")
    println("  -h, --help              Show this help")
end

function parse_cli(args::Vector{String})
    landscape = nothing
    algorithm = nothing
    seed = nothing
    epsilon = nothing
    snapshot_csv = DEFAULT_SNAPSHOT_CSV
    output_dir = DEFAULT_OUTPUT_DIR
    plot_kind = "both"
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--landscape"
            i < length(args) || error("Missing value for --landscape")
            landscape = args[i + 1]
            i += 2
        elseif arg == "--algorithm"
            i < length(args) || error("Missing value for --algorithm")
            algorithm = args[i + 1]
            i += 2
        elseif arg == "--seed"
            i < length(args) || error("Missing value for --seed")
            seed = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--epsilon"
            i < length(args) || error("Missing value for --epsilon")
            epsilon = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--snapshot-csv"
            i < length(args) || error("Missing value for --snapshot-csv")
            snapshot_csv = args[i + 1]
            i += 2
        elseif arg == "--output-dir"
            i < length(args) || error("Missing value for --output-dir")
            output_dir = args[i + 1]
            i += 2
        elseif arg == "--plot"
            i < length(args) || error("Missing value for --plot")
            plot_kind = args[i + 1]
            plot_kind in ALLOWED_PLOT_KINDS || error("--plot must be one of: $(join(sort(collect(ALLOWED_PLOT_KINDS)), ", "))")
            i += 2
        elseif arg in ("-h", "--help")
            usage()
            exit()
        else
            error("Unknown option: $arg")
        end
    end

    isnothing(landscape) && error("--landscape is required")
    isnothing(algorithm) && error("--algorithm is required")
    isnothing(seed) && error("--seed is required")

    return (
        landscape = String(landscape),
        algorithm = String(algorithm),
        seed = Int(seed),
        epsilon = epsilon,
        snapshot_csv = snapshot_csv,
        output_dir = output_dir,
        plot_kind = plot_kind,
    )
end

function read_population_snapshots_csv(path::AbstractString)
    isfile(path) || error("Could not find population snapshot CSV: $path")
    lines = readlines(path)
    length(lines) >= 2 || error("Population snapshot CSV has no data rows: $path")

    header = split(lines[1], ',')
    column_index = Dict(name => i for (i, name) in enumerate(header))
    required = [
        "algorithm", "landscape", "seed", "epsilon", "snapshot_order", "snapshot_label",
        "generation", "index", "bitstring", "count", "num_selected", "accuracy", "time",
        "penalized_fitness",
    ]
    for column in required
        haskey(column_index, column) || error("Missing required column '$column' in $path")
    end

    rows = SnapshotRow[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')

        push!(
            rows,
            SnapshotRow(
                fields[column_index["algorithm"]],
                fields[column_index["landscape"]],
                parse(Int, fields[column_index["seed"]]),
                parse(Float64, fields[column_index["epsilon"]]),
                parse(Int, fields[column_index["snapshot_order"]]),
                fields[column_index["snapshot_label"]],
                parse(Int, fields[column_index["generation"]]),
                parse(Int, fields[column_index["index"]]),
                fields[column_index["bitstring"]],
                parse(Int, fields[column_index["count"]]),
                parse(Int, fields[column_index["num_selected"]]),
                parse(Float64, fields[column_index["accuracy"]]),
                parse(Float64, fields[column_index["time"]]),
                parse(Float64, fields[column_index["penalized_fitness"]]),
            ),
        )
    end

    isempty(rows) && error("Population snapshot CSV has no usable rows: $path")
    return rows
end

function filter_snapshot_rows(rows::Vector{SnapshotRow},
                              landscape::AbstractString,
                              algorithm::AbstractString,
                              seed::Integer,
                              epsilon)
    matching = [
        row for row in rows
        if row.landscape == landscape && row.algorithm == algorithm && row.seed == Int(seed)
    ]
    isempty(matching) && error("No snapshot rows found for landscape=$(landscape), algorithm=$(algorithm), seed=$(seed)")

    chosen_epsilon = isnothing(epsilon) ? sort(unique([row.epsilon for row in matching]))[1] : Float64(epsilon)
    matching = [row for row in matching if isapprox(row.epsilon, chosen_epsilon; atol=1e-12, rtol=1e-12)]
    isempty(matching) && error("No snapshot rows found for epsilon=$(chosen_epsilon)")

    sort!(matching; by = row -> (row.snapshot_order, row.index))
    return matching, chosen_epsilon
end

function rows_by_generation(rows::Vector{SnapshotRow})
    groups = Dict{Int, Vector{SnapshotRow}}()
    for row in rows
        push!(get!(groups, row.generation, SnapshotRow[]), row)
    end
    return [(generation, groups[generation]) for generation in sort(collect(keys(groups)))]
end

function safe_filename_part(value)
    text = string(value)
    return replace(text, r"[^A-Za-z0-9_.-]" => "_")
end

function snapshot_title(kind::AbstractString,
                        landscape::AbstractString,
                        algorithm::AbstractString,
                        seed::Integer,
                        epsilon::Real,
                        generation::Integer)
    return "$(landscape) $(algorithm) $(kind) snapshot, seed=$(seed), epsilon=$(epsilon), generation=$(generation)"
end

function count_labels(rows::Vector{SnapshotRow})
    return [row.count > 1 ? string(row.count) : "" for row in rows]
end

function population_marker_sizes(rows::Vector{SnapshotRow}; base::Real = 6.0, scale::Real = 5.0)
    return [Float64(base + scale * sqrt(row.count)) for row in rows]
end

function population_layers(rows::Vector{SnapshotRow}, local_set::Set, global_set::Set)
    other_rows = [row for row in rows if !(row.index in local_set) && !(row.index in global_set)]
    local_rows = [row for row in rows if row.index in local_set && !(row.index in global_set)]
    global_rows = [row for row in rows if row.index in global_set]
    return (
        (other_rows, :orange, "Population on other subsets"),
        (local_rows, :dodgerblue3, "Population on local optima"),
        (global_rows, :red2, "Population on global optima"),
    )
end

function add_feature_count_totals!(plt, rows::Vector{SnapshotRow}, values)
    totals = Dict{Int, Int}()
    for row in rows
        totals[row.num_selected] = get(totals, row.num_selected, 0) + row.count
    end

    min_value = minimum(values)
    max_value = maximum(values)
    y = min_value + 0.04 * max(max_value - min_value, eps())

    for feature_count in sort(collect(keys(totals)))
        annotate!(plt, feature_count, y, text("n=$(totals[feature_count])", 8, :black))
    end

    return plt
end

function save_feature_count_snapshot_plot(landscape::Landscape,
                                          rows::Vector{SnapshotRow},
                                          output_path::AbstractString;
                                          algorithm::AbstractString,
                                          seed::Integer,
                                          epsilon::Real)
    values = penalized_fitness_values(landscape, epsilon)
    local_set = Set(local_optima(landscape; values=values))
    global_set = Set(global_optima(landscape; values=values))
    generation = first(rows).generation
    plt = plot_fitness_by_feature_count(
        landscape;
        values=values,
        title=snapshot_title("feature-count", landscape.name, algorithm, seed, epsilon, generation),
        fitness_label="Penalized fitness",
    )

    for (layer_rows, color, label) in population_layers(rows, local_set, global_set)
        isempty(layer_rows) && continue
        scatter!(
            plt,
            [row.num_selected for row in layer_rows],
            [row.penalized_fitness for row in layer_rows];
            ms=population_marker_sizes(layer_rows),
            markercolor=color,
            markerstrokecolor=:black,
            markerstrokewidth=1.2,
            alpha=0.9,
            label=label,
            series_annotations=text.(count_labels(layer_rows), 8, :black),
        )
    end
    add_feature_count_totals!(plt, rows, values)

    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function save_hbm_snapshot_plot(landscape::Landscape,
                                rows::Vector{SnapshotRow},
                                output_path::AbstractString;
                                algorithm::AbstractString,
                                seed::Integer,
                                epsilon::Real)
    values = penalized_fitness_values(landscape, epsilon)
    nodes = build_hbm(landscape; values=values)
    node_lookup = Dict(node.index => node for node in nodes)
    generation = first(rows).generation
    x_bits = ceil(Int, landscape.num_features / 2)
    y_bits = fld(landscape.num_features, 2)
    x_max = (1 << x_bits) - 1
    y_max = (1 << y_bits) - 1

    plt = scatter(
        Float64[node.x for node in nodes],
        Float64[node.y for node in nodes];
        marker_z=Float64[node.fitness for node in nodes],
        color=cgrad([:darkgreen, :ivory, :purple]),
        ms=clamp(min(520 / (x_max + 1), 340 / (y_max + 1)), 1.5, 18.0),
        markerstrokewidth=0,
        xlabel="First half of bitstring",
        ylabel="Second half of bitstring",
        title=snapshot_title("HBM", landscape.name, algorithm, seed, epsilon, generation),
        label="",
        legend=:topright,
        colorbar=:right,
        colorbar_title="Penalized fitness",
        aspect_ratio=:equal,
        size=(2200, 1400),
        dpi=300,
        xticks=([0.0, Float64(x_max)], ["2^0 - 1", "2^$(x_bits) - 1"]),
        yticks=([0.0, Float64(y_max)], ["2^0 - 1", "2^$(y_bits) - 1"]),
        xlims=(-0.8, x_max + 0.8),
        ylims=(-0.8, y_max + 0.8),
        grid=true,
        gridalpha=0.18,
        background_color=:white,
    )

    local_indices = local_optima(nodes, landscape.num_features; allow_zero=landscape.allow_zero)
    global_indices = global_optima(nodes)
    global_set = Set(global_indices)
    local_set = Set(local_indices)
    local_only = [index for index in local_indices if !(index in global_set)]
    sort!(local_only; by = index -> (-node_lookup[index].fitness, index))
    local_only = local_only[1:min(length(local_only), 50)]

    if !isempty(local_only)
        local_nodes = [node_lookup[index] for index in local_only]
        scatter!(
            plt,
            Float64[node.x for node in local_nodes],
            Float64[node.y for node in local_nodes];
            ms=18,
            markercolor=:dodgerblue3,
            markerstrokecolor=:white,
            markerstrokewidth=2.0,
            label="Local optimum",
        )
    end

    if !isempty(global_indices)
        global_nodes = [node_lookup[index] for index in global_indices]
        scatter!(
            plt,
            Float64[node.x for node in global_nodes],
            Float64[node.y for node in global_nodes];
            ms=22,
            markercolor=:red2,
            markerstrokecolor=:white,
            markerstrokewidth=2.4,
            label="Global optimum",
        )
    end

    for (layer_rows, color, label) in population_layers(rows, local_set, global_set)
        isempty(layer_rows) && continue
        layer_nodes = [node_lookup[row.index] for row in layer_rows]
        scatter!(
            plt,
            Float64[node.x for node in layer_nodes],
            Float64[node.y for node in layer_nodes];
            ms=population_marker_sizes(layer_rows; base=7.0, scale=6.0),
            markercolor=color,
            markerstrokecolor=:black,
            markerstrokewidth=2.0,
            alpha=0.9,
            label=label,
            series_annotations=text.(count_labels(layer_rows), 12, :black),
        )
    end

    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

function snapshot_output_path(output_dir::AbstractString,
                              landscape::AbstractString,
                              algorithm::AbstractString,
                              seed::Integer,
                              epsilon::Real,
                              generation::Integer,
                              plot_kind::AbstractString)
    plot_dir = plot_kind == "feature-count" ? "feature_count" : safe_filename_part(plot_kind)
    filename = join(
        [
            "seed$(seed)",
            "epsilon-$(safe_filename_part(epsilon))",
            "gen$(generation)",
        ],
        "_",
    ) * ".png"
    return joinpath(
        output_dir,
        safe_filename_part(landscape),
        safe_filename_part(algorithm),
        plot_dir,
        filename,
    )
end

function plot_population_snapshots(; snapshot_csv::AbstractString,
                                     output_dir::AbstractString,
                                     landscape_key::AbstractString,
                                     algorithm::AbstractString,
                                     seed::Integer,
                                     epsilon,
                                     plot_kind::AbstractString)
    all_rows = read_population_snapshots_csv(snapshot_csv)
    rows, chosen_epsilon = filter_snapshot_rows(all_rows, landscape_key, algorithm, seed, epsilon)
    landscape = load_landscape_key(landscape_key)
    saved_paths = String[]

    for (generation, snapshot_rows) in rows_by_generation(rows)
        if plot_kind in ("both", "feature-count")
            output_path = snapshot_output_path(
                output_dir,
                landscape.name,
                algorithm,
                seed,
                chosen_epsilon,
                generation,
                "feature-count",
            )
            push!(
                saved_paths,
                save_feature_count_snapshot_plot(
                    landscape,
                    snapshot_rows,
                    output_path;
                    algorithm=algorithm,
                    seed=seed,
                    epsilon=chosen_epsilon,
                ),
            )
        end

        if plot_kind in ("both", "hbm")
            output_path = snapshot_output_path(
                output_dir,
                landscape.name,
                algorithm,
                seed,
                chosen_epsilon,
                generation,
                "hbm",
            )
            push!(
                saved_paths,
                save_hbm_snapshot_plot(
                    landscape,
                    snapshot_rows,
                    output_path;
                    algorithm=algorithm,
                    seed=seed,
                    epsilon=chosen_epsilon,
                ),
            )
        end
    end

    return saved_paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_cli(ARGS)
    saved_paths = plot_population_snapshots(
        snapshot_csv=options.snapshot_csv,
        output_dir=options.output_dir,
        landscape_key=options.landscape,
        algorithm=options.algorithm,
        seed=options.seed,
        epsilon=options.epsilon,
        plot_kind=options.plot_kind,
    )

    println("Saved $(length(saved_paths)) plot(s):")
    for path in saved_paths
        println("  $(path)")
    end
end
