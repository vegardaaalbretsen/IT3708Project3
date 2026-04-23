using Plots
using Statistics

const DEFAULT_INPUT_PATH = joinpath("exports", "csv", "experiments", "generation_stats.csv")
const DEFAULT_OUTPUT_DIR = joinpath("exports", "plots", "experiments")

const METRIC_LABELS = Dict(
    "min_fitness" => "Minimum fitness",
    "mean_fitness" => "Mean fitness",
    "max_fitness" => "Maximum fitness",
    "best_so_far_fitness" => "Best-so-far fitness",
    "diversity_entropy" => "Normalized diversity entropy",
    "global_optima_seen" => "Cumulative global optima seen",
    "global_optima_fraction" => "Global optima coverage",
)

struct GenerationStatRow
    algorithm::String
    landscape::String
    seed::Int
    epsilon::Float64
    generation::Int
    values::Dict{String, Float64}
end

function usage()
    println("Usage: julia --project=. plot_experiment_fitness.jl [options]")
    println("")
    println("Options:")
    println("  --input PATH        generation_stats.csv path, default $(DEFAULT_INPUT_PATH)")
    println("  --output-dir PATH   Plot output directory, default $(DEFAULT_OUTPUT_DIR)")
    println("  --metric NAME       Metric to plot, default best_so_far_fitness")
    println("                      Supported: $(join(sort(collect(keys(METRIC_LABELS))), ", "))")
    println("  -h, --help          Show this help")
end

function parse_cli(args::Vector{String})
    input_path = DEFAULT_INPUT_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    metric = "best_so_far_fitness"
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--input"
            i < length(args) || error("Missing value for --input")
            input_path = args[i + 1]
            i += 2
        elseif arg == "--output-dir"
            i < length(args) || error("Missing value for --output-dir")
            output_dir = args[i + 1]
            i += 2
        elseif arg == "--metric"
            i < length(args) || error("Missing value for --metric")
            metric = args[i + 1]
            i += 2
        elseif arg in ("-h", "--help")
            usage()
            exit()
        else
            error("Unknown option: $arg")
        end
    end

    haskey(METRIC_LABELS, metric) || error("Unknown metric: $metric")
    return (input_path = input_path, output_dir = output_dir, metric = metric)
end

function read_generation_stats_csv(path::AbstractString)
    isfile(path) || error("Could not find generation stats CSV: $path")
    lines = readlines(path)
    length(lines) >= 2 || error("Generation stats CSV has no data rows: $path")

    header = split(lines[1], ',')
    column_index = Dict(name => i for (i, name) in enumerate(header))
    required = ["algorithm", "landscape", "seed", "epsilon", "generation"]
    for column in required
        haskey(column_index, column) || error("Missing required column '$column' in $path")
    end

    rows = GenerationStatRow[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        fields = split(line, ',')

        values = Dict{String, Float64}()
        for metric in keys(METRIC_LABELS)
            if haskey(column_index, metric) && column_index[metric] <= length(fields)
                text = strip(fields[column_index[metric]])
                !isempty(text) && (values[metric] = parse(Float64, text))
            end
        end

        push!(
            rows,
            GenerationStatRow(
                fields[column_index["algorithm"]],
                fields[column_index["landscape"]],
                parse(Int, fields[column_index["seed"]]),
                parse(Float64, fields[column_index["epsilon"]]),
                parse(Int, fields[column_index["generation"]]),
                values,
            ),
        )
    end

    isempty(rows) && error("Generation stats CSV has no usable data rows: $path")
    return rows
end

function grouped_average_by_generation(rows::Vector{GenerationStatRow}, metric::AbstractString)
    groups = Dict{Tuple{String, Float64, String, Int}, Vector{Float64}}()

    for row in rows
        haskey(row.values, metric) || continue
        key = (row.landscape, row.epsilon, row.algorithm, row.generation)
        push!(get!(groups, key, Float64[]), row.values[metric])
    end

    isempty(groups) && error("Metric '$metric' was not found in the CSV")

    averaged = Dict{Tuple{String, Float64, String}, Vector{Tuple{Int, Float64}}}()
    for (key, values) in groups
        series_key = (key[1], key[2], key[3])
        push!(get!(averaged, series_key, Tuple{Int, Float64}[]), (key[4], mean(values)))
    end

    for points in values(averaged)
        sort!(points; by = first)
    end

    return averaged
end

function safe_filename_part(value)
    text = string(value)
    return replace(text, r"[^A-Za-z0-9_.-]" => "_")
end

function display_metric_label(metric::AbstractString, landscape::AbstractString)
    label = METRIC_LABELS[metric]

    if landscape == "triangle" && occursin("fitness", metric)
        label = replace(label, "Fitness" => "Objective value")
        label = replace(label, "fitness" => "objective value")
    end

    return label
end

function plot_title(metric::AbstractString, landscape::AbstractString, epsilon::Real)
    label = display_metric_label(metric, landscape)
    if landscape == "triangle"
        return "$(label) on $(landscape) (unpenalized)"
    end
    return "$(label) on $(landscape), epsilon=$(epsilon)"
end

function plot_metric_from_generation_stats(input_path::AbstractString,
                                           output_dir::AbstractString,
                                           metric::AbstractString)
    rows = read_generation_stats_csv(input_path)
    averaged = grouped_average_by_generation(rows, metric)
    plot_groups = sort(unique([(key[1], key[2]) for key in keys(averaged)]); by = key -> (key[1], key[2]))
    saved_paths = String[]

    mkpath(output_dir)

    for (landscape, epsilon) in plot_groups
        metric_label = display_metric_label(metric, landscape)
        plt = plot(
            xlabel = "Generation",
            ylabel = metric_label,
            title = plot_title(metric, landscape, epsilon),
            legend = :bottomright,
            linewidth = 3,
            size = (1100, 700),
            dpi = 180,
            left_margin = 18Plots.mm,
            right_margin = 8Plots.mm,
            top_margin = 8Plots.mm,
            bottom_margin = 12Plots.mm,
            gridalpha = 0.25,
            background_color = :white,
        )

        algorithms = sort([key[3] for key in keys(averaged) if key[1] == landscape && key[2] == epsilon])
        for algorithm in algorithms
            points = averaged[(landscape, epsilon, algorithm)]
            generations = [point[1] for point in points]
            values = [point[2] for point in points]
            plot!(plt, generations, values; label = algorithm)
        end

        filename_parts = [safe_filename_part(landscape)]
        if landscape == "triangle"
            push!(filename_parts, "unpenalized")
        else
            push!(filename_parts, "epsilon-$(safe_filename_part(epsilon))")
        end
        push!(filename_parts, safe_filename_part(metric))
        filename = join(filename_parts, "_") * ".png"
        output_path = joinpath(output_dir, filename)
        savefig(plt, output_path)
        push!(saved_paths, output_path)
    end

    return saved_paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_cli(ARGS)
    saved_paths = plot_metric_from_generation_stats(options.input_path, options.output_dir, options.metric)

    println("Saved $(length(saved_paths)) plot(s):")
    for path in saved_paths
        println("  $(path)")
    end
end
