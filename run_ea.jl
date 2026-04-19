using IT3708Project3
using Random

function usage()
    println("Usage: julia --project=. run_ea.jl [dataset-key|triangle] [iterations] [epsilon] [seed] [initial-index] [plot-kind] [output-path]")
    println("       julia --project=. run_ea.jl [dataset-key|triangle] [iterations] [epsilon] [--seed N] [--initial-index I] [--plot none|trace|feature-count|both] [--output path]")
    println("       dataset-key defaults to \"breast-w\" when omitted")
    println("")
    println("Plot kinds: none, trace, feature-count, both")
    println("")
    println("Examples:")
    println("  julia --project=. run_ea.jl breast-w")
    println("  julia --project=. run_ea.jl breast-w 10000 0.01")
    println("  julia --project=. run_ea.jl triangle 5000 0.0 42 0")
    println("  julia --project=. run_ea.jl breast-w 10000 0.01 --plot trace --seed 42")
    println("  julia --project=. run_ea.jl breast-w 10000 0.01 --plot feature-count --seed 42")
    println("  julia --project=. run_ea.jl triangle 5000 0.1 --plot both --seed 42 --initial-index 0")
end

function parse_cli(args::Vector{String})
    positional = String[]
    seed = nothing
    initial_index = nothing
    plot_kind = nothing
    output_path = nothing
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--seed"
            i < length(args) || error("Missing value for --seed")
            seed = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--initial-index"
            i < length(args) || error("Missing value for --initial-index")
            initial_index = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--plot"
            i < length(args) || error("Missing value for --plot")
            plot_kind = args[i + 1]
            i += 2
        elseif arg == "--output"
            i < length(args) || error("Missing value for --output")
            output_path = args[i + 1]
            i += 2
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
            i += 1
        end
    end

    length(positional) <= 7 || error("Too many positional arguments")

    dataset_key = length(positional) >= 1 ? positional[1] : "breast-w"
    iterations = length(positional) >= 2 ? parse(Int, positional[2]) : 10_000
    epsilon = length(positional) >= 3 ? parse(Float64, positional[3]) : 0.0

    if isnothing(seed) && length(positional) >= 4
        seed = parse(Int, positional[4])
    end

    if isnothing(initial_index) && length(positional) >= 5
        initial_index = parse(Int, positional[5])
    end

    if isnothing(plot_kind) && length(positional) >= 6
        plot_kind = positional[6]
    end

    if isnothing(output_path) && length(positional) >= 7
        output_path = positional[7]
    end

    plot_kind = isnothing(plot_kind) ? "none" : plot_kind
    plot_kind in ("none", "trace", "feature-count", "both") ||
        error("plot-kind must be one of: none, trace, feature-count, both")

    if plot_kind == "both" && !isnothing(output_path)
        error("--output can only be used with plot kinds 'trace' or 'feature-count'")
    end

    return (
        dataset_key = dataset_key,
        iterations = iterations,
        epsilon = epsilon,
        seed = seed,
        initial_index = initial_index,
        plot_kind = plot_kind,
        output_path = output_path,
    )
end

function default_trace_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_ea_trace"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_ea_trace_e$(epsilon_tag)"
end

function default_feature_count_overlay_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_ea_feature_count"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_ea_feature_count_e$(epsilon_tag)"
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

cli = parse_cli(ARGS)
rng = isnothing(cli.seed) ? Random.default_rng() : MersenneTwister(cli.seed)

landscape = load_landscape_key(cli.dataset_key)
result = run_standard_ea(
    landscape;
    iterations=cli.iterations,
    epsilon=cli.epsilon,
    rng=rng,
    initial_index=cli.initial_index,
    keep_history=cli.plot_kind != "none",
)

println("Standard EA on `$(landscape.name)`")
println("Iterations: $(result.iterations)")
println("Epsilon: $(result.epsilon)")
println("Accepted moves: $(result.accepted_moves)")
println("Initial: index=$(result.initial_index), features=$(result.initial_num_selected), accuracy=$(result.initial_accuracy), penalized=$(result.initial_penalized_fitness)")
println("Final:   index=$(result.final_index), features=$(result.final_num_selected), accuracy=$(result.final_accuracy), penalized=$(result.final_penalized_fitness)")
println("Best:    index=$(result.best_index), features=$(result.best_num_selected), accuracy=$(result.best_accuracy), penalized=$(result.best_penalized_fitness)")

values = cli.epsilon == 0 ? fitness_values(landscape) : penalized_fitness_values(landscape, cli.epsilon)
fitness_label = cli.epsilon == 0 ? "Fitness" : "Penalized fitness"

if cli.plot_kind in ("trace", "both")
    trace_output = if cli.plot_kind == "trace" && !isnothing(cli.output_path)
        cli.output_path
    else
        default_ea_plot_path(default_trace_plot_name(cli.dataset_key, cli.epsilon))
    end

    trace_title = cli.epsilon == 0 ?
        "$(landscape.name) standard EA trace" :
        "$(landscape.name) standard EA trace (epsilon=$(cli.epsilon))"

    saved_path = save_ea_trace_plot(result, trace_output; title=trace_title, fitness_label=fitness_label)
    println("Saved EA trace plot for `$(landscape.name)` to `$saved_path`.")
end

if cli.plot_kind in ("feature-count", "both")
    overlay_output = if cli.plot_kind == "feature-count" && !isnothing(cli.output_path)
        cli.output_path
    else
        default_ea_plot_path(default_feature_count_overlay_name(cli.dataset_key, cli.epsilon))
    end

    overlay_title = cli.epsilon == 0 ?
        "$(landscape.name) fitness by feature count with EA path" :
        "$(landscape.name) fitness by feature count with EA path (epsilon=$(cli.epsilon))"

    saved_path = save_fitness_by_feature_count_with_ea_plot(
        landscape,
        result,
        overlay_output;
        values=values,
        title=overlay_title,
        fitness_label=fitness_label,
    )
    println("Saved EA feature-count plot for `$(landscape.name)` to `$saved_path`.")
end
