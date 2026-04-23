using IT3708Project3
using Random

function usage()
    println("Usage: julia --project=. run_swarm.jl <dataset-key|triangle> [iterations] [epsilon] [seed] [swarm-size] [w] [c1] [c2] [plot-kind] [output-path]")
    println("       julia --project=. run_swarm.jl <dataset-key|triangle> [iterations] [epsilon] [--seed N] [--swarm-size N] [--w V] [--c1 V] [--c2 V] [--plot none|trace|feature-count|hbm|all] [--output path]")
    println("")
    println("Plot kinds: none, trace, feature-count, hbm, all")
    println("")
    println("Examples:")
    println("  julia --project=. run_swarm.jl breast-w")
    println("  julia --project=. run_swarm.jl breast-w 300 0.01")
    println("  julia --project=. run_swarm.jl triangle 300 0.0 --seed 42")
    println("  julia --project=. run_swarm.jl breast-w 300 0.01 --swarm-size 100 --w 0.95 --c1 2.0 --c2 0.4")
    println("  julia --project=. run_swarm.jl breast-w 300 0.01 --plot feature-count --seed 42")
    println("  julia --project=. run_swarm.jl breast-w 300 0.01 --plot trace --seed 42")
    println("  julia --project=. run_swarm.jl triangle 300 0.0 --plot hbm --seed 42")
end

default_cli_epsilon(dataset_key::AbstractString) = dataset_key == "triangle" ? 0.0 : 0.01

function parse_cli(args::Vector{String})
    positional = String[]
    seed = nothing
    swarm_size = nothing
    w = nothing
    c1 = nothing
    c2 = nothing
    plot_kind = nothing
    output_path = nothing
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--seed"
            i < length(args) || error("Missing value for --seed")
            seed = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--swarm-size"
            i < length(args) || error("Missing value for --swarm-size")
            swarm_size = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--w"
            i < length(args) || error("Missing value for --w")
            w = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--c1"
            i < length(args) || error("Missing value for --c1")
            c1 = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--c2"
            i < length(args) || error("Missing value for --c2")
            c2 = parse(Float64, args[i + 1])
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

    length(positional) <= 10 || error("Too many positional arguments")

    dataset_key = length(positional) >= 1 ? positional[1] : "breast-w"
    iterations = length(positional) >= 2 ? parse(Int, positional[2]) : 300
    epsilon = length(positional) >= 3 ? parse(Float64, positional[3]) : default_cli_epsilon(dataset_key)

    if isnothing(seed) && length(positional) >= 4
        seed = parse(Int, positional[4])
    end

    if isnothing(swarm_size) && length(positional) >= 5
        swarm_size = parse(Int, positional[5])
    end

    if isnothing(w) && length(positional) >= 6
        w = parse(Float64, positional[6])
    end

    if isnothing(c1) && length(positional) >= 7
        c1 = parse(Float64, positional[7])
    end

    if isnothing(c2) && length(positional) >= 8
        c2 = parse(Float64, positional[8])
    end

    if isnothing(plot_kind) && length(positional) >= 9
        plot_kind = positional[9]
    end

    if isnothing(output_path) && length(positional) >= 10
        output_path = positional[10]
    end

    plot_kind = isnothing(plot_kind) ? "none" : plot_kind
    plot_kind in ("none", "trace", "feature-count", "hbm", "all") ||
        error("plot-kind must be one of: none, trace, feature-count, hbm, all")

    if plot_kind == "all" && !isnothing(output_path)
        error("--output can only be used with plot kinds 'trace', 'feature-count', or 'hbm'")
    end

    return (
        dataset_key = dataset_key,
        iterations = iterations,
        epsilon = epsilon,
        seed = seed,
        swarm_size = isnothing(swarm_size) ? 100 : swarm_size,
        w = isnothing(w) ? 0.95 : w,
        c1 = isnothing(c1) ? 2.0 : c1,
        c2 = isnothing(c2) ? 0.4 : c2,
        plot_kind = plot_kind,
        output_path = output_path,
    )
end

function default_swarm_trace_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_swarm_trace"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_swarm_trace_e$(epsilon_tag)"
end

function default_swarm_feature_count_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_swarm_feature_count"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_swarm_feature_count_e$(epsilon_tag)"
end

function default_swarm_hbm_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_swarm"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_swarm_e$(epsilon_tag)"
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

cli = parse_cli(ARGS)
rng = isnothing(cli.seed) ? Random.default_rng() : MersenneTwister(cli.seed)

landscape = load_landscape_key(cli.dataset_key)
result = run_swarm_ea(
    landscape;
    iterations=cli.iterations,
    epsilon=cli.epsilon,
    swarm_size=cli.swarm_size,
    w=cli.w,
    c1=cli.c1,
    c2=cli.c2,
    keep_history=cli.plot_kind != "none",
    rng=rng,
)

println("Swarm EA on `$(landscape.name)`")
println("Iterations: $(result.iterations)")
println("Swarm size: $(result.swarm_size)")
println("Epsilon: $(result.epsilon)")
println("Threaded evaluation: $(result.threaded_evaluation)")
println("Parameters: w=$(result.w), c1=$(result.c1), c2=$(result.c2)")
println("Evaluations: $(result.evaluations)")
println("Runtime: $(round(result.runtime; digits=6)) seconds")
println("Best: index=$(result.best_index), features=$(result.best_num_selected), accuracy=$(result.best_accuracy), penalized=$(result.best_penalized_fitness)")
println("Best position: $(result.best_position)")

values = cli.epsilon == 0 ? fitness_values(landscape) : penalized_fitness_values(landscape, cli.epsilon)
fitness_label = cli.epsilon == 0 ? "Fitness" : "Penalized fitness"

if cli.plot_kind in ("trace", "all")
    trace_output = if cli.plot_kind == "trace" && !isnothing(cli.output_path)
        cli.output_path
    else
        default_ea_plot_path(default_swarm_trace_plot_name(cli.dataset_key, cli.epsilon))
    end

    trace_title = "$(landscape.name) swarm trace"
    if cli.epsilon != 0
        trace_title *= " (epsilon=$(cli.epsilon))"
    end

    saved_path = save_swarm_trace_plot(
        landscape,
        result,
        trace_output;
        title=trace_title,
        fitness_label=fitness_label,
    )
    println("Saved swarm trace plot for `$(landscape.name)` to `$saved_path`.")
end

if cli.plot_kind in ("feature-count", "all")
    feature_count_output = if cli.plot_kind == "feature-count" && !isnothing(cli.output_path)
        cli.output_path
    else
        default_ea_plot_path(default_swarm_feature_count_plot_name(cli.dataset_key, cli.epsilon))
    end

    feature_count_title = "$(landscape.name) fitness by feature count with swarm"
    if cli.epsilon != 0
        feature_count_title *= " (epsilon=$(cli.epsilon))"
    end

    saved_path = save_fitness_by_feature_count_with_swarm_plot(
        landscape,
        result,
        feature_count_output;
        values=values,
        title=feature_count_title,
        fitness_label=fitness_label,
    )
    println("Saved swarm feature-count plot for `$(landscape.name)` to `$saved_path`.")
end

if cli.plot_kind in ("hbm", "all")
    hbm_output = if cli.plot_kind == "hbm" && !isnothing(cli.output_path)
        cli.output_path
    else
        default_hbm_plot_path(default_swarm_hbm_plot_name(cli.dataset_key, cli.epsilon))
    end

    hbm_title = "$(landscape.name) HBM with swarm"
    if cli.epsilon != 0
        hbm_title *= " (epsilon=$(cli.epsilon))"
    end

    saved_path = save_hbm_with_swarm_plot(
        landscape,
        result,
        hbm_output;
        values=values,
        title=hbm_title,
        fitness_label=fitness_label,
    )
    println("Saved swarm HBM plot for `$(landscape.name)` to `$saved_path`.")
end
