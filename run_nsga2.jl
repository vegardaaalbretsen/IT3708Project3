using IT3708Project3
using Random

function usage()
    println("Usage: julia --project=. run_nsga2.jl [dataset-key|triangle] [iterations] [epsilon] [seed] [initial-index] [output-path]")
    println("       julia --project=. run_nsga2.jl [dataset-key|triangle] [iterations] [epsilon] [--seed N] [--initial-index I] [--popsize N] [--pc V] [--pm V] [--log-every N] [--plot none|front|trace|stn|both|all] [--stn-first N] [--stn-last N] [--output path] [--plot-output path]")
    println("")
    println("Examples:")
    println("  julia --project=. run_nsga2.jl breast-w")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01")
    println("  julia --project=. run_nsga2.jl triangle 500 0.0 42 0")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --popsize 300 --pc 0.6 --pm 0.1875")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot front")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot stn")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot stn --stn-first 5")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot stn --stn-last 100")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot both")
    println("  julia --project=. run_nsga2.jl breast-w 300 0.01 --plot all")
end

default_cli_epsilon(dataset_key::AbstractString) = dataset_key == "triangle" ? 0.0 : 0.01

function parse_cli(args::Vector{String})
    positional = String[]
    seed = nothing
    initial_index = nothing
    population_size = nothing
    crossover_probability = nothing
    mutation_probability = nothing
    log_every = nothing
    plot_kind = nothing
    stn_first_generations = nothing
    stn_last_generations = nothing
    output_path = nothing
    plot_output_path = nothing
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
        elseif arg == "--popsize"
            i < length(args) || error("Missing value for --popsize")
            population_size = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--pc"
            i < length(args) || error("Missing value for --pc")
            crossover_probability = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--pm"
            i < length(args) || error("Missing value for --pm")
            mutation_probability = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--log-every"
            i < length(args) || error("Missing value for --log-every")
            log_every = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--plot"
            i < length(args) || error("Missing value for --plot")
            plot_kind = args[i + 1]
            i += 2
        elseif arg == "--stn-first"
            i < length(args) || error("Missing value for --stn-first")
            stn_first_generations = parse(Int, args[i + 1])
            stn_first_generations > 0 || error("--stn-first must be positive")
            i += 2
        elseif arg == "--stn-last"
            i < length(args) || error("Missing value for --stn-last")
            stn_last_generations = parse(Int, args[i + 1])
            stn_last_generations > 0 || error("--stn-last must be positive")
            i += 2
        elseif arg == "--output"
            i < length(args) || error("Missing value for --output")
            output_path = args[i + 1]
            i += 2
        elseif arg == "--plot-output"
            i < length(args) || error("Missing value for --plot-output")
            plot_output_path = args[i + 1]
            i += 2
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional, arg)
            i += 1
        end
    end

    length(positional) <= 6 || error("Too many positional arguments")

    dataset_key = length(positional) >= 1 ? positional[1] : "breast-w"
    iterations = length(positional) >= 2 ? parse(Int, positional[2]) : 300
    epsilon = length(positional) >= 3 ? parse(Float64, positional[3]) : default_cli_epsilon(dataset_key)

    if isnothing(seed) && length(positional) >= 4
        seed = parse(Int, positional[4])
    end

    if isnothing(initial_index) && length(positional) >= 5
        initial_index = parse(Int, positional[5])
    end

    if isnothing(output_path) && length(positional) >= 6
        output_path = positional[6]
    end

    plot_kind = isnothing(plot_kind) ? "none" : plot_kind
    plot_kind in ("none", "front", "trace", "stn", "both", "all") ||
        error("plot-kind must be one of: none, front, trace, stn, both, all")

    if plot_kind in ("both", "all") && !isnothing(plot_output_path)
        error("--plot-output can only be used with plot kinds 'front', 'trace', or 'stn'")
    end

    if !isnothing(stn_first_generations) && !isnothing(stn_last_generations)
        error("--stn-first and --stn-last cannot be used together")
    end

    return (
        dataset_key = dataset_key,
        iterations = iterations,
        epsilon = epsilon,
        seed = seed,
        initial_index = initial_index,
        population_size = isnothing(population_size) ? 300 : population_size,
        crossover_probability = isnothing(crossover_probability) ? 0.60 : crossover_probability,
        mutation_probability = isnothing(mutation_probability) ? 0.1875 : mutation_probability,
        log_every = isnothing(log_every) ? 0 : log_every,
        plot_kind = plot_kind,
        stn_first_generations = stn_first_generations,
        stn_last_generations = stn_last_generations,
        output_path = output_path,
        plot_output_path = plot_output_path,
    )
end

function default_nsga2_output_name(dataset_key::AbstractString, epsilon::Real)
    return if epsilon == 0
        "$(dataset_key)_nsga2_front"
    else
        epsilon_tag = replace(string(epsilon), "." => "p")
        "$(dataset_key)_nsga2_front_e$(epsilon_tag)"
    end
end

function default_nsga2_pareto_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_nsga2_pareto"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_nsga2_pareto_e$(epsilon_tag)"
end

function default_nsga2_trace_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_nsga2_trace"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_nsga2_trace_e$(epsilon_tag)"
end

function default_nsga2_stn_plot_name(dataset_key::AbstractString, epsilon::Real)
    if epsilon == 0
        return "$(dataset_key)_nsga2_stn"
    end

    epsilon_tag = replace(string(epsilon), "." => "p")
    return "$(dataset_key)_nsga2_stn_e$(epsilon_tag)"
end

function nsga2_stn_title(dataset_key::AbstractString,
                         epsilon::Real,
                         stn_first_generations,
                         stn_last_generations)
    if isnothing(stn_first_generations) && isnothing(stn_last_generations)
        return epsilon == 0 ?
            "$(dataset_key) NSGA-II search trajectory network" :
            "$(dataset_key) NSGA-II search trajectory network (epsilon=$(epsilon), reporting only)"
    end

    if !isnothing(stn_first_generations)
        return epsilon == 0 ?
            "$(dataset_key) NSGA-II search trajectory network (first $(stn_first_generations) generations)" :
            "$(dataset_key) NSGA-II search trajectory network (epsilon=$(epsilon), first $(stn_first_generations) generations)"
    end

    return epsilon == 0 ?
        "$(dataset_key) NSGA-II search trajectory network (last $(stn_last_generations) generations)" :
        "$(dataset_key) NSGA-II search trajectory network (epsilon=$(epsilon), last $(stn_last_generations) generations)"
end

function write_pareto_front(result, output_path::AbstractString)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, "index,accuracy,num_selected,time,penalized_fitness")
        for i in eachindex(result.pareto_indices)
            println(
                io,
                string(
                    result.pareto_indices[i], ",",
                    result.pareto_accuracy[i], ",",
                    result.pareto_num_selected[i], ",",
                    result.pareto_time[i], ",",
                    result.pareto_penalized_fitness[i],
                ),
            )
        end
    end

    return output_path
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

cli = parse_cli(ARGS)
rng = isnothing(cli.seed) ? Random.default_rng() : MersenneTwister(cli.seed)
output_path = isnothing(cli.output_path) ?
    default_nsga2_result_path(default_nsga2_output_name(cli.dataset_key, cli.epsilon)) :
    cli.output_path

landscape = load_landscape_key(cli.dataset_key)
result = run_nsga2_feature_ea(
    landscape;
    iterations=cli.iterations,
    epsilon=cli.epsilon,
    initial_index=cli.initial_index,
    population_size=cli.population_size,
    crossover_probability=cli.crossover_probability,
    mutation_probability=cli.mutation_probability,
    log_every=cli.log_every,
    rng=rng,
    keep_history=cli.plot_kind in ("trace", "stn", "both", "all"),
)

println("NSGA-II feature EA on `$(landscape.name)`")
println("Iterations: $(result.iterations)")
println("Population size: $(result.population_size)")
println("Epsilon (reporting only): $(result.epsilon)")
println("Threaded evaluation: $(result.threaded_evaluation)")
println("Parameters: pc=$(result.crossover_probability), pm=$(result.mutation_probability)")
println("Evaluations: $(result.evaluations)")
println("Pareto front size: $(length(result.pareto_indices))")
println(
    "Best penalized: index=$(result.best_penalized_index), features=$(result.best_penalized_num_selected), time=$(result.best_penalized_time), " *
    "accuracy=$(result.best_penalized_accuracy), penalized=$(result.best_penalized_fitness)",
)

println("Pareto front (sorted by selected features, then accuracy):")
max_rows = min(20, length(result.pareto_indices))
for i in 1:max_rows
    println(
        "  $(i): index=$(result.pareto_indices[i]), features=$(result.pareto_num_selected[i]), time=$(result.pareto_time[i]), " *
        "accuracy=$(result.pareto_accuracy[i]), penalized=$(result.pareto_penalized_fitness[i])",
    )
end

if length(result.pareto_indices) > max_rows
    println("  ... ($(length(result.pareto_indices) - max_rows) more front points)")
end

saved_path = write_pareto_front(result, output_path)
println("Saved Pareto front CSV for `$(landscape.name)` to `$saved_path`.")

if cli.plot_kind in ("front", "both", "all")
    front_output = if cli.plot_kind == "front" && !isnothing(cli.plot_output_path)
        cli.plot_output_path
    else
        default_ea_plot_path(default_nsga2_pareto_plot_name(cli.dataset_key, cli.epsilon))
    end

    front_title = cli.epsilon == 0 ?
        "$(landscape.name) NSGA-II Pareto front" :
        "$(landscape.name) NSGA-II Pareto front (epsilon=$(cli.epsilon), reporting only)"

    saved_front = save_nsga2_pareto_front_plot(
        landscape,
        result,
        front_output;
        title=front_title,
    )
    println("Saved NSGA-II Pareto front plot for `$(landscape.name)` to `$saved_front`.")
end

if cli.plot_kind in ("trace", "both", "all")
    trace_output = if cli.plot_kind == "trace" && !isnothing(cli.plot_output_path)
        cli.plot_output_path
    else
        default_ea_plot_path(default_nsga2_trace_plot_name(cli.dataset_key, cli.epsilon))
    end

    trace_title = cli.epsilon == 0 ?
        "$(landscape.name) NSGA-II trace" :
        "$(landscape.name) NSGA-II trace (epsilon=$(cli.epsilon), reporting only)"

    saved_trace = save_nsga2_trace_plot(
        result,
        trace_output;
        title=trace_title,
    )
    println("Saved NSGA-II trace plot for `$(landscape.name)` to `$saved_trace`.")
end

if cli.plot_kind in ("stn", "all")
    stn_output = if cli.plot_kind == "stn" && !isnothing(cli.plot_output_path)
        cli.plot_output_path
    else
        base_name = default_nsga2_stn_plot_name(cli.dataset_key, cli.epsilon)
        if !isnothing(cli.stn_first_generations)
            base_name *= "_first$(cli.stn_first_generations)"
        elseif !isnothing(cli.stn_last_generations)
            base_name *= "_last$(cli.stn_last_generations)"
        end
        default_stn_plot_path(base_name)
    end

    stn_title = nsga2_stn_title(
        landscape.name,
        cli.epsilon,
        cli.stn_first_generations,
        cli.stn_last_generations,
    )

    saved_stn = save_nsga2_search_trajectory_network_plot(
        landscape,
        result,
        stn_output;
        first_generations=cli.stn_first_generations,
        values=penalized_fitness_values(landscape, cli.epsilon),
        last_generations=cli.stn_last_generations,
        title=stn_title,
        fitness_label=cli.epsilon == 0 ? "Fitness" : "Penalized fitness",
    )
    println("Saved NSGA-II search trajectory network for `$(landscape.name)` to `$saved_stn`.")
end
