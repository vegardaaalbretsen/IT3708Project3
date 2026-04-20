using IT3708Project3
using Random
using Statistics

const DEFAULT_LANDSCAPES = ["breast-w", "credit-a", "letter-r", "triangle"]
const DEFAULT_ALGORITHMS = [:ga, :nsga2, :swarm]
const DEFAULT_OUTPUT_DIR = joinpath("exports", "csv", "experiments")

struct ExperimentConfig
    landscape_keys::Vector{String}
    algorithms::Vector{Symbol}
    epsilons::Vector{Float64}
    seeds::Vector{Int}
    output_dir::String
    ga_iterations::Int
    ga_population_size::Int
    ga_crossover_probability::Float64
    ga_mutation_probability::Union{Nothing, Float64}
    ga_tournament_size::Int
    ga_survivor_mode::Symbol
    ga_elite::Int
    nsga2_iterations::Int
    nsga2_population_size::Int
    nsga2_crossover_probability::Float64
    nsga2_mutation_probability::Union{Nothing, Float64}
    swarm_iterations::Int
    swarm_size::Int
    swarm_w::Float64
    swarm_c1::Float64
    swarm_c2::Float64
end

struct ExperimentRun
    algorithm::Symbol
    landscape::String
    seed::Int
    epsilon::Float64
    best_index::Int
    best_bitstring::String
    accuracy::Float64
    num_selected::Int
    penalized_fitness::Float64
    evaluations::Int
    runtime::Float64
    parameters::String
end

struct GenerationStat
    algorithm::Symbol
    landscape::String
    seed::Int
    epsilon::Float64
    generation::Int
    min_fitness::Float64
    mean_fitness::Float64
    max_fitness::Float64
    best_so_far_fitness::Float64
end

struct ExperimentSummary
    algorithm::Symbol
    landscape::String
    epsilon::Float64
    runs::Int
    mean_best_fitness::Float64
    std_best_fitness::Float64
    min_best_fitness::Float64
    max_best_fitness::Float64
    mean_accuracy::Float64
    mean_num_selected::Float64
    mean_runtime::Float64
    best_index::Int
    best_bitstring::String
    best_accuracy::Float64
    best_num_selected::Int
    best_fitness::Float64
end

function usage()
    println("Usage: julia --threads auto --project=. run_experiments.jl [options]")
    println("")
    println("Options:")
    println("  --seeds N                     Use seeds 1:N, default 10")
    println("  --epsilon E[,E2,...]          Epsilon value(s), default 0.01")
    println("  --datasets a,b,c              Dataset keys, default breast-w,credit-a,letter-r,triangle")
    println("  --algorithms ga,nsga2,swarm   Algorithms to run, default all three")
    println("  --output-dir PATH             Output directory, default exports/csv/experiments")
    println("  -h, --help                    Show this help")
end

function default_config(; seed_count::Int = 10,
                        epsilons::Vector{Float64} = [0.01],
                        landscape_keys::Vector{String} = copy(DEFAULT_LANDSCAPES),
                        algorithms::Vector{Symbol} = copy(DEFAULT_ALGORITHMS),
                        output_dir::AbstractString = DEFAULT_OUTPUT_DIR)
    return ExperimentConfig(
        landscape_keys,
        algorithms,
        epsilons,
        collect(1:seed_count),
        String(output_dir),
        500,
        100,
        0.95,
        nothing,
        4,
        :elitist,
        4,
        500,
        100,
        0.95,
        nothing,
        500,
        40,
        0.7,
        1.4,
        1.4,
    )
end

function parse_list(value::AbstractString)
    items = strip.(split(value, ','))
    return [String(item) for item in items if !isempty(item)]
end

function parse_algorithms(value::AbstractString)
    algorithms = Symbol.(parse_list(value))
    allowed = Set(DEFAULT_ALGORITHMS)
    for algorithm in algorithms
        algorithm in allowed || error("Unknown algorithm: $algorithm")
    end
    return algorithms
end

function parse_epsilons(value::AbstractString)
    epsilons = parse.(Float64, parse_list(value))
    for epsilon in epsilons
        0 <= epsilon <= 1 || error("epsilon must be between 0 and 1, got $epsilon")
    end
    return epsilons
end

function parse_cli(args::Vector{String})
    seed_count = 10
    epsilons = [0.01]
    landscape_keys = copy(DEFAULT_LANDSCAPES)
    algorithms = copy(DEFAULT_ALGORITHMS)
    output_dir = DEFAULT_OUTPUT_DIR
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg == "--seeds"
            i < length(args) || error("Missing value for --seeds")
            seed_count = parse(Int, args[i + 1])
            seed_count > 0 || error("--seeds must be positive")
            i += 2
        elseif arg == "--epsilon"
            i < length(args) || error("Missing value for --epsilon")
            epsilons = parse_epsilons(args[i + 1])
            i += 2
        elseif arg == "--datasets"
            i < length(args) || error("Missing value for --datasets")
            landscape_keys = parse_list(args[i + 1])
            isempty(landscape_keys) && error("--datasets must not be empty")
            i += 2
        elseif arg == "--algorithms"
            i < length(args) || error("Missing value for --algorithms")
            algorithms = parse_algorithms(args[i + 1])
            isempty(algorithms) && error("--algorithms must not be empty")
            i += 2
        elseif arg == "--output-dir"
            i < length(args) || error("Missing value for --output-dir")
            output_dir = args[i + 1]
            i += 2
        elseif arg in ("-h", "--help")
            usage()
            exit()
        else
            error("Unknown option: $arg")
        end
    end

    return default_config(
        seed_count=seed_count,
        epsilons=epsilons,
        landscape_keys=landscape_keys,
        algorithms=algorithms,
        output_dir=output_dir,
    )
end

function index_to_bitstring(index::Integer, n_features::Integer)
    n = Int(n_features)
    index = Int(index)
    chars = Vector{Char}(undef, n)

    for bit in 1:n
        chars[n - bit + 1] = iszero(index & (1 << (bit - 1))) ? '0' : '1'
    end

    return String(chars)
end

function values_for_indices(landscape::Landscape, indices::AbstractVector{<:Integer}, epsilon::Real)
    return [
        IT3708Project3.candidate_state(landscape, index, epsilon).penalized_fitness
        for index in indices
    ]
end

function csv_value(value)
    if ismissing(value) || isnothing(value)
        return ""
    end

    text = string(value)
    if occursin('"', text) || occursin(',', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv_rows(path::AbstractString, header, rows, row_values::Function)
    mkpath(dirname(path))

    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csv_value.(row_values(row)), ','))
        end
    end

    return path
end

function parameters_string(algorithm::Symbol, config::ExperimentConfig)
    if algorithm == :ga
        pm = isnothing(config.ga_mutation_probability) ? "1/n" : string(config.ga_mutation_probability)
        return join((
            "iterations=$(config.ga_iterations)",
            "population_size=$(config.ga_population_size)",
            "pc=$(config.ga_crossover_probability)",
            "pm=$pm",
            "tournament_size=$(config.ga_tournament_size)",
            "survivor_mode=$(config.ga_survivor_mode)",
            "elite=$(config.ga_elite)",
        ), ';')
    elseif algorithm == :nsga2
        pm = isnothing(config.nsga2_mutation_probability) ? "1/n" : string(config.nsga2_mutation_probability)
        return join((
            "iterations=$(config.nsga2_iterations)",
            "population_size=$(config.nsga2_population_size)",
            "pc=$(config.nsga2_crossover_probability)",
            "pm=$pm",
        ), ';')
    elseif algorithm == :swarm
        return join((
            "iterations=$(config.swarm_iterations)",
            "swarm_size=$(config.swarm_size)",
            "w=$(config.swarm_w)",
            "c1=$(config.swarm_c1)",
            "c2=$(config.swarm_c2)",
        ), ';')
    end

    error("Unknown algorithm: $algorithm")
end

function make_generation_stat(algorithm::Symbol,
                              landscape::Landscape,
                              seed::Int,
                              epsilon::Float64,
                              generation::Int,
                              values::AbstractVector{<:Real},
                              best_so_far::Real)
    return GenerationStat(
        algorithm,
        landscape.name,
        seed,
        epsilon,
        generation,
        minimum(values),
        mean(values),
        maximum(values),
        Float64(best_so_far),
    )
end

function ga_generation_stats(landscape::Landscape, result, epsilon::Float64, seed::Int)
    rows = GenerationStat[]

    for i in eachindex(result.min_history)
        push!(
            rows,
            GenerationStat(
                :ga,
                landscape.name,
                seed,
                epsilon,
                i - 1,
                Float64(result.min_history[i]),
                Float64(result.mean_history[i]),
                Float64(result.max_history[i]),
                Float64(result.best_history[i]),
            ),
        )
    end

    return rows
end

function nsga2_generation_stats(landscape::Landscape, result, epsilon::Float64, seed::Int)
    rows = GenerationStat[]
    best_so_far = -Inf

    for i in eachindex(result.population_indices_history)
        values = values_for_indices(landscape, result.population_indices_history[i], epsilon)
        best_so_far = max(best_so_far, maximum(values))
        push!(rows, make_generation_stat(:nsga2, landscape, seed, epsilon, i - 1, values, best_so_far))
    end

    return rows
end

function swarm_generation_stats(landscape::Landscape, result, epsilon::Float64, seed::Int)
    rows = GenerationStat[]

    for i in eachindex(result.particle_index_history)
        values = values_for_indices(landscape, result.particle_index_history[i], epsilon)
        best_so_far = result.best_penalized_fitness_history[i]
        push!(rows, make_generation_stat(:swarm, landscape, seed, epsilon, i - 1, values, best_so_far))
    end

    return rows
end

function ga_run_row(landscape::Landscape, result, epsilon::Float64, seed::Int, runtime::Float64, config::ExperimentConfig)
    state = IT3708Project3.candidate_state(landscape, result.best_index, epsilon)
    evaluations = config.ga_population_size * (1 + 2 * config.ga_iterations)

    return ExperimentRun(
        :ga,
        landscape.name,
        seed,
        epsilon,
        result.best_index,
        index_to_bitstring(result.best_index, landscape.num_features),
        state.accuracy,
        state.num_selected,
        state.penalized_fitness,
        evaluations,
        runtime,
        parameters_string(:ga, config),
    )
end

function nsga2_run_row(landscape::Landscape, result, epsilon::Float64, seed::Int, runtime::Float64, config::ExperimentConfig)
    state = IT3708Project3.candidate_state(landscape, result.best_penalized_index, epsilon)

    return ExperimentRun(
        :nsga2,
        landscape.name,
        seed,
        epsilon,
        result.best_penalized_index,
        index_to_bitstring(result.best_penalized_index, landscape.num_features),
        state.accuracy,
        state.num_selected,
        state.penalized_fitness,
        result.evaluations,
        runtime,
        parameters_string(:nsga2, config),
    )
end

function swarm_run_row(landscape::Landscape, result, epsilon::Float64, seed::Int, runtime::Float64, config::ExperimentConfig)
    state = IT3708Project3.candidate_state(landscape, result.best_index, epsilon)

    return ExperimentRun(
        :swarm,
        landscape.name,
        seed,
        epsilon,
        result.best_index,
        index_to_bitstring(result.best_index, landscape.num_features),
        state.accuracy,
        state.num_selected,
        state.penalized_fitness,
        result.evaluations,
        runtime,
        parameters_string(:swarm, config),
    )
end

function run_one_experiment(landscape::Landscape,
                            algorithm::Symbol,
                            epsilon::Float64,
                            seed::Int,
                            config::ExperimentConfig)
    rng = MersenneTwister(seed)

    if algorithm == :ga
        result_ref = Ref{Any}()
        runtime = @elapsed begin
            result_ref[] = run_single_objective_ea(
                landscape;
                iterations=config.ga_iterations,
                epsilon=epsilon,
                population_size=config.ga_population_size,
                crossover_probability=config.ga_crossover_probability,
                mutation_probability=config.ga_mutation_probability,
                tournament_size=config.ga_tournament_size,
                survivor_mode=config.ga_survivor_mode,
                elite=config.ga_elite,
                rng=rng,
                keep_history=true,
            )
        end
        result = result_ref[]
        return ga_run_row(landscape, result, epsilon, seed, runtime, config),
               ga_generation_stats(landscape, result, epsilon, seed)
    elseif algorithm == :nsga2
        result_ref = Ref{Any}()
        runtime = @elapsed begin
            result_ref[] = run_nsga2_feature_ea(
                landscape;
                iterations=config.nsga2_iterations,
                epsilon=epsilon,
                population_size=config.nsga2_population_size,
                crossover_probability=config.nsga2_crossover_probability,
                mutation_probability=config.nsga2_mutation_probability,
                rng=rng,
                keep_history=true,
            )
        end
        result = result_ref[]
        return nsga2_run_row(landscape, result, epsilon, seed, runtime, config),
               nsga2_generation_stats(landscape, result, epsilon, seed)
    elseif algorithm == :swarm
        result_ref = Ref{Any}()
        runtime = @elapsed begin
            result_ref[] = run_swarm_ea(
                landscape;
                iterations=config.swarm_iterations,
                epsilon=epsilon,
                swarm_size=config.swarm_size,
                w=config.swarm_w,
                c1=config.swarm_c1,
                c2=config.swarm_c2,
                rng=rng,
                keep_history=true,
            )
        end
        result = result_ref[]
        return swarm_run_row(landscape, result, epsilon, seed, runtime, config),
               swarm_generation_stats(landscape, result, epsilon, seed)
    end

    error("Unknown algorithm: $algorithm")
end

function run_repeated_experiments(landscape::Landscape,
                                  algorithm::Symbol,
                                  epsilon::Float64,
                                  seeds::AbstractVector{<:Integer},
                                  config::ExperimentConfig)
    run_rows = ExperimentRun[]
    generation_rows = GenerationStat[]

    for seed in seeds
        println("Running $(algorithm) on $(landscape.name), epsilon=$(epsilon), seed=$(seed)")
        run_row, run_generation_rows = run_one_experiment(landscape, algorithm, epsilon, Int(seed), config)
        push!(run_rows, run_row)
        append!(generation_rows, run_generation_rows)
    end

    return run_rows, generation_rows
end

function run_algorithm_suite(landscape::Landscape,
                             algorithms::AbstractVector{Symbol},
                             epsilon::Float64,
                             seeds::AbstractVector{<:Integer},
                             config::ExperimentConfig)
    run_rows = ExperimentRun[]
    generation_rows = GenerationStat[]

    for algorithm in algorithms
        algorithm_run_rows, algorithm_generation_rows =
            run_repeated_experiments(landscape, algorithm, epsilon, seeds, config)
        append!(run_rows, algorithm_run_rows)
        append!(generation_rows, algorithm_generation_rows)
    end

    return run_rows, generation_rows
end

function sample_std(values)
    length(values) <= 1 && return 0.0
    return std(values)
end

function summarize_runs(run_rows::Vector{ExperimentRun})
    groups = Dict{Tuple{Symbol, String, Float64}, Vector{ExperimentRun}}()

    for row in run_rows
        key = (row.algorithm, row.landscape, row.epsilon)
        push!(get!(groups, key, ExperimentRun[]), row)
    end

    summaries = ExperimentSummary[]
    for key in sort(collect(keys(groups)); by = key -> (String(key[2]), String(key[1]), key[3]))
        rows = groups[key]
        fitnesses = [row.penalized_fitness for row in rows]
        best = rows[argmax(fitnesses)]

        push!(
            summaries,
            ExperimentSummary(
                key[1],
                key[2],
                key[3],
                length(rows),
                mean(fitnesses),
                sample_std(fitnesses),
                minimum(fitnesses),
                maximum(fitnesses),
                mean(row.accuracy for row in rows),
                mean(row.num_selected for row in rows),
                mean(row.runtime for row in rows),
                best.best_index,
                best.best_bitstring,
                best.accuracy,
                best.num_selected,
                best.penalized_fitness,
            ),
        )
    end

    return summaries
end

function write_raw_runs_csv(rows::Vector{ExperimentRun}, path::AbstractString)
    header = [
        "algorithm", "landscape", "seed", "epsilon", "best_index", "best_bitstring",
        "accuracy", "num_selected", "penalized_fitness", "evaluations", "runtime", "parameters",
    ]

    return write_csv_rows(
        path,
        header,
        rows,
        row -> (
            row.algorithm, row.landscape, row.seed, row.epsilon, row.best_index,
            row.best_bitstring, row.accuracy, row.num_selected, row.penalized_fitness,
            row.evaluations, row.runtime, row.parameters,
        ),
    )
end

function write_generation_stats_csv(rows::Vector{GenerationStat}, path::AbstractString)
    header = [
        "algorithm", "landscape", "seed", "epsilon", "generation",
        "min_fitness", "mean_fitness", "max_fitness", "best_so_far_fitness",
    ]

    return write_csv_rows(
        path,
        header,
        rows,
        row -> (
            row.algorithm, row.landscape, row.seed, row.epsilon, row.generation,
            row.min_fitness, row.mean_fitness, row.max_fitness, row.best_so_far_fitness,
        ),
    )
end

function write_summary_csv(rows::Vector{ExperimentSummary}, path::AbstractString)
    header = [
        "algorithm", "landscape", "epsilon", "runs", "mean_best_fitness",
        "std_best_fitness", "min_best_fitness", "max_best_fitness",
        "mean_accuracy", "mean_num_selected", "mean_runtime", "best_index",
        "best_bitstring", "best_accuracy", "best_num_selected", "best_fitness",
    ]

    return write_csv_rows(
        path,
        header,
        rows,
        row -> (
            row.algorithm, row.landscape, row.epsilon, row.runs,
            row.mean_best_fitness, row.std_best_fitness, row.min_best_fitness,
            row.max_best_fitness, row.mean_accuracy, row.mean_num_selected,
            row.mean_runtime, row.best_index, row.best_bitstring, row.best_accuracy,
            row.best_num_selected, row.best_fitness,
        ),
    )
end

function run_full_experiment_suite(config::ExperimentConfig)
    run_rows = ExperimentRun[]
    generation_rows = GenerationStat[]

    for landscape_key in config.landscape_keys
        println("Loading landscape $(landscape_key)")
        landscape = load_landscape_key(landscape_key)

        for epsilon in config.epsilons
            suite_run_rows, suite_generation_rows =
                run_algorithm_suite(landscape, config.algorithms, epsilon, config.seeds, config)
            append!(run_rows, suite_run_rows)
            append!(generation_rows, suite_generation_rows)
        end
    end

    summary_rows = summarize_runs(run_rows)
    raw_path = joinpath(config.output_dir, "raw_runs.csv")
    generation_path = joinpath(config.output_dir, "generation_stats.csv")
    summary_path = joinpath(config.output_dir, "summary.csv")

    write_raw_runs_csv(run_rows, raw_path)
    write_generation_stats_csv(generation_rows, generation_path)
    write_summary_csv(summary_rows, summary_path)

    println("Saved raw run results to $(raw_path)")
    println("Saved generation statistics to $(generation_path)")
    println("Saved summary statistics to $(summary_path)")

    return run_rows, generation_rows, summary_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_cli(ARGS)
    println("Experiment configuration")
    println("  landscapes: $(join(config.landscape_keys, ", "))")
    println("  algorithms: $(join(string.(config.algorithms), ", "))")
    println("  epsilons: $(join(config.epsilons, ", "))")
    println("  seeds: $(first(config.seeds)):$(last(config.seeds))")
    println("  output dir: $(config.output_dir)")
    run_full_experiment_suite(config)
end
