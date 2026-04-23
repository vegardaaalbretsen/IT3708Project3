using IT3708Project3
using Random

include("plot_population_snapshots.jl")

const DEFAULT_OUTPUT_DIR = joinpath("exports", "plots", "population_history_frames")
const ALLOWED_PLOT_KINDS = Set(["both", "feature-count", "hbm"])
const ALLOWED_ALGORITHMS = Set(["ga", "nsga2", "swarm"])

function history_frame_usage()
    println("Usage: julia --project=. plot_population_history_frames.jl <dataset-key|triangle> [iterations] [epsilon] [options]")
    println("")
    println("Options:")
    println("  --algorithm ALG         ga, nsga2, or swarm, default swarm")
    println("  --seed N                Random seed for the run")
    println("  --plot both|feature-count|hbm")
    println("  --generations LIST      Comma-separated generations to render, default 0,25%,50%,75%,final")
    println("  --output-dir PATH       Output directory, default $(DEFAULT_OUTPUT_DIR)")
    println("")
    println("GA options:")
    println("  --popsize N             Population size, default 100")
    println("  --pc V                  Crossover probability, default 0.95")
    println("  --pm V                  Mutation probability, default 1/n")
    println("  --tournament-size N     Tournament size, default 4")
    println("  --survivor-mode MODE    elitist or generational, default elitist")
    println("  --elite N               Number of elites, default 4")
    println("")
    println("Swarm options:")
    println("  --swarm-size N          Number of particles, default 40")
    println("  --w V                   Inertia coefficient, default 0.7")
    println("  --c1 V                  Cognitive coefficient, default 1.4")
    println("  --c2 V                  Social coefficient, default 1.4")
    println("")
    println("Examples:")
    println("  julia --project=. plot_population_history_frames.jl breast-w 100 0.01 --algorithm swarm --seed 1 --plot hbm")
    println("  julia --project=. plot_population_history_frames.jl breast-w 100 0.01 --algorithm swarm --seed 1 --plot hbm --generations 0,10,25,50,100")
end

Base.@kwdef mutable struct HistoryFrameCLIState
    positional::Vector{String} = String[]
    algorithm::String = "swarm"
    seed::Union{Nothing, Int} = nothing
    plot_kind::String = "both"
    generations::Union{Nothing, String} = nothing
    output_dir::String = DEFAULT_OUTPUT_DIR
    population_size::Int = 100
    crossover_probability::Float64 = 0.95
    mutation_probability::Union{Nothing, Float64} = nothing
    tournament_size::Int = 4
    survivor_mode::Symbol = :elitist
    elite::Int = 4
    swarm_size::Int = 40
    w::Float64 = 0.7
    c1::Float64 = 1.4
    c2::Float64 = 1.4
end

function require_history_option_value(args::Vector{String}, index::Integer, option::AbstractString)
    index < length(args) || error("Missing value for $(option)")
    return args[index + 1], index + 2
end

function require_history_choice(value, allowed, option::AbstractString)
    value in allowed || error("$(option) must be one of: $(join(sort(collect(allowed)), ", "))")
    return value
end

function parse_generation_list(text::AbstractString)
    values = Int[]

    for part in split(text, ',')
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, parse(Int, stripped))
    end

    isempty(values) && error("--generations must contain at least one integer")
    return values
end

function default_generation_plan(final_generation::Integer)
    final = max(Int(final_generation), 0)
    candidates = [
        0,
        floor(Int, final / 4),
        floor(Int, final / 2),
        floor(Int, 3 * final / 4),
        final,
    ]
    selected = Int[]
    seen = Set{Int}()

    for generation in candidates
        if !(generation in seen)
            push!(selected, generation)
            push!(seen, generation)
        end
    end

    return selected
end

function parse_history_frame_cli(args::Vector{String})
    state = HistoryFrameCLIState()
    survivor_modes = Set((:elitist, :generational))
    option_handlers = Dict{String, Function}(
        "--algorithm" => value -> (state.algorithm = require_history_choice(value, ALLOWED_ALGORITHMS, "--algorithm")),
        "--seed" => value -> (state.seed = parse(Int, value)),
        "--plot" => value -> (state.plot_kind = require_history_choice(value, ALLOWED_PLOT_KINDS, "--plot")),
        "--generations" => value -> (state.generations = value),
        "--output-dir" => value -> (state.output_dir = value),
        "--popsize" => value -> (state.population_size = parse(Int, value)),
        "--pc" => value -> (state.crossover_probability = parse(Float64, value)),
        "--pm" => value -> (state.mutation_probability = parse(Float64, value)),
        "--tournament-size" => value -> (state.tournament_size = parse(Int, value)),
        "--survivor-mode" => value -> begin
            mode = Symbol(value)
            mode in survivor_modes || error("--survivor-mode must be one of: elitist, generational")
            state.survivor_mode = mode
        end,
        "--elite" => value -> (state.elite = parse(Int, value)),
        "--swarm-size" => value -> (state.swarm_size = parse(Int, value)),
        "--w" => value -> (state.w = parse(Float64, value)),
        "--c1" => value -> (state.c1 = parse(Float64, value)),
        "--c2" => value -> (state.c2 = parse(Float64, value)),
    )
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg in ("-h", "--help")
            history_frame_usage()
            exit()
        elseif startswith(arg, "--")
            handler = get(option_handlers, arg, nothing)
            isnothing(handler) && error("Unknown option: $arg")
            value, i = require_history_option_value(args, i, arg)
            handler(value)
        else
            push!(state.positional, arg)
            i += 1
        end
    end

    length(state.positional) <= 3 || error("Too many positional arguments")

    dataset_key = length(state.positional) >= 1 ? state.positional[1] : "breast-w"
    iterations = length(state.positional) >= 2 ? parse(Int, state.positional[2]) : 100
    epsilon = length(state.positional) >= 3 ? parse(Float64, state.positional[3]) : 0.0

    iterations >= 0 || error("iterations must be non-negative")
    state.population_size > 0 || error("--popsize must be positive")
    state.swarm_size > 0 || error("--swarm-size must be positive")

    return (
        dataset_key = dataset_key,
        iterations = iterations,
        epsilon = epsilon,
        algorithm = state.algorithm,
        seed = state.seed,
        plot_kind = state.plot_kind,
        generations = isnothing(state.generations) ? nothing : parse_generation_list(state.generations),
        output_dir = state.output_dir,
        population_size = state.population_size,
        crossover_probability = state.crossover_probability,
        mutation_probability = state.mutation_probability,
        tournament_size = state.tournament_size,
        survivor_mode = state.survivor_mode,
        elite = state.elite,
        swarm_size = state.swarm_size,
        w = state.w,
        c1 = state.c1,
        c2 = state.c2,
    )
end

function history_safe_part(value)
    return replace(string(value), r"[^A-Za-z0-9_.-]" => "_")
end

function index_to_bitstring(index::Integer, n_features::Integer)
    n = Int(n_features)
    bits = Vector{Char}(undef, n)

    for bit in 1:n
        bits[n - bit + 1] = iszero(Int(index) & (1 << (bit - 1))) ? '0' : '1'
    end

    return String(bits)
end

function grouped_index_counts(indices::AbstractVector{<:Integer})
    counts = Dict{Int, Int}()

    for index in indices
        counts[Int(index)] = get(counts, Int(index), 0) + 1
    end

    return [(index, counts[index]) for index in sort!(collect(keys(counts)))]
end

function rows_for_generation(landscape::Landscape,
                             algorithm::AbstractString,
                             population_history,
                             generation::Integer;
                             seed::Integer,
                             epsilon::Real)
    isnothing(population_history) &&
        throw(ArgumentError("Population history is missing. Run the algorithm with keep_history=true before plotting frames."))

    position = Int(generation) + 1
    1 <= position <= length(population_history) ||
        throw(ArgumentError("Generation $(generation) is out of range 0:$(length(population_history) - 1)"))

    rows = SnapshotRow[]
    for (index, count) in grouped_index_counts(population_history[position])
        state = IT3708Project3.candidate_state(landscape, index, epsilon)
        push!(
            rows,
            SnapshotRow(
                algorithm,
                landscape.name,
                Int(seed),
                Float64(epsilon),
                position,
                "generation_$(generation)",
                Int(generation),
                index,
                index_to_bitstring(index, landscape.num_features),
                count,
                state.num_selected,
                state.accuracy,
                state.time,
                state.penalized_fitness,
            ),
        )
    end

    return rows
end

function run_history_algorithm(cli, landscape::Landscape, rng::AbstractRNG)
    if cli.algorithm == "ga"
        result = run_single_objective_ea(
            landscape;
            iterations=cli.iterations,
            epsilon=cli.epsilon,
            population_size=cli.population_size,
            crossover_probability=cli.crossover_probability,
            mutation_probability=cli.mutation_probability,
            tournament_size=cli.tournament_size,
            survivor_mode=cli.survivor_mode,
            elite=cli.elite,
            keep_history=true,
            rng=rng,
        )
        return result, result.population_indices_history
    end

    if cli.algorithm == "nsga2"
        result = run_nsga2_feature_ea(
            landscape;
            iterations=cli.iterations,
            epsilon=cli.epsilon,
            population_size=cli.population_size,
            crossover_probability=cli.crossover_probability,
            mutation_probability=cli.mutation_probability,
            keep_history=true,
            rng=rng,
        )
        return result, result.population_indices_history
    end

    result = run_swarm_ea(
        landscape;
        iterations=cli.iterations,
        epsilon=cli.epsilon,
        swarm_size=cli.swarm_size,
        w=cli.w,
        c1=cli.c1,
        c2=cli.c2,
        keep_history=true,
        rng=rng,
    )
    return result, result.particle_index_history
end

function frame_output_path(output_dir::AbstractString,
                           landscape::AbstractString,
                           algorithm::AbstractString,
                           plot_kind::AbstractString,
                           generation::Integer)
    plot_dir = plot_kind == "feature-count" ? "feature_count" : history_safe_part(plot_kind)
    filename = "gen$(Int(generation)).png"
    return joinpath(
        output_dir,
        history_safe_part(landscape),
        history_safe_part(algorithm),
        plot_dir,
        filename,
    )
end

function print_history_summary(cli, landscape::Landscape, result, generations)
    println("Population history frames on `$(landscape.name)`")
    println("Algorithm: $(cli.algorithm)")
    println("Iterations: $(result.iterations)")
    println("Seed: $(isnothing(cli.seed) ? "random" : string(cli.seed))")
    println("Generations: $(join(generations, ", "))")

    if cli.algorithm == "ga"
        println("Population size: $(result.population_size)")
        println("Parameters: pc=$(result.crossover_probability), pm=$(result.mutation_probability), tournament=$(result.tournament_size), survivor=$(result.survivor_mode), elite=$(result.elite)")
    elseif cli.algorithm == "nsga2"
        println("Population size: $(result.population_size)")
        println("Parameters: pc=$(result.crossover_probability), pm=$(result.mutation_probability)")
    else
        println("Swarm size: $(result.swarm_size)")
        println("Parameters: w=$(result.w), c1=$(result.c1), c2=$(result.c2)")
    end
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    history_frame_usage()
    exit()
end

cli = parse_history_frame_cli(ARGS)
seed_value = isnothing(cli.seed) ? 0 : Int(cli.seed)
rng = isnothing(cli.seed) ? Random.default_rng() : MersenneTwister(cli.seed)
landscape = load_landscape_key(cli.dataset_key)
result, population_history = run_history_algorithm(cli, landscape, rng)
final_generation = length(population_history) - 1
generations = isnothing(cli.generations) ? default_generation_plan(final_generation) : sort(unique(cli.generations))
saved_paths = String[]

print_history_summary(cli, landscape, result, generations)

for generation in generations
    rows = rows_for_generation(
        landscape,
        cli.algorithm,
        population_history,
        generation;
        seed=seed_value,
        epsilon=cli.epsilon,
    )

    if cli.plot_kind in ("both", "feature-count")
        output_path = frame_output_path(cli.output_dir, landscape.name, cli.algorithm, "feature-count", generation)
        push!(
            saved_paths,
            save_feature_count_snapshot_plot(
                landscape,
                rows,
                output_path;
                algorithm=cli.algorithm,
                seed=seed_value,
                epsilon=cli.epsilon,
            ),
        )
    end

    if cli.plot_kind in ("both", "hbm")
        output_path = frame_output_path(cli.output_dir, landscape.name, cli.algorithm, "hbm", generation)
        push!(
            saved_paths,
            save_hbm_snapshot_plot(
                landscape,
                rows,
                output_path;
                algorithm=cli.algorithm,
                seed=seed_value,
                epsilon=cli.epsilon,
            ),
        )
    end
end

println("Saved $(length(saved_paths)) plot(s):")
for path in saved_paths
    println("  $(path)")
end
