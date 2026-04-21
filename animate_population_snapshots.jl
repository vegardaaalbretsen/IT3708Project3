using IT3708Project3
using Random
using Plots

include("plot_population_snapshots.jl")

const DEFAULT_OUTPUT_DIR = joinpath("exports", "plots", "population_snapshot_animations")
const ALLOWED_PLOT_KINDS = Set(["both", "feature-count", "hbm"])
const ALLOWED_ALGORITHMS = Set(["ga", "nsga2", "swarm"])

function animation_usage()
    println("Usage: julia --project=. animate_population_snapshots.jl <dataset-key|triangle> [iterations] [epsilon] [options]")
    println("")
    println("Options:")
    println("  --algorithm ALG         ga, nsga2, or swarm, default swarm")
    println("  --seed N                Random seed for the run")
    println("  --plot both|feature-count|hbm")
    println("  --fps N                 GIF frames per second, default 8")
    println("  --final-hold N          Extra copies of the final frame, default 12")
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
    println("  -h, --help              Show this help")
    println("")
    println("Examples:")
    println("  julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm swarm --seed 1 --plot feature-count")
    println("  julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm ga --seed 1 --plot feature-count")
    println("  julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm nsga2 --seed 1 --plot feature-count")
    println("  julia --project=. animate_population_snapshots.jl breast-w 150 0.01 --algorithm ga --seed 1 --popsize 100 --pc 0.95 --plot hbm")
end

Base.@kwdef mutable struct AnimationCLIState
    positional::Vector{String} = String[]
    algorithm::String = "swarm"
    seed::Union{Nothing, Int} = nothing
    plot_kind::String = "both"
    fps::Int = 8
    final_hold_frames::Int = 12
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

function require_option_value(args::Vector{String}, index::Integer, option::AbstractString)
    index < length(args) || error("Missing value for $(option)")
    return args[index + 1], index + 2
end

function require_choice(value, allowed, option::AbstractString)
    value in allowed || error("$(option) must be one of: $(join(sort(collect(allowed)), ", "))")
    return value
end

function parse_animation_cli(args::Vector{String})
    state = AnimationCLIState()
    survivor_modes = Set((:elitist, :generational))
    option_handlers = Dict{String, Function}(
        "--algorithm" => value -> (state.algorithm = require_choice(value, ALLOWED_ALGORITHMS, "--algorithm")),
        "--seed" => value -> (state.seed = parse(Int, value)),
        "--plot" => value -> (state.plot_kind = require_choice(value, ALLOWED_PLOT_KINDS, "--plot")),
        "--fps" => value -> (state.fps = parse(Int, value)),
        "--final-hold" => value -> (state.final_hold_frames = parse(Int, value)),
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
            animation_usage()
            exit()
        elseif startswith(arg, "--")
            handler = get(option_handlers, arg, nothing)
            isnothing(handler) && error("Unknown option: $arg")
            value, i = require_option_value(args, i, arg)
            handler(value)
        else
            push!(state.positional, arg)
            i += 1
        end
    end

    length(state.positional) <= 3 || error("Too many positional arguments")

    dataset_key = length(state.positional) >= 1 ? state.positional[1] : "breast-w"
    iterations = length(state.positional) >= 2 ? parse(Int, state.positional[2]) : 150
    epsilon = length(state.positional) >= 3 ? parse(Float64, state.positional[3]) : 0.0

    state.fps > 0 || error("--fps must be positive")
    state.final_hold_frames >= 0 || error("--final-hold must be non-negative")
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
        fps = state.fps,
        final_hold_frames = state.final_hold_frames,
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

function safe_animation_part(value)
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

function rows_all_generations(landscape::Landscape,
                              algorithm::AbstractString,
                              population_history;
                              seed::Integer,
                              epsilon::Real)
    isnothing(population_history) &&
        throw(ArgumentError("Population history is missing. Run the algorithm with keep_history=true before animating snapshots."))

    rows = SnapshotRow[]

    for (position, indices) in pairs(population_history)
        generation = position - 1

        for (index, count) in grouped_index_counts(indices)
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
                    generation,
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
    end

    return rows
end

function append_hold_frames!(anim::Plots.Animation, plt, hold_frames::Integer)
    hold = max(Int(hold_frames), 0)

    for _ in 1:hold
        Plots.frame(anim, plt)
    end

    return anim
end

function parameter_tag(cli)
    if cli.algorithm == "ga"
        pm_tag = isnothing(cli.mutation_probability) ? "pm-auto" : "pm-$(safe_animation_part(cli.mutation_probability))"
        return join(
            [
                "pop$(cli.population_size)",
                "pc-$(safe_animation_part(cli.crossover_probability))",
                pm_tag,
                "tour$(cli.tournament_size)",
                "survivor-$(cli.survivor_mode)",
                "elite$(cli.elite)",
            ],
            "_",
        )
    elseif cli.algorithm == "nsga2"
        pm_tag = isnothing(cli.mutation_probability) ? "pm-auto" : "pm-$(safe_animation_part(cli.mutation_probability))"
        return join(
            [
                "pop$(cli.population_size)",
                "pc-$(safe_animation_part(cli.crossover_probability))",
                pm_tag,
            ],
            "_",
        )
    end

    return join(
        [
            "swarm$(cli.swarm_size)",
            "w$(safe_animation_part(cli.w))",
            "c1-$(safe_animation_part(cli.c1))",
            "c2-$(safe_animation_part(cli.c2))",
        ],
        "_",
    )
end

function snapshot_animation_filename(landscape_name::AbstractString,
                                     algorithm::AbstractString,
                                     plot_kind::AbstractString,
                                     seed::Integer,
                                     epsilon::Real,
                                     cli)
    return join(
        [
            safe_animation_part(landscape_name),
            safe_animation_part(algorithm),
            "snapshots",
            safe_animation_part(plot_kind),
            "seed$(seed)",
            "epsilon-$(safe_animation_part(epsilon))",
            parameter_tag(cli),
        ],
        "_",
    ) * ".gif"
end

function save_feature_count_snapshot_animation(landscape::Landscape,
                                               rows::Vector{SnapshotRow},
                                               output_path::AbstractString;
                                               algorithm::AbstractString,
                                               seed::Integer,
                                               epsilon::Real,
                                               fps::Integer = 8,
                                               final_hold_frames::Integer = 12)
    anim = Plots.Animation()
    last_plot = nothing

    for (_, snapshot_rows) in rows_by_generation(rows)
        plt = plot_feature_count_snapshot(
            landscape,
            snapshot_rows;
            algorithm=algorithm,
            seed=seed,
            epsilon=epsilon,
        )
        Plots.frame(anim, plt)
        last_plot = plt
    end

    isnothing(last_plot) || append_hold_frames!(anim, last_plot, final_hold_frames)
    mkpath(dirname(output_path))
    Plots.gif(anim, output_path; fps=Int(fps))
    return output_path
end

function save_hbm_snapshot_animation(landscape::Landscape,
                                     rows::Vector{SnapshotRow},
                                     output_path::AbstractString;
                                     algorithm::AbstractString,
                                     seed::Integer,
                                     epsilon::Real,
                                     fps::Integer = 8,
                                     final_hold_frames::Integer = 12)
    anim = Plots.Animation()
    last_plot = nothing

    for (_, snapshot_rows) in rows_by_generation(rows)
        plt = plot_hbm_snapshot(
            landscape,
            snapshot_rows;
            algorithm=algorithm,
            seed=seed,
            epsilon=epsilon,
        )
        Plots.frame(anim, plt)
        last_plot = plt
    end

    isnothing(last_plot) || append_hold_frames!(anim, last_plot, final_hold_frames)
    mkpath(dirname(output_path))
    Plots.gif(anim, output_path; fps=Int(fps))
    return output_path
end

function run_algorithm(cli, landscape::Landscape, rng::AbstractRNG)
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

function print_run_summary(cli, landscape::Landscape, result, seed_label::AbstractString)
    println("Population snapshot animation on `$(landscape.name)`")
    println("Algorithm: $(cli.algorithm)")
    println("Iterations: $(result.iterations)")
    println("Seed: $(seed_label)")
    println("GIF fps: $(cli.fps), final hold: $(cli.final_hold_frames) frames")

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
    animation_usage()
    exit()
end

cli = parse_animation_cli(ARGS)
seed_value = isnothing(cli.seed) ? 0 : Int(cli.seed)
seed_label = isnothing(cli.seed) ? "random" : string(cli.seed)
rng = isnothing(cli.seed) ? Random.default_rng() : MersenneTwister(cli.seed)
landscape = load_landscape_key(cli.dataset_key)
result, population_history = run_algorithm(cli, landscape, rng)
rows = rows_all_generations(landscape, cli.algorithm, population_history; seed=seed_value, epsilon=cli.epsilon)
saved_paths = String[]

print_run_summary(cli, landscape, result, seed_label)

if cli.plot_kind in ("both", "feature-count")
    output_path = joinpath(
        cli.output_dir,
        safe_animation_part(landscape.name),
        safe_animation_part(cli.algorithm),
        "feature_count",
        snapshot_animation_filename(landscape.name, cli.algorithm, "feature-count", seed_value, cli.epsilon, cli),
    )
    push!(
        saved_paths,
        save_feature_count_snapshot_animation(
            landscape,
            rows,
            output_path;
            algorithm=cli.algorithm,
            seed=seed_value,
            epsilon=cli.epsilon,
            fps=cli.fps,
            final_hold_frames=cli.final_hold_frames,
        ),
    )
end

if cli.plot_kind in ("both", "hbm")
    output_path = joinpath(
        cli.output_dir,
        safe_animation_part(landscape.name),
        safe_animation_part(cli.algorithm),
        "hbm",
        snapshot_animation_filename(landscape.name, cli.algorithm, "hbm", seed_value, cli.epsilon, cli),
    )
    push!(
        saved_paths,
        save_hbm_snapshot_animation(
            landscape,
            rows,
            output_path;
            algorithm=cli.algorithm,
            seed=seed_value,
            epsilon=cli.epsilon,
            fps=cli.fps,
            final_hold_frames=cli.final_hold_frames,
        ),
    )
end

println("Saved $(length(saved_paths)) snapshot GIF(s):")
for path in saved_paths
    println("  $(path)")
end
