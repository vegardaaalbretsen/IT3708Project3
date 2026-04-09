using IT3708Project3

function print_usage()
    println("Usage: julia --project=. main.jl [dataset_key] [n] [m] [s]")
    println("")
    println("Defaults:")
    println("  dataset_key = credit-a")
    println("  n = 16")
    println("  m = 1")
    println("  s = 4")
    println("")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "credit-a"
    n = length(args) >= 2 ? parse(Int, args[2]) : 16
    m = length(args) >= 3 ? parse(Int, args[3]) : 1
    s = length(args) >= 4 ? parse(Int, args[4]) : 4

    return dataset_key, n, m, s
end

function summarize_real_landscape(dataset_key)
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")

    spec = DATASET_SPECS[dataset_key]
    landscape = read_feature_selection_landscape(spec.path, spec.n_features)
    best_raw = best_raw_subset(landscape)
    best_penalized = best_penalized_subset(landscape)

    println("Real dataset: $dataset_key")
    println("  file: $(spec.path)")
    println("  features: $(spec.n_features)")
    println("  subsets: $(length(landscape.subset_indices))")
    println("  epsilon: $(landscape.epsilon)")
    println("  first 5 raw accuracies: $(landscape.raw_accuracy_table[1:5])")
    println("  first 5 mean times: $(landscape.raw_time_table[1:5])")
    println("  first 5 penalized fitness values: $(landscape.penalized_table[1:5])")
    println("  best raw subset: $(best_raw.subset_index) -> columns $(best_raw.active_columns)")
    println("    raw accuracy: $(best_raw.raw_accuracy)")
    println("    mean time: $(best_raw.mean_time)")
    println("  best penalized subset: $(best_penalized.subset_index) -> columns $(best_penalized.active_columns)")
    println("    raw accuracy: $(best_penalized.raw_accuracy)")
    println("    penalty: $(best_penalized.penalty)")
    println("    penalized fitness: $(best_penalized.penalized_fitness)")
end

function summarize_triangle_landscape(n, m, s)
    triangle = triangle_landscape(n, m, s)

    println("Triangle landscape")
    println("  n: $n")
    println("  m: $m")
    println("  s: $s")
    println("  states: $(length(triangle.subset_indices))")
    println("  first 10 fitness values: $(triangle.fitness[1:10])")
    println("  max fitness: $(maximum(triangle.fitness))")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, n, m, s = parsed
    summarize_real_landscape(dataset_key)
    println("")
    summarize_triangle_landscape(n, m, s)
end
