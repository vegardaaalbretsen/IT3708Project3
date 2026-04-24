using IT3708Project3

function usage()
    println("Usage: julia --project=. parse_landscape.jl <dataset-key> [output-csv]")
    println("")
    println("Available dataset keys: breast-w, credit-a, letter-r, zoo, hepatitis")
    println("")
    println("Examples:")
    println("  julia --project=. parse_landscape.jl breast-w")
    println("  julia --project=. parse_landscape.jl zoo")
    println("  julia --project=. parse_landscape.jl credit-a exports/csv/credit-a_metrics.csv")
end

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "credit-a"
haskey(DATASETS, dataset_key) || error("Unknown dataset key: $dataset_key")

output_csv = length(ARGS) >= 2 ? ARGS[2] : default_output_path(dataset_key)
dataset = DATASETS[dataset_key]

landscape = parse_dataset(dataset.path, dataset.num_features; name=dataset_key)
write_csv(landscape, output_csv)

println("Wrote CSV for dataset `$dataset_key` to `$output_csv`.")
