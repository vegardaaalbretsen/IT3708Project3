using IT3708Project3

function print_usage()
    println("Usage: julia --project=. export_csv.jl [dataset_key] [output_csv]")
    println("")
    println("Defaults:")
    println("  dataset_key = credit-a")
    println("  output_csv = <dataset_key>_landscape.csv")
    println("")
    println("Available dataset keys: $(join(sort(collect(keys(DATASET_SPECS))), ", "))")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    dataset_key = length(args) >= 1 ? args[1] : "credit-a"
    output_csv = length(args) >= 2 ? args[2] : "$(dataset_key)_landscape.csv"

    return dataset_key, output_csv
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    dataset_key, output_csv = parsed
    haskey(DATASET_SPECS, dataset_key) || error("Unknown dataset key: $dataset_key")

    spec = DATASET_SPECS[dataset_key]
    landscape = read_feature_selection_landscape(spec.path, spec.n_features)
    write_feature_selection_csv(landscape, output_csv)

    println("Wrote CSV for dataset `$dataset_key` to `$output_csv`.")
end
