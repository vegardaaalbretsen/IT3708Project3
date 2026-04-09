using IT3708Project3

dataset_key = length(ARGS) >= 1 ? ARGS[1] : "credit-a"
output_csv = length(ARGS) >= 2 ? ARGS[2] : default_output_path(dataset_key)

haskey(DATASETS, dataset_key) || error("Unknown dataset key: $dataset_key")

dataset = DATASETS[dataset_key]
data = parse_dataset(dataset.path, dataset.num_features)
write_csv(data, output_csv)

println("Wrote CSV for dataset `$dataset_key` to `$output_csv`.")
