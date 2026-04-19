const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW_DATA_DIR = joinpath(PROJECT_ROOT, "data", "raw")
const CSV_EXPORT_DIR = joinpath(PROJECT_ROOT, "exports", "csv")
const PLOT_EXPORT_DIR = joinpath(PROJECT_ROOT, "exports", "plots")
const HBM_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "hbm")
const FEATURE_COUNT_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "feature_count")
const EA_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "ea")
const STN_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "stn")

const DATASETS = Dict(
    "breast-w" => (
        path = joinpath(RAW_DATA_DIR, "01-breast-w_lr_F.h5"),
        num_features = 9,
    ),
    "credit-a" => (
        path = joinpath(RAW_DATA_DIR, "05-credit-a_rf_F.h5"),
        num_features = 15,
    ),
    "letter-r" => (
        path = joinpath(RAW_DATA_DIR, "08-letter-r_knn_F.h5"),
        num_features = 16,
    ),
)

function default_output_path(dataset_key::AbstractString)
    return joinpath(CSV_EXPORT_DIR, "$(dataset_key)_metrics.csv")
end

function dataset_plot_metadata(dataset_key::AbstractString)
    if dataset_key == "triangle"
        return (
            dataset = "triangle",
            model = "synthetic",
            variant = nothing,
            slug = "triangle_synthetic",
            label = "triangle (model=synthetic)",
        )
    end

    haskey(DATASETS, dataset_key) || throw(ArgumentError("Unknown dataset key: $dataset_key"))
    basename_parts = split(splitext(basename(DATASETS[dataset_key].path))[1], '_')
    length(basename_parts) >= 2 || throw(ArgumentError("Could not extract model information from dataset path for `$dataset_key`"))

    model = basename_parts[2]
    variant = length(basename_parts) >= 3 ? join(basename_parts[3:end], "_") : nothing
    slug = isnothing(variant) ? "$(dataset_key)_$(model)" : "$(dataset_key)_$(model)_$(variant)"
    label = isnothing(variant) ?
        "$(dataset_key) (model=$(model))" :
        "$(dataset_key) (model=$(model), variant=$(variant))"

    return (
        dataset = dataset_key,
        model = model,
        variant = variant,
        slug = slug,
        label = label,
    )
end

function default_plot_dir(base_dir::AbstractString; dataset_key::Union{Nothing, AbstractString} = nothing)
    if isnothing(dataset_key)
        return base_dir
    end

    metadata = dataset_plot_metadata(dataset_key)
    parts = String[base_dir, metadata.dataset, metadata.model]
    !isnothing(metadata.variant) && push!(parts, metadata.variant)
    return joinpath(parts...)
end

function default_hbm_plot_path(name::AbstractString; dataset_key::Union{Nothing, AbstractString} = nothing)
    return joinpath(default_plot_dir(HBM_PLOT_EXPORT_DIR; dataset_key=dataset_key), "$(name)_hbm.png")
end

function default_feature_count_plot_path(name::AbstractString; dataset_key::Union{Nothing, AbstractString} = nothing)
    return joinpath(default_plot_dir(FEATURE_COUNT_PLOT_EXPORT_DIR; dataset_key=dataset_key), "$(name).png")
end

function default_ea_plot_path(name::AbstractString; dataset_key::Union{Nothing, AbstractString} = nothing)
    return joinpath(default_plot_dir(EA_PLOT_EXPORT_DIR; dataset_key=dataset_key), "$(name).png")
end

function default_stn_plot_path(name::AbstractString; dataset_key::Union{Nothing, AbstractString} = nothing)
    return joinpath(default_plot_dir(STN_PLOT_EXPORT_DIR; dataset_key=dataset_key), "$(name).png")
end
