const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW_DATA_DIR = joinpath(PROJECT_ROOT, "data", "raw")
const CSV_EXPORT_DIR = joinpath(PROJECT_ROOT, "exports", "csv")
const PLOT_EXPORT_DIR = joinpath(PROJECT_ROOT, "exports", "plots")
const HBM_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "hbm")
const FEATURE_COUNT_PLOT_EXPORT_DIR = joinpath(PLOT_EXPORT_DIR, "feature_count")

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

function default_hbm_plot_path(name::AbstractString)
    return joinpath(HBM_PLOT_EXPORT_DIR, "$(name)_hbm.png")
end

function default_feature_count_plot_path(name::AbstractString)
    return joinpath(FEATURE_COUNT_PLOT_EXPORT_DIR, "$(name).png")
end
