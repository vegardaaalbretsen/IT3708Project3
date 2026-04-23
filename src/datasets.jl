const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW_DATA_DIR = joinpath(PROJECT_ROOT, "data", "raw")
const TEST_DATA_DIR = joinpath(PROJECT_ROOT, "data", "test_data")
const CSV_EXPORT_DIR = joinpath(PROJECT_ROOT, "exports", "csv")
const NSGA2_RESULT_EXPORT_DIR = joinpath(CSV_EXPORT_DIR, "results")
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
    "zoo" => (
        path = joinpath(TEST_DATA_DIR, "06-zoo_lr_F.h5"),
        num_features = 16,
    ),
    "hepatitis" => (
        path = joinpath(TEST_DATA_DIR, "10-hepatitis_lr_F.h5"),
        num_features = 19,
    ),
)

function default_output_path(dataset_key::AbstractString)
    return joinpath(CSV_EXPORT_DIR, "$(dataset_key)_metrics.csv")
end

function default_nsga2_result_path(name::AbstractString)
    return joinpath(NSGA2_RESULT_EXPORT_DIR, "$(name).csv")
end

function default_hbm_plot_path(name::AbstractString)
    return joinpath(HBM_PLOT_EXPORT_DIR, "$(name)_hbm.png")
end

function default_feature_count_plot_path(name::AbstractString)
    return joinpath(FEATURE_COUNT_PLOT_EXPORT_DIR, "$(name).png")
end

function default_ea_plot_path(name::AbstractString)
    return joinpath(EA_PLOT_EXPORT_DIR, "$(name).png")
end

function default_stn_plot_path(name::AbstractString)
    return joinpath(STN_PLOT_EXPORT_DIR, "$(name).png")
end
