module IT3708Project3

using HDF5
using Statistics

export DATASET_SPECS,
       subset_to_bitvector,
       active_columns,
       feature_penalty,
       mean_accuracy_landscape,
       mean_time_landscape,
       read_feature_selection_landscape,
       summarize_subset,
       best_raw_subset,
       best_penalized_subset,
       triangle_fitness,
       triangle_landscape

const DATASET_SPECS = Dict(
    "breast-w" => (path = joinpath("train data", "01-breast-w_lr_F.h5"), n_features = 9),
    "credit-a" => (path = joinpath("train data", "05-credit-a_rf_F.h5"), n_features = 15),
    "letter-r" => (path = joinpath("train data", "08-letter-r_knn_F.h5"), n_features = 16),
)

"""
    subset_to_bitvector(index, n_features)

Decode a subset index into a bit vector where bit 1 corresponds to column 1.
The least-significant bit maps to the first feature/column.
"""
function subset_to_bitvector(index::Integer, n_features::Integer)
    index >= 0 || throw(ArgumentError("subset index must be non-negative"))
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    index < (1 << n_features) || throw(ArgumentError("subset index $index exceeds $n_features features"))

    bits = falses(n_features)
    for feature in 1:n_features
        bits[feature] = ((index >> (feature - 1)) & 1) == 1
    end
    return bits
end

"""
    active_columns(index, n_features)

Return the 1-based feature/column indices enabled by `index`.
"""
function active_columns(index::Integer, n_features::Integer)
    return findall(identity, subset_to_bitvector(index, n_features))
end

"""
    feature_penalty(index; epsilon = 1/8)

Linear penalty based on the number of selected columns.
This matches the common regularization form `epsilon * sum(b_i)`.
"""
function feature_penalty(index::Integer; epsilon::Real = 1 / 8)
    index >= 0 || throw(ArgumentError("subset index must be non-negative"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be non-negative"))
    return Float64(epsilon) * count_ones(index)
end

"""
    mean_hdf5_dataset(path, dataset_name)

Read one HDF5 dataset and average over the repeated evaluations for each subset.
The provided files are stored as `(32, 2^n - 1)` in Python-style inspection,
but Julia reads them as `(2^n - 1, 32)`, so the mean is taken over dimension 2.
"""
function mean_hdf5_dataset(path::AbstractString, dataset_name::AbstractString)
    dataset = h5open(path, "r") do file
        read(file[dataset_name])
    end

    ndims(dataset) == 2 || throw(ArgumentError("expected a 2D `$dataset_name` dataset"))
    return Float64.(vec(mean(dataset; dims = 2)))
end

"""
    mean_accuracy_landscape(path)

Build the raw mean-accuracy lookup table from the `accuracies` dataset.
"""
function mean_accuracy_landscape(path::AbstractString)
    return mean_hdf5_dataset(path, "accuracies")
end

"""
    mean_time_landscape(path)

Build the raw mean-time lookup table from the `times` dataset.
"""
function mean_time_landscape(path::AbstractString)
    return mean_hdf5_dataset(path, "times")
end

"""
    read_feature_selection_landscape(path, n_features; epsilon = 1/8)

Parse one of the provided HDF5 landscapes and return explicit lookup tables for
raw accuracy, raw training time, penalties, and penalized fitness.

The penalty choice is `epsilon = 1/8` by default, which matches the regularized
feature-selection formulation used in the related course literature.
"""
function read_feature_selection_landscape(path::AbstractString, n_features::Integer; epsilon::Real = 1 / 8)
    n_features > 0 || throw(ArgumentError("n_features must be positive"))

    raw_accuracy_table = mean_accuracy_landscape(path)
    raw_time_table = mean_time_landscape(path)
    expected_subsets = (1 << n_features) - 1
    length(raw_accuracy_table) == expected_subsets || throw(ArgumentError("expected $expected_subsets subsets for $n_features features, got $(length(raw_accuracy_table))"))
    length(raw_time_table) == expected_subsets || throw(ArgumentError("expected $expected_subsets time values for $n_features features, got $(length(raw_time_table))"))

    subset_indices = collect(1:expected_subsets)
    penalty_table = [feature_penalty(index; epsilon = epsilon) for index in subset_indices]
    penalized_table = raw_accuracy_table .- penalty_table

    return (
        subset_indices = subset_indices,
        raw_accuracy_table = raw_accuracy_table,
        raw_time_table = raw_time_table,
        penalty_table = penalty_table,
        penalized_table = penalized_table,
        n_features = n_features,
        epsilon = Float64(epsilon),
    )
end

"""
    summarize_subset(landscape, position)

Summarize one entry in a real-data lookup table, including decoded columns.
"""
function summarize_subset(landscape, position::Integer)
    1 <= position <= length(landscape.subset_indices) || throw(ArgumentError("position $position is out of range"))

    subset_index = landscape.subset_indices[position]
    columns = active_columns(subset_index, landscape.n_features)
    return (
        position = position,
        subset_index = subset_index,
        active_columns = columns,
        n_active = length(columns),
        raw_accuracy = landscape.raw_accuracy_table[position],
        mean_time = landscape.raw_time_table[position],
        penalty = landscape.penalty_table[position],
        penalized_fitness = landscape.penalized_table[position],
    )
end

"""
    best_raw_subset(landscape)

Return the subset with the highest raw mean accuracy.
"""
function best_raw_subset(landscape)
    return summarize_subset(landscape, argmax(landscape.raw_accuracy_table))
end

"""
    best_penalized_subset(landscape)

Return the subset with the highest penalized fitness.
"""
function best_penalized_subset(landscape)
    return summarize_subset(landscape, argmax(landscape.penalized_table))
end

"""
    triangle_fitness(n, m, s)

Synthetic triangle fitness from the lecture definition, evaluated on the
number of active bits `||b||`.
"""
function triangle_fitness(n::Integer, m::Integer, s::Integer)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    m > 0 || throw(ArgumentError("m must be positive"))
    s > 1 || throw(ArgumentError("s must be greater than 1"))

    block = cld(n, s)
    if isodd(block)
        remainder = mod(n, s)
        return Float64(m * (remainder == 0 ? s : remainder))
    end

    return Float64(m * (block * s - n))
end

"""
    triangle_landscape(n = 16, m = 1, s = 4; include_zero = true)

Generate the synthetic triangle landscape for all bitstrings of length `n`.
The returned fitness depends only on each bitstring's number of active bits,
which keeps the landscape analytically tractable.
"""
function triangle_landscape(n::Integer = 16, m::Integer = 1, s::Integer = 4; include_zero::Bool = true)
    n > 0 || throw(ArgumentError("n must be positive"))

    first_index = include_zero ? 0 : 1
    last_index = (1 << n) - 1
    subset_indices = collect(first_index:last_index)
    fitness = [triangle_fitness(count_ones(index), m, s) for index in subset_indices]

    return (
        subset_indices = subset_indices,
        fitness = fitness,
        n_features = n,
        m = m,
        s = s,
    )
end

end # module IT3708Project3
