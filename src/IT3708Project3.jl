module IT3708Project3

using HDF5
using Statistics

export DATASET_SPECS,
       bitflip_neighbors,
       padded_bitstring,
       hinged_bitstring_coordinates,
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
       triangle_landscape,
       local_optima_network,
       triangle_local_optima_network,
       feature_selection_local_optima_network

const FITNESS_TOL = 1e-12

"""
    padded_bitstring(index, n_features)

Return the bitstring for `index` as a length-`n_features` vector with the most
significant bit first.
"""
function padded_bitstring(index::Integer, n_features::Integer)
    index >= 0 || throw(ArgumentError("subset index must be non-negative"))
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    index < (1 << n_features) || throw(ArgumentError("subset index $index exceeds $n_features features"))

    bits = Vector{Int}(undef, n_features)
    for position in 1:n_features
        shift = n_features - position
        bits[position] = (index >> shift) & 1
    end
    return bits
end

function decimal_bits(bits::AbstractVector{<:Integer})
    value = 0
    for bit in bits
        value = (value << 1) | Int(bit)
    end
    return value
end

"""
    hinged_bitstring_coordinates(index, n_features)

Map one bitstring to its hinged bitstring map coordinates by splitting the
bitstring into two halves and converting each half to decimal.

For odd `n_features`, the first half receives the extra bit.
"""
function hinged_bitstring_coordinates(index::Integer, n_features::Integer)
    bits = padded_bitstring(index, n_features)
    split = cld(n_features, 2)
    xbits = bits[1:split]
    ybits = bits[(split + 1):end]

    x = decimal_bits(xbits)
    y = isempty(ybits) ? 0 : decimal_bits(ybits)
    return x, y
end

"""
    bitflip_neighbors(index, n_features)

Return all Hamming-1 neighbors of `index` for a bitstring of length `n_features`.
"""
function bitflip_neighbors(index::Integer, n_features::Integer)
    index >= 0 || throw(ArgumentError("subset index must be non-negative"))
    n_features > 0 || throw(ArgumentError("n_features must be positive"))

    return [xor(index, 1 << (bit - 1)) for bit in 1:n_features]
end

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

function best_improving_neighbor(index::Int, fitness_map::Dict{Int, Float64}, n_features::Int)
    current_fitness = fitness_map[index]
    best_neighbor = index
    best_fitness = current_fitness

    for neighbor in bitflip_neighbors(index, n_features)
        haskey(fitness_map, neighbor) || continue
        neighbor_fitness = fitness_map[neighbor]

        if neighbor_fitness > current_fitness + FITNESS_TOL
            if best_neighbor == index || neighbor_fitness > best_fitness + FITNESS_TOL ||
               (abs(neighbor_fitness - best_fitness) <= FITNESS_TOL && neighbor < best_neighbor)
                best_neighbor = neighbor
                best_fitness = neighbor_fitness
            end
        end
    end

    return best_neighbor
end

function hill_climb_to_optimum(index::Int, fitness_map::Dict{Int, Float64}, n_features::Int, cache::Dict{Int, Int})
    path = Int[]
    current = index

    while true
        if haskey(cache, current)
            optimum = cache[current]
            for state in path
                cache[state] = optimum
            end
            return optimum
        end

        push!(path, current)
        next = best_improving_neighbor(current, fitness_map, n_features)
        if next == current
            for state in path
                cache[state] = current
            end
            return current
        end

        current = next
    end
end

"""
    local_optima_network(subset_indices, fitness_values, n_features)

Build a directed, weighted local optima network for a bitstring landscape.
Each state is hill-climbed by deterministic best-improvement to its local optimum.
Nodes are local optima, node sizes are basin sizes, and edge weights count Hamming-1
crossings between basins.
"""
function local_optima_network(subset_indices::AbstractVector{<:Integer}, fitness_values::AbstractVector{<:Real}, n_features::Integer)
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    length(subset_indices) == length(fitness_values) || throw(ArgumentError("subset_indices and fitness_values must have the same length"))

    fitness_map = Dict(Int(subset_index) => Float64(fitness) for (subset_index, fitness) in zip(subset_indices, fitness_values))
    cache = Dict{Int, Int}()
    optimum_of_state = Dict{Int, Int}()

    for subset_index in subset_indices
        optimum_of_state[Int(subset_index)] = hill_climb_to_optimum(Int(subset_index), fitness_map, Int(n_features), cache)
    end

    node_subset_indices = sort(unique(values(optimum_of_state)))
    node_position = Dict(node => position for (position, node) in enumerate(node_subset_indices))
    basin_sizes = zeros(Int, length(node_subset_indices))

    for optimum in values(optimum_of_state)
        basin_sizes[node_position[optimum]] += 1
    end

    edge_counts = Dict{Tuple{Int, Int}, Int}()
    outgoing_counts = zeros(Int, length(node_subset_indices))

    for subset_index in subset_indices
        source_optimum = optimum_of_state[Int(subset_index)]
        source_node = node_position[source_optimum]

        for neighbor in bitflip_neighbors(Int(subset_index), Int(n_features))
            haskey(fitness_map, neighbor) || continue

            target_optimum = optimum_of_state[neighbor]
            if target_optimum != source_optimum
                target_node = node_position[target_optimum]
                edge_counts[(source_node, target_node)] = get(edge_counts, (source_node, target_node), 0) + 1
                outgoing_counts[source_node] += 1
            end
        end
    end

    edges = sort([
        (
            source = source,
            target = target,
            count = count,
            probability = outgoing_counts[source] == 0 ? 0.0 : count / outgoing_counts[source],
        ) for ((source, target), count) in edge_counts
    ], by = edge -> (-edge.count, edge.source, edge.target))

    return (
        subset_indices = Int.(subset_indices),
        fitness_values = Float64.(fitness_values),
        optimum_of_state = optimum_of_state,
        node_subset_indices = node_subset_indices,
        node_fitness = [fitness_map[node] for node in node_subset_indices],
        node_active_counts = [count_ones(node) for node in node_subset_indices],
        basin_sizes = basin_sizes,
        edges = edges,
        n_features = Int(n_features),
    )
end

"""
    triangle_local_optima_network(n = 16, m = 1, s = 4; include_zero = true)

Build a local optima network for the synthetic triangle landscape.
"""
function triangle_local_optima_network(n::Integer = 16, m::Integer = 1, s::Integer = 4; include_zero::Bool = true)
    landscape = triangle_landscape(n, m, s; include_zero = include_zero)
    return local_optima_network(landscape.subset_indices, landscape.fitness, landscape.n_features)
end

"""
    feature_selection_local_optima_network(path, n_features; epsilon = 1/8, score = :penalized)

Build a local optima network for one of the feature-selection landscapes.
`score` may be `:raw` or `:penalized`.
"""
function feature_selection_local_optima_network(path::AbstractString, n_features::Integer; epsilon::Real = 1 / 8, score::Symbol = :penalized)
    landscape = read_feature_selection_landscape(path, n_features; epsilon = epsilon)
    fitness_values = if score == :raw
        landscape.raw_accuracy_table
    elseif score == :penalized
        landscape.penalized_table
    else
        throw(ArgumentError("score must be :raw or :penalized"))
    end

    return local_optima_network(landscape.subset_indices, fitness_values, n_features)
end

end # module IT3708Project3
