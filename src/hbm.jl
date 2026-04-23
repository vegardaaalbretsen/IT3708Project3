function one_flip_neighbors(index::Int, n_features::Int; allow_zero::Bool = false)
    neighbors = Int[]
    for i in 1:n_features
        flipped = xor(index, 1 << (i - 1))
        if allow_zero || flipped != 0
            push!(neighbors, flipped)
        end
    end
    return neighbors
end

function build_hbm(indices::AbstractVector{<:Integer},
                   fitness_values::AbstractVector{<:Real},
                   n_features::Integer)
    length(indices) == length(fitness_values) || throw(ArgumentError("indices and fitness_values must have the same length"))
    nodes = HBMNode[]

    for i in eachindex(indices)
        index = indices[i]
        fitness = fitness_values[i]
        bits = last(bitstring(index), n_features)
        split = ceil(Int, n_features / 2)
        left = bits[1:split]
        right = bits[(split + 1):n_features]
        x = parse(Int, left, base=2)
        y = parse(Int, right, base=2)
        push!(nodes, HBMNode(index, fitness, x, y))
    end
    return nodes
end

function build_hbm(landscape::Landscape; values = fitness_values(landscape))
    return build_hbm(landscape.indices, values, landscape.num_features)
end

function local_optima(nodes::AbstractVector{HBMNode}, n_features::Int; allow_zero::Bool = false)
    optima = Int[]
    fitness_lookup = Dict(node.index => node.fitness for node in nodes)

    for node in nodes
        is_local = true

        for neighbor in one_flip_neighbors(node.index, n_features; allow_zero=allow_zero)
            haskey(fitness_lookup, neighbor) || continue
            if node.fitness < fitness_lookup[neighbor]
                is_local = false
                break
            end
        end

        if is_local
            push!(optima, node.index)
        end
    end

    return optima
end

function local_optima(landscape::Landscape; values = fitness_values(landscape))
    nodes = build_hbm(landscape; values=values)
    return local_optima(nodes, landscape.num_features; allow_zero=landscape.allow_zero)
end

function global_optima(nodes::AbstractArray{HBMNode})
    isempty(nodes) && return Int[]

    best_fitness = maximum(node.fitness for node in nodes)
    return [node.index for node in nodes if node.fitness == best_fitness]
end

function global_optima(landscape::Landscape; values = fitness_values(landscape))
    return global_optima(build_hbm(landscape; values=values))
end

function optima_coverage(indices::AbstractVector{<:Integer},
                         optima_indices::AbstractVector{<:Integer})
    optima_set = Set(Int.(optima_indices))
    found_set = Set{Int}()

    for index in indices
        index = Int(index)
        index in optima_set && push!(found_set, index)
    end

    found = sort!(collect(found_set))
    total = length(optima_set)
    count = length(found)
    fraction = total == 0 ? 0.0 : count / total

    return (
        found = found,
        count = count,
        total = total,
        fraction = Float64(fraction),
    )
end

function optima_coverage_trace(index_history::AbstractVector,
                               optima_indices::AbstractVector{<:Integer})
    optima_set = Set(Int.(optima_indices))
    seen = Set{Int}()
    cumulative_counts = Int[]
    total = length(optima_set)

    for indices in index_history
        for index in indices
            index = Int(index)
            index in optima_set && push!(seen, index)
        end
        push!(cumulative_counts, length(seen))
    end

    cumulative_fraction = total == 0 ?
        fill(0.0, length(cumulative_counts)) :
        Float64[count / total for count in cumulative_counts]

    return (
        found = sort!(collect(seen)),
        cumulative_counts = cumulative_counts,
        cumulative_fraction = cumulative_fraction,
        total = total,
    )
end
