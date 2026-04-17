struct HBMNode
    index::Int
    fitness::Float64
    x::Int
    y::Int
end

function one_flip_neighbors(index::Int, n_features::Int)
    neighbors = Int[]
    for i in 1:n_features
        flipped = (index ⊻ (1 << (i - 1)))
        if flipped != 0
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

function local_optima(nodes::AbstractVector{HBMNode}, n_features::Int)
    optima = Int[]
    fitness_lookup = Dict(node.index => node.fitness for node in nodes)

    for node in nodes
        is_local = true

        for neighbor in one_flip_neighbors(node.index, n_features)
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

function global_optima(nodes::AbstractArray{HBMNode})
    isempty(nodes) && return Int[]

    best_fitness = maximum(node.fitness for node in nodes)
    return [node.index for node in nodes if node.fitness == best_fitness]
end
