using Random

function sample_index(rng::AbstractRNG, n_features::Integer; allow_zero::Bool = false)
    n = Int(n_features)
    n >= 0 || throw(ArgumentError("n_features must be non-negative"))

    max_index = (1 << n) - 1
    if allow_zero
        return rand(rng, 0:max_index)
    end

    max_index >= 1 || throw(ArgumentError("Cannot sample a non-zero index from an empty landscape"))
    return rand(rng, 1:max_index)
end

function standard_bit_mutation(parent_index::Integer,
                               n_features::Integer;
                               rng::AbstractRNG = Random.default_rng(),
                               allow_zero::Bool = false)
    n = Int(n_features)
    n >= 0 || throw(ArgumentError("n_features must be non-negative"))

    parent = Int(parent_index)
    max_index = (1 << n) - 1
    0 <= parent <= max_index || throw(ArgumentError("parent_index must be between 0 and $max_index"))

    if !allow_zero && parent == 0
        throw(ArgumentError("parent_index must be non-zero when allow_zero is false"))
    end

    if n == 0
        allow_zero || throw(ArgumentError("Cannot mutate a zero-feature landscape when allow_zero is false"))
        return parent
    end

    mutation_rate = inv(n)

    while true
        child = parent

        for bit in 0:(n - 1)
            if rand(rng) < mutation_rate
                child = xor(child, 1 << bit)
            end
        end

        if allow_zero || child != 0
            return child
        end
    end
end

function candidate_state(landscape::Landscape, index::Integer, epsilon::Real)
    accuracy = fitness(landscape, index)
    num_selected = count_ones(index)
    objective = penalized_fitness(accuracy, num_selected, epsilon)

    return (
        index = Int(index),
        accuracy = accuracy,
        penalized_fitness = objective,
        num_selected = num_selected,
    )
end

function run_standard_ea(landscape::Landscape;
                         iterations::Integer,
                         epsilon::Real = 0.0,
                         initial_index::Union{Nothing, Integer} = nothing,
                         rng::AbstractRNG = Random.default_rng(),
                         keep_history::Bool = false)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))

    start_index = isnothing(initial_index) ?
        sample_index(rng, landscape.num_features; allow_zero=landscape.allow_zero) :
        Int(initial_index)

    start = candidate_state(landscape, start_index, epsilon)
    current = start
    best = start
    accepted_moves = 0

    current_history = keep_history ? Float64[start.penalized_fitness] : Float64[]
    best_history = keep_history ? Float64[start.penalized_fitness] : Float64[]

    for _ in 1:Int(iterations)
        child_index = standard_bit_mutation(
            current.index,
            landscape.num_features;
            rng=rng,
            allow_zero=landscape.allow_zero,
        )
        child = candidate_state(landscape, child_index, epsilon)

        if child.penalized_fitness >= current.penalized_fitness
            current = child
            accepted_moves += 1
        end

        if current.penalized_fitness > best.penalized_fitness
            best = current
        end

        if keep_history
            push!(current_history, current.penalized_fitness)
            push!(best_history, best.penalized_fitness)
        end
    end

    return (
        landscape = landscape.name,
        iterations = Int(iterations),
        epsilon = Float64(epsilon),
        accepted_moves = accepted_moves,
        initial_index = start.index,
        initial_accuracy = start.accuracy,
        initial_penalized_fitness = start.penalized_fitness,
        initial_num_selected = start.num_selected,
        final_index = current.index,
        final_accuracy = current.accuracy,
        final_penalized_fitness = current.penalized_fitness,
        final_num_selected = current.num_selected,
        best_index = best.index,
        best_accuracy = best.accuracy,
        best_penalized_fitness = best.penalized_fitness,
        best_num_selected = best.num_selected,
        current_history = keep_history ? current_history : nothing,
        best_history = keep_history ? best_history : nothing,
    )
end
