using Random

function bitvector_to_index(bits::BitVector)
    index = 0

    for i in eachindex(bits)
        bits[i] && (index |= 1 << (i - 1))
    end

    return index
end

function index_to_bitvector(index::Integer, nbits::Integer)
    n = Int(nbits)
    bits = falses(n)

    for i in 1:n
        bits[i] = !iszero(Int(index) & (1 << (i - 1)))
    end

    return bits
end

function best_single_feature_index(landscape::Landscape, epsilon::Real)
    landscape.num_features > 0 ||
        throw(ArgumentError("Cannot compute a repair index for a zero-feature landscape"))

    best_index = 1
    best_value = candidate_state(landscape, best_index, epsilon).penalized_fitness

    for bit in 1:landscape.num_features
        index = 1 << (bit - 1)
        value = candidate_state(landscape, index, epsilon).penalized_fitness

        if value > best_value
            best_index = index
            best_value = value
        end
    end

    return best_index
end

function decode_ga_individual(ind::BitVector;
                              allow_zero::Bool,
                              repair_index::Integer)
    index = bitvector_to_index(ind)

    if !allow_zero && index == 0
        return Int(repair_index)
    end

    return index
end

function order_candidate_states(states)
    ordered = collect(states)
    sort!(ordered; by = state -> (state.num_selected, -state.accuracy, state.index))
    return ordered
end

function unique_candidate_states(states)
    unique_states = typeof(first(states))[]
    seen_indices = Set{Int}()

    for state in order_candidate_states(states)
        if !(state.index in seen_indices)
            push!(unique_states, state)
            push!(seen_indices, state.index)
        end
    end

    return unique_states
end

function best_penalized_candidate_state(states)
    isempty(states) && throw(ArgumentError("states must not be empty"))
    best = first(states)

    for state in Iterators.drop(states, 1)
        if state.penalized_fitness > best.penalized_fitness ||
           (state.penalized_fitness == best.penalized_fitness && state.num_selected < best.num_selected) ||
           (state.penalized_fitness == best.penalized_fitness && state.num_selected == best.num_selected && state.accuracy > best.accuracy) ||
           (state.penalized_fitness == best.penalized_fitness && state.num_selected == best.num_selected && state.accuracy == best.accuracy && state.index < best.index)
            best = state
        end
    end

    return best
end

function run_single_objective_ea(landscape::Landscape;
                                 iterations::Integer,
                                 epsilon::Real = 0.0,
                                 initial_index::Union{Nothing, Integer} = nothing,
                                 population_size::Integer = 100,
                                 crossover_probability::Real = 0.95,
                                 mutation_probability::Union{Nothing, Real} = nothing,
                                 tournament_size::Integer = 4,
                                 survivor_mode::Symbol = :elitist,
                                 elite::Integer = 4,
                                 log_every::Integer = 0,
                                 rng::AbstractRNG = Random.default_rng(),
                                 keep_history::Bool = false)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    population_size > 0 || throw(ArgumentError("population_size must be positive"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))
    0 <= crossover_probability <= 1 || throw(ArgumentError("crossover_probability must be between 0 and 1"))
    tournament_size > 0 || throw(ArgumentError("tournament_size must be positive"))
    elite >= 0 || throw(ArgumentError("elite must be non-negative"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    if landscape.num_features == 0 && !landscape.allow_zero
        throw(ArgumentError("Cannot optimize a zero-feature landscape when allow_zero is false"))
    end

    max_index = (1 << landscape.num_features) - 1
    if !isnothing(initial_index)
        0 <= Int(initial_index) <= max_index ||
            throw(ArgumentError("initial_index must be between 0 and $max_index"))

        if !landscape.allow_zero && Int(initial_index) == 0
            throw(ArgumentError("initial_index must be non-zero when allow_zero is false"))
        end
    end

    pm = isnothing(mutation_probability) ?
        (landscape.num_features == 0 ? 0.0 : inv(landscape.num_features)) :
        Float64(mutation_probability)
    0 <= pm <= 1 || throw(ArgumentError("mutation_probability must be between 0 and 1"))

    repair_index = landscape.allow_zero ? 0 : best_single_feature_index(landscape, epsilon)
    fitness_fn = function (ind::BitVector)
        index = decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index)
        return candidate_state(landscape, index, epsilon).penalized_fitness
    end

    seed = rand(rng, 1:typemax(Int))
    initial_population = nothing

    if !isnothing(initial_index)
        population_seed = seed == typemax(Int) ? 1 : seed + 1
        seeded_population = GACore.init_population(Int(population_size), landscape.num_features, MersenneTwister(population_seed))
        seeded_population[1] = index_to_bitvector(Int(initial_index), landscape.num_features)
        initial_population = seeded_population
    end

    params = GACore.GAParams(
        popsize = Int(population_size),
        generations = Int(iterations),
        pc = Float64(crossover_probability),
        pm = pm,
        tour_k = Int(tournament_size),
        survivor_mode = survivor_mode,
        elite = Int(elite),
        seed = seed,
        objective = :max,
        log_every = Int(log_every),
        record_history = keep_history,
    )

    best_ind, best_raw, worst_ind, worst_raw, history = GACore.run_ga(
        landscape.num_features,
        fitness_fn,
        initial_population;
        params=params,
    )

    current_states = keep_history ? [
        candidate_state(
            landscape,
            decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
            epsilon,
        )
        for ind in history.current_best_ind_hist
    ] : nothing
    best_states = keep_history ? [
        candidate_state(
            landscape,
            decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
            epsilon,
        )
        for ind in history.best_so_far_ind_hist
    ] : nothing

    start = keep_history ? current_states[1] : candidate_state(
        landscape,
        decode_ga_individual(history.initial_best_ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
        epsilon,
    )
    final = keep_history ? current_states[end] : candidate_state(
        landscape,
        decode_ga_individual(history.final_best_ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
        epsilon,
    )
    best = candidate_state(
        landscape,
        decode_ga_individual(best_ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
        epsilon,
    )
    worst = candidate_state(
        landscape,
        decode_ga_individual(worst_ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
        epsilon,
    )

    return (
        algorithm = :single_objective_ga,
        landscape = landscape.name,
        iterations = Int(iterations),
        epsilon = Float64(epsilon),
        population_size = Int(population_size),
        crossover_probability = Float64(crossover_probability),
        mutation_probability = pm,
        tournament_size = Int(tournament_size),
        survivor_mode = survivor_mode,
        elite = Int(elite),
        seed = seed,
        initial_index = start.index,
        initial_accuracy = start.accuracy,
        initial_penalized_fitness = start.penalized_fitness,
        initial_num_selected = start.num_selected,
        final_index = final.index,
        final_accuracy = final.accuracy,
        final_penalized_fitness = final.penalized_fitness,
        final_num_selected = final.num_selected,
        best_index = best.index,
        best_accuracy = best.accuracy,
        best_penalized_fitness = best.penalized_fitness,
        best_num_selected = best.num_selected,
        worst_index = worst.index,
        worst_accuracy = worst.accuracy,
        worst_penalized_fitness = worst_raw,
        worst_num_selected = worst.num_selected,
        current_history = keep_history ? history.current_best_raw_hist : nothing,
        best_history = keep_history ? history.best_so_far_raw_hist : nothing,
        current_accuracy_history = keep_history ? [state.accuracy for state in current_states] : nothing,
        best_accuracy_history = keep_history ? [state.accuracy for state in best_states] : nothing,
        current_num_selected_history = keep_history ? [state.num_selected for state in current_states] : nothing,
        best_num_selected_history = keep_history ? [state.num_selected for state in best_states] : nothing,
        current_index_history = keep_history ? [state.index for state in current_states] : nothing,
        best_index_history = keep_history ? [state.index for state in best_states] : nothing,
        mean_history = keep_history ? history.mean_hist : nothing,
        max_history = keep_history ? history.max_hist : nothing,
        min_history = keep_history ? history.min_hist : nothing,
        entropy_history = keep_history ? history.ent_hist : nothing,
        raw_best = best_raw,
    )
end

function run_nsga2_feature_ea(landscape::Landscape;
                              iterations::Integer,
                              epsilon::Real = 0.0,
                              initial_index::Union{Nothing, Integer} = nothing,
                              population_size::Integer = 100,
                              crossover_probability::Real = 0.95,
                              mutation_probability::Union{Nothing, Real} = nothing,
                              log_every::Integer = 0,
                              rng::AbstractRNG = Random.default_rng(),
                              keep_history::Bool = false)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    population_size > 0 || throw(ArgumentError("population_size must be positive"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))
    0 <= crossover_probability <= 1 || throw(ArgumentError("crossover_probability must be between 0 and 1"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    if landscape.num_features == 0 && !landscape.allow_zero
        throw(ArgumentError("Cannot optimize a zero-feature landscape when allow_zero is false"))
    end

    max_index = (1 << landscape.num_features) - 1
    if !isnothing(initial_index)
        0 <= Int(initial_index) <= max_index ||
            throw(ArgumentError("initial_index must be between 0 and $max_index"))

        if !landscape.allow_zero && Int(initial_index) == 0
            throw(ArgumentError("initial_index must be non-zero when allow_zero is false"))
        end
    end

    pm = isnothing(mutation_probability) ?
        (landscape.num_features == 0 ? 0.0 : inv(landscape.num_features)) :
        Float64(mutation_probability)
    0 <= pm <= 1 || throw(ArgumentError("mutation_probability must be between 0 and 1"))

    repair_index = landscape.allow_zero ? 0 : best_single_feature_index(landscape, epsilon)
    objective_fn = function (ind::BitVector)
        index = decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index)
        state = candidate_state(landscape, index, epsilon)
        return (state.accuracy, state.num_selected)
    end

    seed = rand(rng, 1:typemax(Int))
    initial_population = nothing

    if !isnothing(initial_index)
        population_seed = seed == typemax(Int) ? 1 : seed + 1
        seeded_population = GACore.init_population(Int(population_size), landscape.num_features, MersenneTwister(population_seed))
        seeded_population[1] = index_to_bitvector(Int(initial_index), landscape.num_features)
        initial_population = seeded_population
    end

    params = NSGA2Core.NSGA2Params(
        popsize = Int(population_size),
        generations = Int(iterations),
        pc = Float64(crossover_probability),
        pm = pm,
        seed = seed,
        log_every = Int(log_every),
        record_history = keep_history,
    )

    nsga2_result = NSGA2Core.run_nsga2(
        landscape.num_features,
        objective_fn,
        initial_population;
        params=params,
        directions=(:max, :min),
    )

    final_population_indices = [
        decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index)
        for ind in nsga2_result.population
    ]
    final_population_states = [
        candidate_state(landscape, index, epsilon)
        for index in final_population_indices
    ]
    pareto_states = unique_candidate_states([final_population_states[index] for index in nsga2_result.fronts[1]])
    best_penalized = best_penalized_candidate_state(pareto_states)
    pareto_indices_history = nothing
    pareto_accuracy_history = nothing
    pareto_num_selected_history = nothing
    pareto_penalized_fitness_history = nothing
    front_size_history = nothing
    population_indices_history = nothing
    offspring_indices_history = nothing
    transition_edges_history = nothing

    if keep_history
        history_states = [
            unique_candidate_states([
                candidate_state(
                    landscape,
                    decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index),
                    epsilon,
                )
                for ind in front_population
            ])
            for front_population in nsga2_result.history.pareto_front_population_hist
        ]
        pareto_indices_history = [[state.index for state in states] for states in history_states]
        pareto_accuracy_history = [[state.accuracy for state in states] for states in history_states]
        pareto_num_selected_history = [[state.num_selected for state in states] for states in history_states]
        pareto_penalized_fitness_history = [[state.penalized_fitness for state in states] for states in history_states]
        front_size_history = [length(states) for states in history_states]

        population_indices_history = [
            [
                decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index)
                for ind in population
            ]
            for population in nsga2_result.history.population_hist
        ]
        offspring_indices_history = [
            [
                decode_ga_individual(ind; allow_zero=landscape.allow_zero, repair_index=repair_index)
                for ind in offspring
            ]
            for offspring in nsga2_result.history.offspring_hist
        ]
        transition_edges_history = [
            vcat(
                [
                    (
                        population_indices_history[generation][nsga2_result.history.parent_a_position_hist[generation][i]],
                        offspring_indices_history[generation][i],
                    )
                    for i in eachindex(offspring_indices_history[generation])
                ],
                [
                    (
                        population_indices_history[generation][nsga2_result.history.parent_b_position_hist[generation][i]],
                        offspring_indices_history[generation][i],
                    )
                    for i in eachindex(offspring_indices_history[generation])
                ],
            )
            for generation in eachindex(offspring_indices_history)
        ]
    end

    return (
        algorithm = :nsga2_feature_ea,
        landscape = landscape.name,
        iterations = Int(iterations),
        epsilon = Float64(epsilon),
        population_size = Int(population_size),
        crossover_probability = Float64(crossover_probability),
        mutation_probability = pm,
        seed = seed,
        initial_index = isnothing(initial_index) ? nothing : Int(initial_index),
        evaluations = nsga2_result.evaluations,
        pareto_indices = [state.index for state in pareto_states],
        pareto_accuracy = [state.accuracy for state in pareto_states],
        pareto_num_selected = [state.num_selected for state in pareto_states],
        pareto_penalized_fitness = [state.penalized_fitness for state in pareto_states],
        final_population_indices = final_population_indices,
        final_population_accuracy = [state.accuracy for state in final_population_states],
        final_population_num_selected = [state.num_selected for state in final_population_states],
        final_population_ranks = nsga2_result.rank,
        final_population_crowding = nsga2_result.crowding,
        best_penalized_index = best_penalized.index,
        best_penalized_accuracy = best_penalized.accuracy,
        best_penalized_num_selected = best_penalized.num_selected,
        best_penalized_fitness = best_penalized.penalized_fitness,
        pareto_indices_history = pareto_indices_history,
        pareto_accuracy_history = pareto_accuracy_history,
        pareto_num_selected_history = pareto_num_selected_history,
        pareto_penalized_fitness_history = pareto_penalized_fitness_history,
        front_size_history = front_size_history,
        population_indices_history = population_indices_history,
        offspring_indices_history = offspring_indices_history,
        transition_edges_history = transition_edges_history,
    )
end
