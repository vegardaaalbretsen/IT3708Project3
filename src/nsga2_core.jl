module NSGA2Core

using Random
using Base.Threads
using ..GACore

export NSGA2Params,
       dominates,
       fast_nondominated_sort,
       crowding_distance,
       binary_tournament_select,
       environmental_selection,
       run_nsga2

Base.@kwdef mutable struct NSGA2Params
    popsize::Int = 200
    generations::Int = 10000
    pc::Float64 = 0.90
    pm::Float64 = 0.0
    seed::Int = 42
    log_every::Int = 0
    record_history::Bool = true
    threaded_evaluation::Bool = false
end

function dominates(a, b; directions::Tuple = (:max, :min))
    length(a) == length(b) || throw(ArgumentError("Objective vectors must have the same length"))
    length(a) == length(directions) || throw(ArgumentError("directions must match the number of objectives"))

    strictly_better = false
    for i in eachindex(directions)
        direction = directions[i]

        if direction === :max
            if a[i] < b[i]
                return false
            elseif a[i] > b[i]
                strictly_better = true
            end
        elseif direction === :min
            if a[i] > b[i]
                return false
            elseif a[i] < b[i]
                strictly_better = true
            end
        else
            throw(ArgumentError("direction must be :max or :min, got $direction"))
        end
    end

    return strictly_better
end

function fast_nondominated_sort(objectives::AbstractVector; directions::Tuple = (:max, :min))
    isempty(objectives) && return Vector{Vector{Int}}(), Int[]

    n = length(objectives)
    domination_counts = zeros(Int, n)
    dominated_sets = [Int[] for _ in 1:n]
    rank = zeros(Int, n)
    first_front = Int[]

    for p in 1:n
        for q in 1:n
            p == q && continue

            if dominates(objectives[p], objectives[q]; directions=directions)
                push!(dominated_sets[p], q)
            elseif dominates(objectives[q], objectives[p]; directions=directions)
                domination_counts[p] += 1
            end
        end

        if domination_counts[p] == 0
            rank[p] = 1
            push!(first_front, p)
        end
    end

    fronts = Vector{Vector{Int}}()
    current_front = first_front
    current_rank = 1

    while !isempty(current_front)
        push!(fronts, current_front)
        next_front = Int[]

        for p in current_front
            for q in dominated_sets[p]
                domination_counts[q] -= 1

                if domination_counts[q] == 0
                    rank[q] = current_rank + 1
                    push!(next_front, q)
                end
            end
        end

        current_front = next_front
        current_rank += 1
    end

    return fronts, rank
end

function crowding_distance(objectives::AbstractVector,
                           front::AbstractVector{<:Integer})
    distances = fill(0.0, length(objectives))
    isempty(front) && return distances

    if length(front) <= 2
        for index in front
            distances[Int(index)] = Inf
        end
        return distances
    end

    n_objectives = length(first(objectives))

    for objective_index in 1:n_objectives
        sorted_front = sort(Int.(collect(front)); by = index -> Float64(objectives[index][objective_index]))
        min_value = Float64(objectives[sorted_front[1]][objective_index])
        max_value = Float64(objectives[sorted_front[end]][objective_index])

        distances[sorted_front[1]] = Inf
        distances[sorted_front[end]] = Inf

        max_value == min_value && continue

        for position in 2:(length(sorted_front) - 1)
            previous_value = Float64(objectives[sorted_front[position - 1]][objective_index])
            next_value = Float64(objectives[sorted_front[position + 1]][objective_index])

            if isfinite(distances[sorted_front[position]])
                distances[sorted_front[position]] += (next_value - previous_value) / (max_value - min_value)
            end
        end
    end

    return distances
end

function binary_tournament_select(pop::Vector{BitVector},
                                  rank::Vector{Int},
                                  crowding::Vector{Float64},
                                  rng::AbstractRNG)
    length(pop) == length(rank) == length(crowding) ||
        throw(ArgumentError("pop, rank, and crowding must have the same length"))

    a = rand(rng, eachindex(pop))
    b = rand(rng, eachindex(pop))

    winner = if rank[a] < rank[b]
        a
    elseif rank[b] < rank[a]
        b
    elseif crowding[a] > crowding[b]
        a
    elseif crowding[b] > crowding[a]
        b
    else
        rand(rng, (a, b))
    end

    return pop[winner]
end

function binary_tournament_index(popsize::Int,
                                 rank::Vector{Int},
                                 crowding::Vector{Float64},
                                 rng::AbstractRNG)
    popsize > 0 || throw(ArgumentError("popsize must be positive"))
    popsize == length(rank) == length(crowding) ||
        throw(ArgumentError("popsize, rank, and crowding must have the same length"))

    a = rand(rng, 1:popsize)
    b = rand(rng, 1:popsize)

    if rank[a] < rank[b]
        return a
    elseif rank[b] < rank[a]
        return b
    elseif crowding[a] > crowding[b]
        return a
    elseif crowding[b] > crowding[a]
        return b
    end

    return rand(rng, (a, b))
end

function environmental_selection(population::Vector{BitVector},
                                 objectives::AbstractVector,
                                 popsize::Int;
                                 directions::Tuple = (:max, :min))
    length(population) == length(objectives) ||
        throw(ArgumentError("population and objectives must have the same length"))
    0 < popsize <= length(population) ||
        throw(ArgumentError("popsize must be between 1 and $(length(population))"))

    fronts, _ = fast_nondominated_sort(objectives; directions=directions)
    crowding = fill(0.0, length(objectives))
    selected_positions = Int[]

    for front in fronts
        front_crowding = crowding_distance(objectives, front)
        for index in front
            crowding[index] = front_crowding[index]
        end

        if length(selected_positions) + length(front) <= popsize
            append!(selected_positions, front)
            continue
        end

        remaining = popsize - length(selected_positions)
        sorted_front = sort(collect(front); by = index -> crowding[index], rev=true)
        append!(selected_positions, sorted_front[1:remaining])
        break
    end

    selected_population = [copy(population[index]) for index in selected_positions]
    selected_objectives = [objectives[index] for index in selected_positions]
    selected_fronts, selected_rank = fast_nondominated_sort(selected_objectives; directions=directions)
    selected_crowding = fill(0.0, length(selected_objectives))

    for front in selected_fronts
        front_crowding = crowding_distance(selected_objectives, front)
        for index in front
            selected_crowding[index] = front_crowding[index]
        end
    end

    return (
        population = selected_population,
        objectives = selected_objectives,
        fronts = selected_fronts,
        rank = selected_rank,
        crowding = selected_crowding,
    )
end

function evaluate_population(population::Vector{BitVector},
                             objective_fn::Function;
                             threaded_evaluation::Bool = false)
    isempty(population) && return Any[]

    first_value = objective_fn(population[1])
    results = Vector{typeof(first_value)}(undef, length(population))
    results[1] = first_value

    if threaded_evaluation && nthreads() > 1 && length(population) > 1
        Threads.@threads for i in 2:length(population)
            results[i] = objective_fn(population[i])
        end
    else
        for i in 2:length(population)
            results[i] = objective_fn(population[i])
        end
    end

    return results
end

function run_nsga2(nbits::Int,
                   objective_fn::Function;
                   params::NSGA2Params = NSGA2Params(),
                   directions::Tuple = (:max, :min))
    return run_nsga2(nbits, objective_fn, nothing; params=params, directions=directions)
end

function run_nsga2(nbits::Int,
                   objective_fn::Function,
                   initial_population::Union{Nothing, Vector{BitVector}};
                   params::NSGA2Params = NSGA2Params(),
                   directions::Tuple = (:max, :min))
    rng = MersenneTwister(params.seed)

    params.popsize > 0 || throw(ArgumentError("popsize must be positive"))
    params.generations >= 0 || throw(ArgumentError("generations must be non-negative"))
    0 <= params.pc <= 1 || throw(ArgumentError("pc must be between 0 and 1"))
    0 <= params.pm <= 1 || throw(ArgumentError("pm must be between 0 and 1"))
    params.log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    pop = if isnothing(initial_population)
        GACore.init_population(params.popsize, nbits, rng)
    else
        length(initial_population) == params.popsize ||
            throw(ArgumentError("initial_population must have length $(params.popsize)"))
        [copy(ind) for ind in initial_population]
    end

    for ind in pop
        length(ind) == nbits || throw(ArgumentError("All individuals must have length $nbits"))
    end

    evaluations = 0
    pareto_front_population_hist = params.record_history ? Vector{Vector{BitVector}}() : nothing
    front_size_hist = params.record_history ? Int[] : nothing
    population_hist = params.record_history ? Vector{Vector{BitVector}}() : nothing
    offspring_hist = params.record_history ? Vector{Vector{BitVector}}() : nothing
    parent_a_position_hist = params.record_history ? Vector{Vector{Int}}() : nothing
    parent_b_position_hist = params.record_history ? Vector{Vector{Int}}() : nothing

    function evaluate_population(population::Vector{BitVector})
        results = NSGA2Core.evaluate_population(
            population,
            objective_fn;
            threaded_evaluation=params.threaded_evaluation,
        )
        evaluations += length(population)
        return results
    end

    function record_population!(population::Vector{BitVector},
                                fronts::Vector{Vector{Int}})
        params.record_history || return

        push!(population_hist, [copy(ind) for ind in population])
        pareto_front = isempty(fronts) ? BitVector[] : [copy(population[index]) for index in fronts[1]]
        push!(pareto_front_population_hist, pareto_front)
        push!(front_size_hist, length(pareto_front))
    end

    objectives = evaluate_population(pop)
    fronts, rank = fast_nondominated_sort(objectives; directions=directions)
    crowding = fill(0.0, length(pop))
    for front in fronts
        front_crowding = crowding_distance(objectives, front)
        for index in front
            crowding[index] = front_crowding[index]
        end
    end
    record_population!(pop, fronts)

    for generation in 1:params.generations
        offspring = BitVector[]
        parent_a_positions = Int[]
        parent_b_positions = Int[]

        while length(offspring) < params.popsize
            parent_a_position = binary_tournament_index(length(pop), rank, crowding, rng)
            parent_b_position = binary_tournament_index(length(pop), rank, crowding, rng)
            parent_a = pop[parent_a_position]
            parent_b = pop[parent_b_position]

            child_a, child_b = GACore.crossover(parent_a, parent_b, params.pc, rng)
            GACore.mutate!(child_a, params.pm, rng)
            GACore.mutate!(child_b, params.pm, rng)

            push!(offspring, child_a)
            push!(parent_a_positions, parent_a_position)
            push!(parent_b_positions, parent_b_position)
            if length(offspring) < params.popsize
                push!(offspring, child_b)
                push!(parent_a_positions, parent_a_position)
                push!(parent_b_positions, parent_b_position)
            end
        end

        if params.record_history
            push!(offspring_hist, [copy(ind) for ind in offspring])
            push!(parent_a_position_hist, copy(parent_a_positions))
            push!(parent_b_position_hist, copy(parent_b_positions))
        end

        offspring_objectives = evaluate_population(offspring)
        merged_population = vcat(pop, offspring)
        merged_objectives = vcat(objectives, offspring_objectives)
        selected = environmental_selection(
            merged_population,
            merged_objectives,
            params.popsize;
            directions=directions,
        )

        pop = selected.population
        objectives = selected.objectives
        fronts = selected.fronts
        rank = selected.rank
        crowding = selected.crowding
        record_population!(pop, fronts)

        if params.log_every > 0 && (generation == 1 || generation % params.log_every == 0 || generation == params.generations)
            front_objectives = isempty(fronts) ? [] : [objectives[index] for index in fronts[1]]
            best_accuracy = isempty(front_objectives) ? 0.0 : maximum(Float64(objective[1]) for objective in front_objectives)
            min_features = isempty(front_objectives) ? 0 : minimum(Int(objective[2]) for objective in front_objectives)
            log_message = "gen=$generation  front=$(isempty(fronts) ? 0 : length(fronts[1]))  best_accuracy=$(round(best_accuracy, digits=5))  min_features=$min_features"

            if !isempty(front_objectives) && length(first(front_objectives)) >= 3
                min_time = minimum(Float64(objective[3]) for objective in front_objectives)
                log_message *= "  min_time=$(round(min_time, digits=5))"
            end

            println(log_message)
            flush(stdout)
        end
    end

    history = (
        pareto_front_population_hist = pareto_front_population_hist,
        front_size_hist = front_size_hist,
        population_hist = population_hist,
        offspring_hist = offspring_hist,
        parent_a_position_hist = parent_a_position_hist,
        parent_b_position_hist = parent_b_position_hist,
    )

    return (
        population = pop,
        objectives = objectives,
        fronts = fronts,
        rank = rank,
        crowding = crowding,
        evaluations = evaluations,
        history = history,
    )
end

end # module NSGA2Core
