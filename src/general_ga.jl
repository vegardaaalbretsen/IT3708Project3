module GACore

using Random
using Statistics
using Base.Threads

export GAParams, run_ga, entropy_bits

Base.@kwdef mutable struct GAParams
    popsize::Int = 200
    generations::Int = 10000 
    pc::Float64 = 0.90  #crossover prob
    pm::Float64 = 0.0  #mutation prob
    tour_k::Int = 3  #tournament size
    survivor_mode::Symbol = :elitist  #elitist / generational
    elite::Int = 2  #number of elites kept
    seed::Int = 42
    objective::Symbol = :max
    log_every::Int = 0
    record_history::Bool = true
    threaded_evaluation::Bool = false

    # ----- generalized crowding -----
    crowding_alpha::Float64 = 0.0   # 0 = deterministic, 1 = probabilistic
    crowding_scale::Float64 = 1.0   # temperature / smoothness
end

# ----- utilities -----

random_individual(nbits::Int, rng::AbstractRNG) = BitVector(rand(rng, Bool, nbits))

function init_population(popsize::Int, nbits::Int, rng::AbstractRNG)
    [random_individual(nbits, rng) for _ in 1:popsize]
end

function objective_best_index(values::AbstractVector{<:Real}, objective::Symbol)
    objective === :max && return argmax(values)
    objective === :min && return argmin(values)
    error("objective must be :max or :min, got $objective")
end

function objective_worst_index(values::AbstractVector{<:Real}, objective::Symbol)
    objective === :max && return argmin(values)
    objective === :min && return argmax(values)
    error("objective must be :max or :min, got $objective")
end

# Convert "raw fitness" to "score to maximize"
@inline function to_score(f::Float64, objective::Symbol)
    objective === :max && return f
    objective === :min && return -f
    error("objective must be :max or :min, got $objective")
end

function tournament_select(pop::Vector{BitVector}, scores::Vector{Float64},
                           k::Int, rng::AbstractRNG)
    best = rand(rng, eachindex(pop))
    bests = scores[best]
    for _ in 2:k
        i = rand(rng, eachindex(pop))
        if scores[i] > bests
            best, bests = i, scores[i]
        end
    end
    return pop[best]
end

function crossover(p1::BitVector, p2::BitVector, pc::Float64, rng::AbstractRNG)
    n = length(p1)
    @assert length(p2) == n
    if n <= 1 || rand(rng) > pc
        return copy(p1), copy(p2)
    end
    cut = (n > 2) ? rand(rng, 2:(n-1)) : 1
    c1 = similar(p1); c2 = similar(p2)
    c1[1:cut] = p1[1:cut];      c1[(cut+1):n] = p2[(cut+1):n]
    c2[1:cut] = p2[1:cut];      c2[(cut+1):n] = p1[(cut+1):n]
    return c1, c2
end

function mutate!(x::BitVector, pm::Float64, rng::AbstractRNG)
    @inbounds for i in eachindex(x)
        if rand(rng) < pm
            x[i] = !x[i]
        end
    end
    return x
end

# Survivor selection A: generational
survivors_generational(_parents, offspring, _score_par, _score_off, popsize::Int) =
    offspring[1:popsize]

# Survivor selection B: elitist from parents+offspring (by score)
function survivors_elitist(
    parents::Vector{BitVector},
    offspring::Vector{BitVector},
    score_par::Vector{Float64},
    score_off::Vector{Float64},
    popsize::Int,
    k::Int
)
    k = clamp(k, 0, popsize)

    # k best parents survives
    pidx = sortperm(score_par, rev=true)
    elites = parents[pidx[1:k]]

    # fill the remaining with offspring
    remaining = popsize - k
    oidx = sortperm(score_off, rev=true)
    rest = offspring[oidx[1:remaining]]

    return vcat(elites, rest)
end


# ----- entropy (for Task 2 plots) -----
"""
Population entropy over bit positions.
Returns a Float64. Higher means more diversity.
"""
function entropy_bits(pop::Vector{BitVector})
    popsize = length(pop)
    nbits = length(pop[1])
    H = 0.0
    @inbounds for i in 1:nbits
        ones_count = 0
        for ind in pop
            ones_count += ind[i] ? 1 : 0
        end
        p = ones_count / popsize
        if p > 0 && p < 1
            H -= p*log2(p) + (1-p)*log2(1-p)
        end
    end
    return H
end

# ----- core GA runner -----
"""
run_ga(nbits, fitness_fn; params)

fitness_fn(ind::BitVector)::Float64 returns raw fitness.
If objective=:min, GA will minimize raw fitness (by maximizing -fitness).
Returns:
- best_ind: BitVector
- best_raw: Float64  (best raw fitness according to objective)
- history: named tuple with optional per-generation traces and scalar initial/final-best state
"""
function run_ga(nbits::Int, fitness_fn::Function; params::GAParams=GAParams())
    return run_ga(nbits, fitness_fn, nothing; params=params)
end

function evaluate_population(population::Vector{BitVector},
                             fitness_fn::Function,
                             objective::Symbol;
                             threaded_evaluation::Bool = false)
    raw = Vector{Float64}(undef, length(population))
    score = Vector{Float64}(undef, length(population))

    if threaded_evaluation && nthreads() > 1 && length(population) > 1
        Threads.@threads for i in 1:length(population)
            value = fitness_fn(population[i])
            raw[i] = value
            score[i] = to_score(value, objective)
        end
    else
        for i in eachindex(population)
            value = fitness_fn(population[i])
            raw[i] = value
            score[i] = to_score(value, objective)
        end
    end

    return raw, score
end

function run_ga(nbits::Int,
                fitness_fn::Function,
                initial_population::Union{Nothing, Vector{BitVector}};
                params::GAParams=GAParams())
    rng = MersenneTwister(params.seed)

    params.popsize > 0 || throw(ArgumentError("popsize must be positive"))
    params.generations >= 0 || throw(ArgumentError("generations must be non-negative"))
    0 <= params.pc <= 1 || throw(ArgumentError("pc must be between 0 and 1"))
    0 <= params.pm <= 1 || throw(ArgumentError("pm must be between 0 and 1"))
    params.tour_k > 0 || throw(ArgumentError("tour_k must be positive"))
    params.elite >= 0 || throw(ArgumentError("elite must be non-negative"))
    params.log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    pop = if isnothing(initial_population)
        init_population(params.popsize, nbits, rng)
    else
        length(initial_population) == params.popsize ||
            throw(ArgumentError("initial_population must have length $(params.popsize)"))
        [copy(ind) for ind in initial_population]
    end

    for ind in pop
        length(ind) == nbits || throw(ArgumentError("All individuals must have length $nbits"))
    end

    max_hist = params.record_history ? Float64[] : nothing
    mean_hist = params.record_history ? Float64[] : nothing
    min_hist = params.record_history ? Float64[] : nothing
    ent_hist = params.record_history ? Float64[] : nothing
    current_best_raw_hist = params.record_history ? Float64[] : nothing
    current_best_ind_hist = params.record_history ? BitVector[] : nothing
    best_so_far_raw_hist = params.record_history ? Float64[] : nothing
    best_so_far_ind_hist = params.record_history ? BitVector[] : nothing
    best_ind = BitVector()
    best_raw = 0.0
    worst_ind = BitVector()
    worst_raw = 0.0
    initial_best_ind = BitVector()
    initial_best_raw = 0.0
    final_best_position = 0
    final_best_raw = 0.0
    seen_population = false
    track_population_stats = params.record_history || params.log_every > 0

    better(a, b) = params.objective === :max ? (a > b) : (a < b)
    worse(a, b) = params.objective === :max ? (a < b) : (a > b)

    function record_population!(population::Vector{BitVector})
        raw, score = evaluate_population(
            population,
            fitness_fn,
            params.objective;
            threaded_evaluation=params.threaded_evaluation,
        )

        current_best_i = objective_best_index(raw, params.objective)
        current_worst_i = objective_worst_index(raw, params.objective)
        current_best_raw = raw[current_best_i]
        current_mean_raw = track_population_stats ? mean(raw) : 0.0
        current_entropy = track_population_stats ? entropy_bits(population) : 0.0

        if !seen_population
            best_raw = current_best_raw
            best_ind = copy(population[current_best_i])
            worst_raw = raw[current_worst_i]
            worst_ind = copy(population[current_worst_i])
            initial_best_ind = copy(population[current_best_i])
            initial_best_raw = current_best_raw
            seen_population = true
        else
            if better(current_best_raw, best_raw)
                best_raw = current_best_raw
                best_ind = copy(population[current_best_i])
            end

            if worse(raw[current_worst_i], worst_raw)
                worst_raw = raw[current_worst_i]
                worst_ind = copy(population[current_worst_i])
            end
        end

        final_best_position = current_best_i
        final_best_raw = current_best_raw

        if params.record_history
            push!(max_hist, maximum(raw))
            push!(mean_hist, current_mean_raw)
            push!(min_hist, minimum(raw))
            push!(ent_hist, current_entropy)
            push!(current_best_raw_hist, current_best_raw)
            push!(current_best_ind_hist, copy(population[current_best_i]))
            push!(best_so_far_raw_hist, best_raw)
            push!(best_so_far_ind_hist, copy(best_ind))
        end

        return raw, score, current_best_raw, current_mean_raw, current_entropy
    end

    raw, score, current_best_raw, current_mean_raw, current_entropy = record_population!(pop)

    for gen in 1:params.generations

        offspring = BitVector[]
        
        while length(offspring) < params.popsize
            p1 = tournament_select(pop, score, params.tour_k, rng)
            p2 = tournament_select(pop, score, params.tour_k, rng)
            
            c1, c2 = crossover(p1, p2, params.pc, rng)
            mutate!(c1, params.pm, rng)
            mutate!(c2, params.pm, rng)
            
            push!(offspring, c1)
            if length(offspring) < params.popsize
                push!(offspring, c2)
            end
        end

        raw_off, score_off = evaluate_population(
            offspring,
            fitness_fn,
            params.objective;
            threaded_evaluation=params.threaded_evaluation,
        )
        for i in eachindex(offspring)
            r = raw_off[i]

            if worse(r, worst_raw)
                worst_raw = r
                worst_ind = copy(offspring[i])
            end
            if better(r, best_raw)
                best_raw = r
                best_ind = copy(offspring[i])
            end
        end

        # --- survivor selection ---
        if params.survivor_mode == :generational
            pop = survivors_generational(pop, offspring, score, score_off, params.popsize)
        elseif params.survivor_mode == :elitist
            pop = survivors_elitist(pop, offspring, score, score_off, params.popsize, params.elite)
        elseif params.survivor_mode == :crowding
            error("survivor_mode=:crowding is not available in this project build")
        else
            error("Unknown survivor_mode: $(params.survivor_mode)")
        end

        raw, score, current_best_raw, current_mean_raw, current_entropy = record_population!(pop)

        if params.log_every > 0 && (gen == 1 || gen % params.log_every == 0 || gen == params.generations)
            println("gen=$gen  best=$(round(current_best_raw, digits=5))  mean=$(round(current_mean_raw, digits=5))  entropy=$(round(current_entropy, digits=3))")
            flush(stdout)
        end
    end

    history = (
        max_hist = max_hist,
        mean_hist = mean_hist,
        min_hist = min_hist,
        ent_hist = ent_hist,
        current_best_raw_hist = current_best_raw_hist,
        current_best_ind_hist = current_best_ind_hist,
        best_so_far_raw_hist = best_so_far_raw_hist,
        best_so_far_ind_hist = best_so_far_ind_hist,
        initial_best_ind = initial_best_ind,
        initial_best_raw = initial_best_raw,
        final_best_ind = copy(pop[final_best_position]),
        final_best_raw = final_best_raw,
        worst_raw = worst_raw,
    )
    return best_ind, best_raw, worst_ind, worst_raw, history
end

end # module 
