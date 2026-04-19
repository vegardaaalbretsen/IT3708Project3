using EvoLP
using Random

function bounded_swarm_coordinate(value)
    value = Float64(value)
    return isfinite(value) ? clamp(value, 0.0, 1.0) : 0.0
end

function bounded_swarm_position(position::AbstractVector)
    return bounded_swarm_coordinate.(position)
end

function decode_swarm_position(position::AbstractVector,
                               n_features::Integer;
                               allow_zero::Bool = false)
    n = Int(n_features)
    n >= 0 || throw(ArgumentError("n_features must be non-negative"))
    length(position) == n || throw(ArgumentError("Expected position length $n, got $(length(position))"))

    if n == 0
        allow_zero || throw(ArgumentError("Cannot decode a zero-feature position when allow_zero is false"))
        return 0
    end

    index = 0
    best_feature = 1
    best_value = -Inf

    for i in 1:n
        value = bounded_swarm_coordinate(position[i])

        if value > best_value
            best_value = value
            best_feature = i
        end

        if value >= 0.5
            index |= 1 << (i - 1)
        end
    end

    if !allow_zero && index == 0
        index = 1 << (best_feature - 1)
    end

    return index
end

function initial_swarm_population(objective::Function,
                                  n_features::Integer,
                                  swarm_size::Integer;
                                  rng::AbstractRNG)
    n = Int(n_features)
    swarm_size = Int(swarm_size)
    swarm_size > 0 || throw(ArgumentError("swarm_size must be positive"))

    population = Vector{EvoLP.Particle}(undef, swarm_size)

    for i in 1:swarm_size
        position = rand(rng, n)
        value = objective(position)
        population[i] = EvoLP.Particle(position, zeros(n), value, copy(position), value)
    end

    return population
end

function particle_indices(population::AbstractVector{EvoLP.Particle},
                          n_features::Integer;
                          allow_zero::Bool = false)
    return [
        decode_swarm_position(particle.x, n_features; allow_zero=allow_zero)
        for particle in population
    ]
end

function traced_swarm_pso(objective::Function,
                          population::Vector{EvoLP.Particle},
                          iterations::Integer;
                          w::Real,
                          c1::Real,
                          c2::Real,
                          rng::AbstractRNG,
                          n_features::Integer,
                          allow_zero::Bool,
                          keep_history::Bool)
    d = length(population[1].x)
    x_best, y_best = copy(population[1].x_best), Inf
    particle_index_history = keep_history ? Vector{Vector{Int}}() : nothing
    best_index_history = keep_history ? Int[] : nothing
    best_penalized_fitness_history = keep_history ? Float64[] : nothing

    function record_history!()
        keep_history || return

        push!(particle_index_history, particle_indices(population, n_features; allow_zero=allow_zero))
        push!(best_index_history, decode_swarm_position(x_best, n_features; allow_zero=allow_zero))
        push!(best_penalized_fitness_history, -Float64(y_best))
    end

    # EvoLP.PSO does not expose intermediate swarm states, so we mirror its update rule here
    # to support swarm trace plots, overlays, and animations.
    start_time = time()

    for particle in population
        particle.y = objective(particle.x)
        particle.x_best[:] = particle.x
        particle.y_best = particle.y

        if particle.y < y_best
            x_best[:] = particle.x
            y_best = particle.y
        end
    end

    record_history!()

    @inbounds for _ in 1:Int(iterations)
        for particle in population
            r1 = rand(rng, d)
            r2 = rand(rng, d)
            particle.x += particle.v
            particle.v = @fastmath Float64(w) * particle.v +
                         Float64(c1) * r1 .* (particle.x_best - particle.x) +
                         Float64(c2) * r2 .* (x_best - particle.x)
            particle.y = objective(particle.x)

            if particle.y < y_best
                x_best[:] = particle.x
                y_best = particle.y
            end

            if particle.y < particle.y_best
                particle.x_best[:] = particle.x
                particle.y_best = particle.y
            end
        end

        record_history!()
    end

    best_i = argmin([particle.y_best for particle in population])
    best = population[best_i]
    raw_result = EvoLP.Result(
        best.y_best,
        best.x_best,
        population,
        Int(iterations),
        (1 + Int(iterations)) * length(population),
        time() - start_time,
    )

    return (
        raw_result = raw_result,
        particle_index_history = particle_index_history,
        best_index_history = best_index_history,
        best_penalized_fitness_history = best_penalized_fitness_history,
    )
end

function run_swarm_ea(landscape::Landscape;
                      iterations::Integer,
                      epsilon::Real = 0.0,
                      swarm_size::Integer = 40,
                      w::Real = 0.7,
                      c1::Real = 1.4,
                      c2::Real = 1.4,
                      keep_history::Bool = false,
                      rng::AbstractRNG = Random.default_rng())
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    swarm_size > 0 || throw(ArgumentError("swarm_size must be positive"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))

    for (name, value) in (("w", w), ("c1", c1), ("c2", c2))
        isfinite(value) || throw(ArgumentError("$name must be finite"))
        value >= 0 || throw(ArgumentError("$name must be non-negative"))
    end

    if landscape.num_features == 0 && !landscape.allow_zero
        throw(ArgumentError("Cannot optimize a zero-feature landscape when allow_zero is false"))
    end

    objective = function (position::AbstractVector)
        index = decode_swarm_position(position, landscape.num_features; allow_zero=landscape.allow_zero)
        return -candidate_state(landscape, index, epsilon).penalized_fitness
    end

    population = initial_swarm_population(objective, landscape.num_features, swarm_size; rng=rng)
    traced = traced_swarm_pso(
        objective,
        population,
        iterations;
        w=w,
        c1=c1,
        c2=c2,
        rng=rng,
        n_features=landscape.num_features,
        allow_zero=landscape.allow_zero,
        keep_history=keep_history,
    )
    raw_result = traced.raw_result

    best_position = bounded_swarm_position(EvoLP.optimizer(raw_result))
    best_index = decode_swarm_position(best_position, landscape.num_features; allow_zero=landscape.allow_zero)
    best = candidate_state(landscape, best_index, epsilon)
    final_particle_indices = particle_indices(EvoLP.population(raw_result), landscape.num_features; allow_zero=landscape.allow_zero)

    return (
        landscape = landscape.name,
        iterations = Int(iterations),
        epsilon = Float64(epsilon),
        swarm_size = Int(swarm_size),
        w = Float64(w),
        c1 = Float64(c1),
        c2 = Float64(c2),
        evaluations = EvoLP.f_calls(raw_result),
        runtime = EvoLP.runtime(raw_result),
        best_position = best_position,
        best_objective = Float64(EvoLP.optimum(raw_result)),
        final_particle_indices = final_particle_indices,
        best_index = best.index,
        best_accuracy = best.accuracy,
        best_penalized_fitness = best.penalized_fitness,
        best_num_selected = best.num_selected,
        particle_index_history = keep_history ? traced.particle_index_history : nothing,
        best_index_history = keep_history ? traced.best_index_history : nothing,
        best_penalized_fitness_history = keep_history ? traced.best_penalized_fitness_history : nothing,
    )
end
