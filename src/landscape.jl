function load_landscape(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Empty CSV file: $path")
    lines[1] == "index,num_features,mean_accuracy,mean_time" || error("Unexpected CSV header in $path")

    indices = Int[]
    num_features = Int[]
    mean_accuracy = Float64[]
    mean_time = Float64[]

    for line in lines[2:end]
        values = split(line, ',')
        length(values) == 4 || error("Expected 4 columns in $path, got $(length(values))")

        push!(indices, parse(Int, values[1]))
        push!(num_features, parse(Int, values[2]))
        push!(mean_accuracy, parse(Float64, values[3]))
        push!(mean_time, parse(Float64, values[4]))
    end

    return (
        indices = indices,
        num_features = num_features,
        mean_accuracy = mean_accuracy,
        mean_time = mean_time,
    )
end

function penalty(num_features::Integer, epsilon::Real)
    num_features >= 0 || throw(ArgumentError("num_features must be non-negative"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))
    return Float64(epsilon) * num_features
end

function penalized_fitness(mean_accuracy::Real, num_features::Integer, epsilon::Real)
    return Float64(mean_accuracy) - penalty(num_features, epsilon)
end

function apply_penalty(landscape, epsilon::Real)
    penalties = [penalty(count, epsilon) for count in landscape.num_features]

    return (
        indices = landscape.indices,
        num_features = landscape.num_features,
        mean_accuracy = landscape.mean_accuracy,
        mean_time = landscape.mean_time,
        penalties = penalties,
        fitness = landscape.mean_accuracy .- penalties,
        epsilon = Float64(epsilon),
    )
end
