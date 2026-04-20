function load_landscape(path::AbstractString, num_features::Integer; name::AbstractString = "unknown")
    lines = readlines(path)
    isempty(lines) && error("Empty CSV file: $path")
    lines[1] == "index,num_features,mean_accuracy,mean_time" || error("Unexpected CSV header in $path")

    indices = Int[]
    num_selected = Int[]
    accuracy = Float64[]
    time = Float64[]

    for line in lines[2:end]
        values = split(line, ',')
        length(values) == 4 || error("Expected 4 columns in $path, got $(length(values))")

        push!(indices, parse(Int, values[1]))
        push!(num_selected, parse(Int, values[2]))
        push!(accuracy, parse(Float64, values[3]))
        push!(time, parse(Float64, values[4]))
    end

    return Landscape(name, Int(num_features), indices, num_selected, accuracy, time, false)
end

function load_landscape_key(key::AbstractString)
    if key == "triangle"
        return triangle_landscape()
    end

    haskey(DATASETS, key) || error("Unknown dataset key: $key")
    dataset = DATASETS[key]
    csv_path = default_output_path(key)

    if !isfile(csv_path)
        parsed = parse_dataset(dataset.path, dataset.num_features; name=key)
        write_csv(parsed, csv_path)
    end

    return load_landscape(csv_path, dataset.num_features; name=key)
end

function fitness_values(landscape::Landscape)
    return landscape.accuracy
end

function penalty(num_features::Integer, epsilon::Real)
    num_features >= 0 || throw(ArgumentError("num_features must be non-negative"))
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be between 0 and 1"))
    return Float64(epsilon) * num_features
end

function penalized_fitness(accuracy::Real, num_features::Integer, epsilon::Real)
    return Float64(accuracy) - penalty(num_features, epsilon)
end

function penalized_fitness_values(landscape::Landscape, epsilon::Real)
    return [
        penalized_fitness(accuracy, num_selected, epsilon)
        for (accuracy, num_selected) in zip(landscape.accuracy, landscape.num_selected)
    ]
end

function index_position(landscape::Landscape, index::Integer)
    index = Int(index)
    position = landscape.allow_zero ? index + 1 : index

    if position < 1 || position > length(landscape.indices) || landscape.indices[position] != index
        throw(ArgumentError("index $(index) is not in landscape `$(landscape.name)`"))
    end

    return position
end

function fitness(landscape::Landscape, index::Integer)
    return landscape.accuracy[index_position(landscape, index)]
end

function candidate_state(landscape::Landscape, index::Integer, epsilon::Real)
    position = index_position(landscape, index)
    accuracy = landscape.accuracy[position]
    time = landscape.time[position]
    num_selected = landscape.num_selected[position]
    objective = penalized_fitness(accuracy, num_selected, epsilon)

    return (
        index = Int(index),
        accuracy = accuracy,
        time = time,
        penalized_fitness = objective,
        num_selected = num_selected,
    )
end
