using HDF5
using Statistics

function parse_dataset(path::AbstractString, num_features::Integer; name::AbstractString = "unknown")
    num_features > 0 || throw(ArgumentError("num_features must be positive"))

    accuracy = read_mean_values(path, "accuracies")
    time = read_mean_values(path, "times")
    expected_subsets = (1 << num_features) - 1

    length(accuracy) == expected_subsets || throw(ArgumentError("expected $expected_subsets subsets for $num_features features, got $(length(accuracy))"))
    length(time) == expected_subsets || throw(ArgumentError("expected $expected_subsets time values for $num_features features, got $(length(time))"))

    indices = collect(1:expected_subsets)
    return Landscape(name, Int(num_features), indices, count_ones.(indices), accuracy, time, false)
end

function write_csv(landscape::Landscape, output_path::AbstractString)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, "index,num_features,mean_accuracy,mean_time")
        for position in eachindex(landscape.indices)
            println(io, join((
                landscape.indices[position],
                landscape.num_selected[position],
                landscape.accuracy[position],
                landscape.time[position],
            ), ','))
        end
    end

    return output_path
end

function read_mean_values(path::AbstractString, dataset_name::AbstractString)
    dataset = h5open(path, "r") do file
        read(file[dataset_name])
    end

    ndims(dataset) == 2 || throw(ArgumentError("expected a 2D `$dataset_name` dataset"))
    return Float64.(vec(mean(dataset; dims = 2)))
end
