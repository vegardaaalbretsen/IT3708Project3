using HDF5
using Statistics

function parse_dataset(path::AbstractString, num_features::Integer)
    num_features > 0 || throw(ArgumentError("num_features must be positive"))

    mean_accuracy = read_mean_values(path, "accuracies")
    mean_time = read_mean_values(path, "times")
    expected_subsets = (1 << num_features) - 1

    length(mean_accuracy) == expected_subsets || throw(ArgumentError("expected $expected_subsets subsets for $num_features features, got $(length(mean_accuracy))"))
    length(mean_time) == expected_subsets || throw(ArgumentError("expected $expected_subsets time values for $num_features features, got $(length(mean_time))"))

    indices = collect(1:expected_subsets)

    return (
        indices = indices,
        num_features = count_ones.(indices),
        mean_accuracy = mean_accuracy,
        mean_time = mean_time,
    )
end

function write_csv(data, output_path::AbstractString)
    mkpath(dirname(output_path))

    open(output_path, "w") do io
        println(io, "index,num_features,mean_accuracy,mean_time")
        for position in eachindex(data.indices)
            println(io, join((
                data.indices[position],
                data.num_features[position],
                data.mean_accuracy[position],
                data.mean_time[position],
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
