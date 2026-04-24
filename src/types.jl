abstract type AbstractLandscape end

struct Landscape <: AbstractLandscape
    name::String
    num_features::Int
    indices::Vector{Int}
    num_selected::Vector{Int}
    accuracy::Vector{Float64}
    time::Vector{Float64}
    allow_zero::Bool

    function Landscape(name::AbstractString,
                       num_features::Integer,
                       indices::AbstractVector{<:Integer},
                       num_selected::AbstractVector{<:Integer},
                       accuracy::AbstractVector{<:Real},
                       time::AbstractVector{<:Real},
                       allow_zero::Bool)
        lengths = (length(indices), length(num_selected), length(accuracy), length(time))
        all(==(first(lengths)), lengths) || throw(ArgumentError("Landscape vectors must have the same length"))
        num_features >= 0 || throw(ArgumentError("num_features must be non-negative"))

        return new(
            String(name),
            Int(num_features),
            Int.(indices),
            Int.(num_selected),
            Float64.(accuracy),
            Float64.(time),
            allow_zero,
        )
    end
end

struct TriangleByteLandscape <: AbstractLandscape
    name::String
    num_features::Int
    fitness_table::Vector{UInt8}
    allow_zero::Bool

    function TriangleByteLandscape(name::AbstractString,
                                   num_features::Integer,
                                   fitness_table::AbstractVector{UInt8},
                                   allow_zero::Bool = true)
        num_features >= 0 || throw(ArgumentError("num_features must be non-negative"))
        expected_length = Int(1) << Int(num_features)
        length(fitness_table) == expected_length ||
            throw(ArgumentError("fitness_table length must be $(expected_length) for num_features=$(num_features)"))

        return new(
            String(name),
            Int(num_features),
            fitness_table isa Vector{UInt8} ? fitness_table : UInt8[byte for byte in fitness_table],
            allow_zero,
        )
    end
end

struct HBMNode
    index::Int
    fitness::Float64
    x::Int
    y::Int
end
