# Used same function as provided in Lab lecture
function triangle_fitness(x; m=1, s=4)
    n_ones = sum(x)
    i = ceil(Int, n_ones / s)

    if i % 2 == 1
        if n_ones % s == 0
            return m * s
        else
            return m * (n_ones % s)
        end
    else
        return m * (i * s - n_ones)
    end
end

function triangle_landscape(; n::Integer = 16, m::Real = 1, s::Integer = 4, name::AbstractString = "triangle")
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    s > 0 || throw(ArgumentError("s must be positive"))

    indices = collect(0:((1 << n) - 1))
    num_selected = count_ones.(indices)
    values = [
        triangle_fitness(digits(index, base=2, pad=n); m=m, s=s)
        for index in indices
    ]
    times = zeros(Float64, length(indices))

    return Landscape(name, Int(n), indices, num_selected, Float64.(values), times, true)
end

const TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS = UInt8[
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,
    0, 1, 2, 3, 4, 5, 4, 3, 2, 1,
    0, 6,
]

const TRIANGLE_ASYM_MAGIC = UInt8[0x54, 0x41, 0x53, 0x59]

function countActiveBits(position::Integer, nSize::Integer)
    position >= 0 || throw(ArgumentError("position must be non-negative"))
    nSize >= 0 || throw(ArgumentError("nSize must be non-negative"))
    changeablePosition = UInt64(position)
    countActive = 0

    for i in 0:(Int(nSize) - 1)
        shift = UInt64(1) << (Int(nSize) - i - 1)

        if shift > changeablePosition
            continue
        else
            changeablePosition -= shift
            countActive += 1
        end
    end

    return countActive
end

function triangle_asym_fitness(active_bits::Integer;
                               values::AbstractVector{UInt8} = TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS)
    0 <= active_bits < length(values) || throw(ArgumentError("active_bits must be between 0 and $(length(values) - 1)"))
    return values[Int(active_bits) + 1]
end

function triangle_asym_global_optimum_index(n::Integer = 31)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    return (Int(1) << Int(n)) - 1
end

function triangle_asym_hamming_distance(index::Integer, n::Integer = 31)
    0 <= index <= triangle_asym_global_optimum_index(n) ||
        throw(ArgumentError("index must be between 0 and $(triangle_asym_global_optimum_index(n))"))
    return count_ones(xor(Int(index), triangle_asym_global_optimum_index(n)))
end

function triangle_asym_local_optimum_counts(n::Integer = 31)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    return Int[count for count in 0:Int(n) if triangle_asym_fitness(count) >= triangle_asym_fitness(max(count - 1, 0)) &&
                                             triangle_asym_fitness(count) >= triangle_asym_fitness(min(count + 1, Int(n)))]
end

function triangle_asym_local_optimum_multiplicity(active_bits::Integer, n::Integer = 31)
    0 <= active_bits <= n || throw(ArgumentError("active_bits must be between 0 and $n"))
    return binomial(BigInt(n), BigInt(active_bits))
end

function triangle_asym_fitness_by_count(n::Integer = 31;
                                        values::AbstractVector{UInt8} = TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS)
    0 <= n <= length(values) - 1 || throw(ArgumentError("n must be between 0 and $(length(values) - 1)"))
    return Float64[Float64(values[count + 1]) for count in 0:Int(n)]
end

function triangle_asym_generate_table(; n::Integer = 31,
                                      values::AbstractVector{UInt8} = TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS,
                                      threaded::Bool = Threads.nthreads() > 1)
    0 <= n <= length(values) - 1 || throw(ArgumentError("n must be between 0 and $(length(values) - 1)"))
    state_count = Int(1) << Int(n)
    table = Vector{UInt8}(undef, state_count)

    if threaded && Threads.nthreads() > 1 && state_count > 1
        Threads.@threads for position in 0:(state_count - 1)
            active_bits = countActiveBits(position, n)
            table[position + 1] = triangle_asym_fitness(active_bits; values=values)
        end
    else
        for position in 0:(state_count - 1)
            active_bits = countActiveBits(position, n)
            table[position + 1] = triangle_asym_fitness(active_bits; values=values)
        end
    end

    return table
end

function write_triangle_asym_table(table::AbstractVector{UInt8},
                                   output_path::AbstractString;
                                   n::Integer = 31)
    expected_length = Int(1) << Int(n)
    length(table) == expected_length ||
        throw(ArgumentError("table length must be $(expected_length) for n=$(n)"))

    mkpath(dirname(output_path))
    open(output_path, "w") do io
        write(io, TRIANGLE_ASYM_MAGIC)
        write(io, UInt8(n))
        write(io, table)
    end

    return output_path
end

function read_triangle_asym_table(input_path::AbstractString)
    open(input_path, "r") do io
        magic = read!(io, Vector{UInt8}(undef, length(TRIANGLE_ASYM_MAGIC)))
        magic == TRIANGLE_ASYM_MAGIC || error("Unexpected triangle table header in $input_path")
        n = Int(read(io, UInt8))
        expected_length = Int(1) << n
        table = read!(io, Vector{UInt8}(undef, expected_length))
        length(table) == expected_length ||
            error("Triangle table at $input_path has length $(length(table)), expected $(expected_length)")
        return n, table
    end
end

function triangle_asym_landscape(; n::Integer = 31,
                                 name::AbstractString = "triangle-asym",
                                 input_path::Union{Nothing, AbstractString} = nothing,
                                 force_regenerate::Bool = false,
                                 threaded::Bool = Threads.nthreads() > 1)
    path = isnothing(input_path) ? default_triangle_table_path(String(name)) : String(input_path)

    if !force_regenerate && isfile(path)
        loaded_n, table = read_triangle_asym_table(path)
        loaded_n == Int(n) || error("Triangle table at $path uses n=$(loaded_n), expected n=$(n)")
        return TriangleByteLandscape(name, n, table, true)
    end

    table = triangle_asym_generate_table(; n=n, threaded=threaded)
    write_triangle_asym_table(table, path; n=n)
    return TriangleByteLandscape(name, n, table, true)
end

function triangle_asym_sample_local_optima_indices(; n::Integer = 31,
                                                   max_count::Int = 49)
    0 <= n <= length(TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS) - 1 ||
        throw(ArgumentError("n must be between 0 and $(length(TRIANGLE_ASYM_FITNESS_BY_ACTIVE_BITS) - 1)"))
    max_count >= 0 || throw(ArgumentError("max_count must be non-negative"))
    max_count == 0 && return Int[]

    local_only_counts = [count for count in triangle_asym_local_optimum_counts(n) if count < n]
    isempty(local_only_counts) && return Int[]

    active_bits = maximum(local_only_counts)
    sample = Int[]
    zero_positions = Int[]
    one_positions = Int[]
    zero_count = Int(n) - active_bits

    function push_index_from_zero_positions!()
        index = triangle_asym_global_optimum_index(n)
        for zero_position in zero_positions
            index &= ~(Int(1) << zero_position)
        end
        push!(sample, index)
    end

    function push_index_from_one_positions!()
        index = 0
        for one_position in one_positions
            index |= Int(1) << one_position
        end
        push!(sample, index)
    end

    function visit(start::Int, remaining::Int)
        length(sample) >= max_count && return

        if remaining == 0
            if zero_count <= active_bits
                push_index_from_zero_positions!()
            else
                push_index_from_one_positions!()
            end
            return
        end

        max_start = Int(n) - remaining
        for position in start:max_start
            if zero_count <= active_bits
                push!(zero_positions, position)
            else
                push!(one_positions, position)
            end
            visit(position + 1, remaining - 1)
            if zero_count <= active_bits
                pop!(zero_positions)
            else
                pop!(one_positions)
            end
            length(sample) >= max_count && return
        end
    end

    visit(0, min(zero_count, active_bits))
    sort!(sample)
    return sample
end

function triangle_asym_hbm_nodes(landscape::TriangleByteLandscape; max_local_optima::Int = 49)
    local_indices = triangle_asym_sample_local_optima_indices(; n=landscape.num_features, max_count=max_local_optima)
    global_indices = [triangle_asym_global_optimum_index(landscape.num_features)]
    sample_indices = vcat(local_indices, global_indices)
    sample_values = [fitness(landscape, index) for index in sample_indices]

    return (
        nodes = build_hbm(sample_indices, sample_values, landscape.num_features),
        local_indices = local_indices,
        global_indices = global_indices,
    )
end
