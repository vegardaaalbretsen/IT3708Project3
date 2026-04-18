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
