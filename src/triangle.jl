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



