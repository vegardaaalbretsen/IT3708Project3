using IT3708Project3
using Plots

function print_usage()
    println("Usage: julia --project=. plot_triangle.jl [n] [m] [s] [output]")
    println("")
    println("Defaults:")
    println("  n = 16")
    println("  m = 1")
    println("  s = 4")
    println("  output = triangle_plot.png")
end

function parse_args(args)
    any(arg -> arg in ("-h", "--help"), args) && return nothing

    n = length(args) >= 1 ? parse(Int, args[1]) : 16
    m = length(args) >= 2 ? parse(Int, args[2]) : 1
    s = length(args) >= 3 ? parse(Int, args[3]) : 4
    output = length(args) >= 4 ? args[4] : "triangle_plot.png"

    return n, m, s, output
end

function plot_triangle(n::Integer, m::Integer, s::Integer, output::AbstractString)
    weights = collect(0:n)
    fitness = [triangle_fitness(weight, m, s) for weight in weights]
    multiplicity = [binomial(n, weight) for weight in weights]
    peak_weights = weights[fitness .== maximum(fitness)]

    default(size = (950, 700), legend = :topright)

    p1 = plot(
        weights,
        fitness;
        marker = :circle,
        linewidth = 2,
        xlabel = "Hamming weight ||b||",
        ylabel = "Triangle fitness",
        title = "Triangle fitness by Hamming weight",
        label = "Triangle(||b||, m=$m, s=$s)",
    )
    vline!(p1, peak_weights; linestyle = :dash, alpha = 0.35, color = :red, label = "peak weights")

    p2 = bar(
        weights,
        multiplicity;
        xlabel = "Hamming weight ||b||",
        ylabel = "Number of states",
        title = "How many bitstrings share each weight",
        label = "binomial($n, ||b||)",
        alpha = 0.8,
    )
    vline!(p2, peak_weights; linestyle = :dash, alpha = 0.35, color = :red, label = "peak weights")

    combined = plot(p1, p2; layout = (2, 1))
    savefig(combined, output)

    println("Saved plot to $(abspath(output))")
    println("Peak weights: $peak_weights")
    println("Fitness values: $fitness")
end

parsed = parse_args(ARGS)
if parsed === nothing
    print_usage()
else
    n, m, s, output = parsed
    plot_triangle(n, m, s, output)
end
