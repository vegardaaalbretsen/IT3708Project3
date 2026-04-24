using IT3708Project3

function usage()
    println("Usage: julia --project=. generate_triangle_asym.jl [output-path] [n]")
    println("")
    println("Examples:")
    println("  julia --project=. generate_triangle_asym.jl")
    println("  julia --project=. generate_triangle_asym.jl exports/triangle/triangle-asym.bin")
    println("  julia --project=. generate_triangle_asym.jl exports/triangle/triangle-asym-small.bin 8")
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

output_path = length(ARGS) >= 1 ? ARGS[1] : default_triangle_table_path("triangle-asym")
n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 31

landscape = triangle_asym_landscape(; n=n, input_path=output_path, force_regenerate=true)
println("Saved asymmetric triangle lookup table for n=$(landscape.num_features) to `$output_path`.")
