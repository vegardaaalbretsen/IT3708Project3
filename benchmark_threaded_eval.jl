using IT3708Project3
using Random
using Statistics
using Base.Threads

function usage()
    println("Usage: julia --threads auto --project=. benchmark_threaded_eval.jl [ga|nsga2|swarm|all] [work] [repeats]")
    println("")
    println("Examples:")
    println("  julia --threads auto --project=. benchmark_threaded_eval.jl")
    println("  julia --threads auto --project=. benchmark_threaded_eval.jl all 4000 3")
    println("  julia --threads auto --project=. benchmark_threaded_eval.jl ga 8000 5")
end

function heavy_scalar_objective(bits::BitVector, work::Int)
    ones_count = count(bits)
    total = 0.0

    @inbounds for rep in 1:work
        x = ones_count + 0.0001 * rep
        total += sin(x) * cos(0.7 * x) + sqrt(x + 1.0)
    end

    return total
end

function heavy_nsga2_objective(bits::BitVector, work::Int)
    ones_count = count(bits)
    quality = heavy_scalar_objective(bits, work)
    time_like = 0.0

    @inbounds for rep in 1:work
        y = ones_count + 0.0002 * rep
        time_like += abs(sin(1.3 * y)) + abs(cos(0.9 * y))
    end

    return (quality, ones_count, time_like / work)
end

function heavy_swarm_objective(position::AbstractVector, work::Int)
    total = 0.0

    @inbounds for rep in 1:work
        offset = 0.0001 * rep
        for value in position
            x = Float64(value) + offset
            total += sin(x) * cos(0.5 * x) + sqrt(abs(x) + 1.0)
        end
    end

    return total
end

function timed_runs(run_once::Function, repeats::Int)
    times = Float64[]

    for _ in 1:repeats
        GC.gc()
        push!(times, @elapsed run_once())
    end

    return times
end

function report_benchmark(label::AbstractString, serial_times, threaded_times)
    serial_best = minimum(serial_times)
    threaded_best = minimum(threaded_times)
    serial_mean = mean(serial_times)
    threaded_mean = mean(threaded_times)
    speedup = serial_best / threaded_best

    println(label)
    println("  serial best:    $(round(serial_best; digits=4)) s")
    println("  threaded best:  $(round(threaded_best; digits=4)) s")
    println("  serial mean:    $(round(serial_mean; digits=4)) s")
    println("  threaded mean:  $(round(threaded_mean; digits=4)) s")
    println("  best speedup:   $(round(speedup; digits=2))x")
end

function benchmark_ga(work::Int, repeats::Int)
    nbits = 64
    popsize = 512
    generations = 40
    params(threaded) = IT3708Project3.GACore.GAParams(
        popsize=popsize,
        generations=generations,
        pc=0.9,
        pm=1 / nbits,
        seed=123,
        objective=:max,
        record_history=false,
        threaded_evaluation=threaded,
    )

    run_once(threaded) = IT3708Project3.GACore.run_ga(
        nbits,
        ind -> heavy_scalar_objective(ind, work);
        params=params(threaded),
    )

    run_once(false)
    serial_times = timed_runs(() -> run_once(false), repeats)
    threaded_times = timed_runs(() -> run_once(true), repeats)
    report_benchmark("GA population evaluation benchmark", serial_times, threaded_times)
end

function benchmark_nsga2(work::Int, repeats::Int)
    nbits = 64
    popsize = 256
    generations = 20
    params(threaded) = NSGA2Params(
        popsize=popsize,
        generations=generations,
        pc=0.9,
        pm=1 / nbits,
        seed=123,
        record_history=false,
        threaded_evaluation=threaded,
    )

    run_once(threaded) = run_nsga2(
        nbits,
        ind -> heavy_nsga2_objective(ind, work);
        params=params(threaded),
        directions=(:max, :min, :min),
    )

    run_once(false)
    serial_times = timed_runs(() -> run_once(false), repeats)
    threaded_times = timed_runs(() -> run_once(true), repeats)
    report_benchmark("NSGA-II population evaluation benchmark", serial_times, threaded_times)
end

function benchmark_swarm(work::Int, repeats::Int)
    nbits = 64
    swarm_size = 256
    iterations = 60

    function run_once(threaded)
        objective = position -> heavy_swarm_objective(position, work)
        init_rng = MersenneTwister(123)
        update_rng = MersenneTwister(456)
        population = IT3708Project3.initial_swarm_population(
            objective,
            nbits,
            swarm_size;
            rng=init_rng,
            threaded_evaluation=threaded,
        )

        return IT3708Project3.traced_swarm_pso(
            objective,
            population,
            iterations;
            w=0.7,
            c1=1.4,
            c2=1.4,
            rng=update_rng,
            n_features=nbits,
            allow_zero=true,
            keep_history=false,
            threaded_evaluation=threaded,
        )
    end

    run_once(false)
    serial_times = timed_runs(() -> run_once(false), repeats)
    threaded_times = timed_runs(() -> run_once(true), repeats)
    report_benchmark("Swarm particle evaluation benchmark", serial_times, threaded_times)
end

if any(arg -> arg in ("-h", "--help"), ARGS)
    usage()
    exit()
end

mode = length(ARGS) >= 1 ? ARGS[1] : "all"
work = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000
repeats = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3

mode in ("ga", "nsga2", "swarm", "all") || error("mode must be one of: ga, nsga2, swarm, all")
work > 0 || error("work must be positive")
repeats > 0 || error("repeats must be positive")

println("Julia threads: $(Threads.nthreads())")
println("Work factor: $work")
println("Repeats: $repeats")

Threads.nthreads() > 1 || println("Run this script with `julia --threads auto --project=. ...` to enable threaded evaluation.")

mode in ("ga", "all") && benchmark_ga(work, repeats)
mode in ("nsga2", "all") && benchmark_nsga2(work, repeats)
mode in ("swarm", "all") && benchmark_swarm(work, repeats)
