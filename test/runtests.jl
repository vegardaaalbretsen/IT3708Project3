using IT3708Project3
using Random
using Test

@testset "Real landscape parsing" begin
    dataset = IT3708Project3.DATASETS["breast-w"]
    data = IT3708Project3.parse_dataset(dataset.path, dataset.num_features; name="breast-w")

    @test data isa Landscape
    @test data.name == "breast-w"
    @test data.num_features == 9
    @test data.allow_zero == false
    @test length(data.indices) == 511
    @test isapprox(data.accuracy[1], 0.7878048419952393; atol=1e-9)
    @test isapprox(data.accuracy[end], 0.9672255516052246; atol=1e-9)
    @test isapprox(data.time[1], 0.08813527971506119; atol=1e-9)
    @test data.num_selected[1] == 1
    @test data.num_selected[3] == 2

    output_csv = tempname() * ".csv"
    IT3708Project3.write_csv(data, output_csv)
    lines = readlines(output_csv)
    @test lines[1] == "index,num_features,mean_accuracy,mean_time"
    @test startswith(lines[2], "1,1,")
    @test length(lines) == 512
end

@testset "CSV landscape" begin
    dataset = IT3708Project3.DATASETS["breast-w"]
    parsed = IT3708Project3.parse_dataset(dataset.path, dataset.num_features; name="breast-w")
    output_csv = tempname() * ".csv"
    IT3708Project3.write_csv(parsed, output_csv)

    landscape = IT3708Project3.load_landscape(output_csv, dataset.num_features; name="breast-w")

    @test landscape isa Landscape
    @test landscape.name == "breast-w"
    @test landscape.num_features == 9
    @test landscape.allow_zero == false
    @test length(landscape.indices) == 511
    @test landscape.indices[1] == 1
    @test landscape.num_selected[1] == 1
    @test isapprox(landscape.accuracy[1], parsed.accuracy[1]; atol=1e-12)
    @test isapprox(landscape.time[1], parsed.time[1]; atol=1e-12)
    @test isapprox(IT3708Project3.fitness(landscape, 1), parsed.accuracy[1]; atol=1e-12)
    @test_throws ArgumentError IT3708Project3.fitness(landscape, 0)
    @test isapprox(IT3708Project3.penalty(3, 0.1), 0.3; atol=1e-12)
    @test isapprox(IT3708Project3.penalized_fitness(0.8, 3, 0.1), 0.5; atol=1e-12)
    @test isapprox(IT3708Project3.penalized_fitness_values(landscape, 0.1)[1], landscape.accuracy[1] - 0.1; atol=1e-12)
end

@testset "Triangle landscape" begin
    landscape = triangle_landscape()

    @test landscape isa Landscape
    @test landscape.name == "triangle"
    @test landscape.num_features == 16
    @test landscape.allow_zero == true
    @test length(landscape.indices) == 65536
    @test first(landscape.indices) == 0
    @test last(landscape.indices) == 65535
    @test all(==(0.0), landscape.time)
    @test IT3708Project3.fitness(landscape, 0) == 0.0
    @test IT3708Project3.fitness(landscape, 15) == 4.0
end

@testset "HBM" begin
    @test sort(one_flip_neighbors(1, 4)) == [3, 5, 9]
    @test sort(one_flip_neighbors(7, 4)) == [3, 5, 6, 15]
    @test sort(one_flip_neighbors(15, 4)) == [7, 11, 13, 14]

    nodes = build_hbm(collect(1:7), [0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.7], 3)
    plot_data = hbm_plot_data(nodes, 3)

    @test length(nodes) == 7
    @test nodes[1] == HBMNode(1, 0.1, 0, 1)
    @test nodes[7] == HBMNode(7, 0.7, 3, 1)
    @test sort(local_optima(nodes, 3)) == [5, 7]
    @test sort(global_optima(nodes)) == [5, 7]
    @test IT3708Project3.top_local_optima(nodes, [5, 6, 7], [7]; max_count=1) == [5]
    @test plot_data.x == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    @test plot_data.y == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    @test sort(plot_data.local_optima) == [5, 7]
    @test sort(plot_data.global_optima) == [5, 7]

    output_png = tempname() * ".png"
    exported_path = save_hbm_plot(nodes, 3, output_png; title="HBM Test")
    @test exported_path == output_png
    @test isfile(output_png)
    @test filesize(output_png) > 0
end

@testset "Landscape visualization methods" begin
    landscape = Landscape(
        "tiny",
        3,
        collect(0:7),
        count_ones.(collect(0:7)),
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.7],
        zeros(8),
        true,
    )

    nodes = build_hbm(landscape)
    plot_data = feature_count_plot_data(landscape)

    @test length(nodes) == 8
    @test nodes[1] == HBMNode(0, 0.0, 0, 0)
    @test sort(local_optima(landscape)) == [5, 7]
    @test sort(global_optima(landscape)) == [5, 7]
    @test plot_data.feature_counts == [0, 1, 2, 3]
    @test plot_data.max_fitness == [0.0, 0.4, 0.7, 0.7]

    output_png = tempname() * ".png"
    exported_path = save_fitness_by_feature_count_plot(landscape, output_png; title="Feature Count Test")
    @test exported_path == output_png
    @test isfile(output_png)
    @test filesize(output_png) > 0
end

@testset "Single-objective GA" begin
    tiny = Landscape(
        "tiny-ga",
        2,
        collect(0:3),
        count_ones.(collect(0:3)),
        [0.0, 0.8, 0.9, 1.0],
        zeros(4),
        true,
    )

    result = run_single_objective_ea(
        tiny;
        iterations=8,
        epsilon=0.0,
        initial_index=3,
        population_size=8,
        crossover_probability=0.85,
        mutation_probability=0.1,
        tournament_size=3,
        survivor_mode=:elitist,
        elite=2,
        rng=MersenneTwister(17),
        keep_history=true,
    )

    @test result.algorithm == :single_objective_ga
    @test result.best_index == 3
    @test result.best_accuracy == 1.0
    @test result.best_penalized_fitness == 1.0
    @test result.best_num_selected == 2
    @test result.initial_index == 3
    @test result.population_size == 8
    @test result.crossover_probability == 0.85
    @test result.mutation_probability == 0.1
    @test result.tournament_size == 3
    @test result.survivor_mode == :elitist
    @test result.elite == 2
    @test length(result.current_history) == 9
    @test length(result.best_history) == 9
    @test length(result.current_num_selected_history) == 9
    @test length(result.best_num_selected_history) == 9
    @test length(result.current_index_history) == 9
    @test length(result.best_index_history) == 9
    @test result.best_index_history[end] == 3
    @test result.mean_history !== nothing
    @test result.entropy_history !== nothing

    trace_data = ea_trace_plot_data(result)
    @test trace_data.iterations == collect(0:8)
    @test trace_data.current_fitness == result.current_history
    @test trace_data.best_num_selected == result.best_num_selected_history

    output_png = tempname() * ".png"
    exported_path = save_ea_trace_plot(result, output_png; title="GA Trace Test")
    @test exported_path == output_png
    @test isfile(output_png)
    @test filesize(output_png) > 0

    overlay_png = tempname() * ".png"
    overlay_path = save_fitness_by_feature_count_with_ea_plot(tiny, result, overlay_png; title="GA Feature Count Test")
    @test overlay_path == overlay_png
    @test isfile(overlay_png)
    @test filesize(overlay_png) > 0

    no_history = run_single_objective_ea(
        tiny;
        iterations=8,
        epsilon=0.0,
        initial_index=3,
        population_size=8,
        rng=MersenneTwister(17),
        keep_history=false,
    )

    @test no_history.best_index == 3
    @test no_history.initial_index == 3
    @test no_history.current_history === nothing
    @test no_history.best_history === nothing
    @test no_history.current_accuracy_history === nothing
    @test no_history.best_accuracy_history === nothing
    @test no_history.current_num_selected_history === nothing
    @test no_history.best_num_selected_history === nothing
    @test no_history.current_index_history === nothing
    @test no_history.best_index_history === nothing
    @test no_history.mean_history === nothing
    @test no_history.max_history === nothing
    @test no_history.min_history === nothing
    @test no_history.entropy_history === nothing

    direct_params = IT3708Project3.GACore.GAParams(
        popsize=3,
        generations=0,
        seed=1,
        objective=:max,
        record_history=false,
    )
    initial_population = BitVector[
        BitVector([false, false]),
        BitVector([true, false]),
        BitVector([true, true]),
    ]
    best_ind, best_raw, worst_ind, worst_raw, history = IT3708Project3.GACore.run_ga(
        2,
        ind -> Float64(count(ind)),
        initial_population;
        params=direct_params,
    )

    @test best_ind == BitVector([true, true])
    @test best_raw == 2.0
    @test worst_ind == BitVector([false, false])
    @test worst_raw == 0.0
    @test history.max_hist === nothing
    @test history.mean_hist === nothing
    @test history.min_hist === nothing
    @test history.ent_hist === nothing
    @test history.current_best_raw_hist === nothing
    @test history.current_best_ind_hist === nothing
    @test history.best_so_far_raw_hist === nothing
    @test history.best_so_far_ind_hist === nothing
    @test history.initial_best_ind == BitVector([true, true])
    @test history.initial_best_raw == 2.0
    @test history.final_best_ind == BitVector([true, true])
    @test history.final_best_raw == 2.0
end

@testset "Swarm EA" begin
    @test decode_swarm_position([0.6, 0.4, 0.8], 3; allow_zero=true) == 5
    @test decode_swarm_position([-1.0, 0.4, NaN], 3; allow_zero=false) == 2
    @test_throws ArgumentError decode_swarm_position([0.2, 0.8], 3; allow_zero=true)

    tiny = Landscape(
        "tiny-swarm",
        1,
        collect(0:1),
        count_ones.(collect(0:1)),
        [0.1, 0.9],
        zeros(2),
        true,
    )

    result = run_swarm_ea(
        tiny;
        iterations=10,
        epsilon=0.0,
        swarm_size=8,
        rng=MersenneTwister(21),
    )

    @test result.best_index == 1
    @test result.best_accuracy == 0.9
    @test result.best_penalized_fitness == 0.9
    @test result.best_num_selected == 1
    @test result.best_objective == -0.9
    @test result.evaluations == 88
    @test length(result.best_position) == 1
    @test all(0 .<= result.best_position .<= 1)
end

@testset "Swarm visualization methods" begin
    tiny = Landscape(
        "tiny-swarm-vis",
        2,
        collect(0:3),
        count_ones.(collect(0:3)),
        [0.1, 0.8, 0.9, 1.0],
        zeros(4),
        true,
    )

    result = run_swarm_ea(
        tiny;
        iterations=4,
        epsilon=0.1,
        swarm_size=5,
        rng=MersenneTwister(9),
        keep_history=true,
    )

    @test length(result.final_particle_indices) == 5
    @test length(result.particle_index_history) == 5
    @test length(result.best_index_history) == 5
    @test length(result.best_penalized_fitness_history) == 5
    @test result.best_penalized_fitness_history[end] == result.best_penalized_fitness

    trace_data = swarm_trace_plot_data(tiny, result)
    path_data = IT3708Project3.swarm_best_path_plot_data(tiny, result)
    @test trace_data.iterations == collect(0:4)
    @test length(trace_data.mean_fitness) == 5
    @test length(trace_data.median_fitness) == 5
    @test length(trace_data.unique_subsets) == 5
    @test length(trace_data.mean_pairwise_hamming_distance) == 5
    @test length(trace_data.global_best_fraction) == 5
    @test all(0 .<= trace_data.global_best_fraction)
    @test all(trace_data.global_best_fraction .<= 1)
    @test !isnothing(path_data)
    @test length(path_data.indices) == 5
    @test first(path_data.feature_path_positions) == 1
    @test first(path_data.hbm_path_positions) == 1

    feature_png = tempname() * ".png"
    feature_path = save_fitness_by_feature_count_with_swarm_plot(
        tiny,
        result,
        feature_png;
        size=(900, 600),
        dpi=120,
    )
    @test feature_path == feature_png
    @test isfile(feature_png)
    @test filesize(feature_png) > 0

    hbm_png = tempname() * ".png"
    hbm_path = save_hbm_with_swarm_plot(
        tiny,
        result,
        hbm_png;
        size=(1000, 700),
        dpi=120,
    )
    @test hbm_path == hbm_png
    @test isfile(hbm_png)
    @test filesize(hbm_png) > 0

    trace_png = tempname() * ".png"
    trace_path = save_swarm_trace_plot(
        tiny,
        result,
        trace_png;
        size=(900, 700),
        dpi=120,
    )
    @test trace_path == trace_png
    @test isfile(trace_png)
    @test filesize(trace_png) > 0

    gif_path = tempname() * ".gif"
    saved_gif = save_fitness_by_feature_count_swarm_animation(
        tiny,
        result,
        gif_path;
        fps=4,
        size=(700, 500),
        dpi=100,
    )
    @test saved_gif == gif_path
    @test isfile(gif_path)
    @test filesize(gif_path) > 0

    no_history = run_swarm_ea(
        tiny;
        iterations=4,
        epsilon=0.1,
        swarm_size=5,
        rng=MersenneTwister(9),
        keep_history=false,
    )
    @test_throws ArgumentError swarm_trace_plot_data(tiny, no_history)
    @test isnothing(IT3708Project3.swarm_best_path_plot_data(tiny, no_history))
end
