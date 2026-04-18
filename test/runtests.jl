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

@testset "Standard EA" begin
    tiny = Landscape(
        "tiny-ea",
        2,
        collect(0:3),
        count_ones.(collect(0:3)),
        [0.0, 0.8, 0.9, 1.0],
        zeros(4),
        true,
    )

    rng = MersenneTwister(7)
    for _ in 1:100
        child = standard_bit_mutation(1, 2; rng=rng, allow_zero=false)
        @test 1 <= child <= 3
    end

    raw_result = run_standard_ea(
        tiny;
        iterations=20,
        epsilon=0.0,
        initial_index=0,
        rng=MersenneTwister(11),
        keep_history=true,
    )

    @test raw_result.best_index == 3
    @test raw_result.best_accuracy == 1.0
    @test raw_result.best_penalized_fitness == 1.0
    @test raw_result.best_num_selected == 2
    @test length(raw_result.current_history) == 21
    @test length(raw_result.best_history) == 21
    @test length(raw_result.current_num_selected_history) == 21
    @test length(raw_result.best_num_selected_history) == 21
    @test raw_result.current_index_history[1] == 0
    @test raw_result.best_index_history[end] == 3

    penalized_result = run_standard_ea(
        tiny;
        iterations=20,
        epsilon=0.15,
        initial_index=3,
        rng=MersenneTwister(5),
    )

    @test penalized_result.best_index == 2
    @test penalized_result.best_accuracy == 0.9
    @test isapprox(penalized_result.best_penalized_fitness, 0.75; atol=1e-12)
    @test penalized_result.best_num_selected == 1
    @test penalized_result.current_history === nothing
    @test penalized_result.best_history === nothing
    @test penalized_result.current_num_selected_history === nothing
    @test penalized_result.best_num_selected_history === nothing

    trace_data = ea_trace_plot_data(raw_result)
    @test trace_data.iterations == collect(0:20)
    @test trace_data.current_fitness == raw_result.current_history
    @test trace_data.best_num_selected == raw_result.best_num_selected_history

    output_png = tempname() * ".png"
    exported_path = save_ea_trace_plot(raw_result, output_png; title="EA Trace Test")
    @test exported_path == output_png
    @test isfile(output_png)
    @test filesize(output_png) > 0
end
