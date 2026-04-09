using IT3708Project3
using Test

@testset "Feature decoding" begin
    @test IT3708Project3.subset_to_bitvector(13, 4) == Bool[1, 0, 1, 1]
    @test IT3708Project3.subset_to_bitstring(13, 4) == "1011"
    @test IT3708Project3.subset_to_bitstring(13, 4; feature1_first = false) == "1101"
    @test IT3708Project3.active_columns(13, 4) == [1, 3, 4]
    @test IT3708Project3.feature_penalty(13; epsilon = 1 / 8) == 3 / 8
end

@testset "Real landscape parsing" begin
    dataset = IT3708Project3.DATASET_SPECS["breast-w"]
    landscape = IT3708Project3.read_feature_selection_landscape(dataset.path, dataset.n_features)
    best_raw = IT3708Project3.best_raw_subset(landscape)
    best_penalized = IT3708Project3.best_penalized_subset(landscape)

    @test length(landscape.subset_indices) == 511
    @test landscape.raw_accuracy_table[1] ≈ 0.7878048419952393 atol = 1e-9
    @test landscape.raw_accuracy_table[end] ≈ 0.9672255516052246 atol = 1e-9
    @test landscape.raw_time_table[1] ≈ 0.08813527971506119 atol = 1e-9
    @test landscape.penalty_table[1] == 1 / 8
    @test landscape.penalized_table[1] ≈ landscape.raw_accuracy_table[1] - 1 / 8 atol = 1e-12
    @test best_raw.raw_accuracy == maximum(landscape.raw_accuracy_table)
    @test best_penalized.penalized_fitness == maximum(landscape.penalized_table)
    @test best_penalized.n_active == length(best_penalized.active_columns)

    rows = IT3708Project3.feature_selection_rows(landscape)
    @test length(rows) == 511
    @test rows[1].index == 1
    @test rows[1].bitstring == "100000000"
    @test rows[1].num_active_features == 1
    @test rows[1].active_features == "1"
    @test rows[1].mean_accuracy ≈ landscape.raw_accuracy_table[1] atol = 1e-12
    @test rows[1].mean_time ≈ landscape.raw_time_table[1] atol = 1e-12

    output_csv = tempname() * ".csv"
    IT3708Project3.write_feature_selection_csv(landscape, output_csv)
    lines = readlines(output_csv)
    @test lines[1] == "index,bitstring,num_active_features,active_features,mean_accuracy,mean_time"
    @test startswith(lines[2], "1,100000000,1,1,")
    @test length(lines) == 512
end

@testset "Triangle landscape" begin
    @test IT3708Project3.triangle_fitness(0, 1, 4) == 0
    @test IT3708Project3.triangle_fitness(1, 1, 4) == 1
    @test IT3708Project3.triangle_fitness(4, 1, 4) == 4
    @test IT3708Project3.triangle_fitness(5, 1, 4) == 3
    @test IT3708Project3.triangle_fitness(8, 1, 4) == 0
    @test IT3708Project3.triangle_fitness(12, 1, 4) == 4

    landscape = IT3708Project3.triangle_landscape()
    @test length(landscape.subset_indices) == 1 << 16
    @test count(==(maximum(landscape.fitness)), landscape.fitness) == 3640

    nozero = IT3708Project3.triangle_landscape(; include_zero = false)
    @test length(nozero.subset_indices) == (1 << 16) - 1
end
