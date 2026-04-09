using IT3708Project3
using Test

@testset "Feature decoding" begin
    @test IT3708Project3.subset_to_bitvector(13, 4) == Bool[1, 0, 1, 1]
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
