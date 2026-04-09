using IT3708Project3
using Test

@testset "Real landscape parsing" begin
    dataset = IT3708Project3.DATASETS["breast-w"]
    data = IT3708Project3.parse_dataset(dataset.path, dataset.num_features)

    @test length(data.indices) == 511
    @test data.mean_accuracy[1] ≈ 0.7878048419952393 atol = 1e-9
    @test data.mean_accuracy[end] ≈ 0.9672255516052246 atol = 1e-9
    @test data.mean_time[1] ≈ 0.08813527971506119 atol = 1e-9
    @test data.num_features[1] == 1
    @test data.num_features[3] == 2

    output_csv = tempname() * ".csv"
    IT3708Project3.write_csv(data, output_csv)
    lines = readlines(output_csv)
    @test lines[1] == "index,num_features,mean_accuracy,mean_time"
    @test startswith(lines[2], "1,1,")
    @test length(lines) == 512
end

@testset "CSV landscape" begin
    dataset = IT3708Project3.DATASETS["breast-w"]
    parsed = IT3708Project3.parse_dataset(dataset.path, dataset.num_features)
    output_csv = tempname() * ".csv"
    IT3708Project3.write_csv(parsed, output_csv)

    landscape = IT3708Project3.load_landscape(output_csv)
    penalized = IT3708Project3.apply_penalty(landscape, 0.1)

    @test length(landscape.indices) == 511
    @test landscape.indices[1] == 1
    @test landscape.num_features[1] == 1
    @test landscape.mean_accuracy[1] ≈ parsed.mean_accuracy[1] atol = 1e-12
    @test landscape.mean_time[1] ≈ parsed.mean_time[1] atol = 1e-12
    @test IT3708Project3.penalty(3, 0.1) ≈ 0.3 atol = 1e-12
    @test IT3708Project3.penalized_fitness(0.8, 3, 0.1) ≈ 0.5 atol = 1e-12
    @test penalized.penalties[1] ≈ 0.1 atol = 1e-12
    @test penalized.fitness[1] ≈ landscape.mean_accuracy[1] - 0.1 atol = 1e-12
end
