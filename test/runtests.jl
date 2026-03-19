using IT3708Project3
using Test

@testset "IT3708Project3.jl" begin
    @test sprint(IT3708Project3.greet) == "Hello World!"
end
