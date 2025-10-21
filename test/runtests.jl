using EncodedPieceMaker
using Test

@testset "EncodedPieceMaker.jl" begin
    @testset "hello function tests" begin
        @test hello() == "Hello, World!"
        @test hello("Julia") == "Hello, Julia!"
        @test hello("Test") == "Hello, Test!"
    end
end