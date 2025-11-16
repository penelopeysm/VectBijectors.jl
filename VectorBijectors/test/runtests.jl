using Test

# VBT isn't released yet, so we need to manually develop
import Pkg
Pkg.develop(; path = joinpath(@__DIR__, "..", "..", "VectorBijectorsTest"))

@testset verbose = true "VectorBijectors.jl" begin
    include("univariate.jl")
end
