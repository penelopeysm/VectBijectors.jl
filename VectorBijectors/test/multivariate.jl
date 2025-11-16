module VBMultivariateTests

using Distributions
using LinearAlgebra
using Test
using VectorBijectorsTest

dists = [
    Multinomial(10, [0.2, 0.5, 0.3]),
    MvNormal([0.0, 0.0], I),
    # TODO: test more kinds of mvnormal. See
    # https://github.com/JuliaStats/Distributions.jl/blob/master/test/multivariate/mvnormal.jl
    # for some inspiration.
    # TODO: MvNormalCanon
    # TODO: MvLogitNormal (returns a probability vector, so can use the same transform as Dirichlet)
    # TODO: MvLogNormal
    # TODO: Dirichlet
]

@testset "Multivariates" begin
    for d in dists
        VectorBijectorsTest.test_all(d)
    end
end

end # module VBUnivariateTests
