module VectBijectorsTestExt

using Test
using VectBijectors
using Distributions

function VectBijectors.TestUtils.test_roundtrip(d::Distribution)
    @testset "roundtrip: $(typeof(d))" begin
        x = rand(d)
        ffwd = to_linked_vec(d)
        frvs = from_linked_vec(d)
        @test x ≈ frvs(ffwd(x))
    end
end

function VectBijectors.TestUtils.test_type_stability(d::Distribution)
    @testset "type stability: $(typeof(d))" begin
        @inferred to_linked_vec(d)
        @inferred from_linked_vec(d)
        x = rand(d)
        ffwd = to_linked_vec(d)
        frvs = from_linked_vec(d)
        @inferred ffwd(x)
        y = ffwd(x)
        @inferred frvs(y)
    end
end

end # module VectBijectorsTestExt
