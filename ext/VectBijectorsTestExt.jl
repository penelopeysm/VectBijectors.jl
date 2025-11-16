module VectBijectorsTestExt

using Test
using VectBijectors
using Distributions

function VectBijectors.TestUtils.test_roundtrip(d::Distribution)
    @testset "roundtrip: $(typeof(d))" begin
        for _ in 1:1000
            x = rand(d)
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @test x ≈ frvs(ffwd(x))
        end
    end
    @testset "roundtrip (linked): $(typeof(d))" begin
        for _ in 1:1000
            x = rand(d)
            ffwd = to_linked_vec(d)
            frvs = from_linked_vec(d)
            @test x ≈ frvs(ffwd(x))
        end
    end
end

function VectBijectors.TestUtils.test_type_stability(d::Distribution)
    x = rand(d)
    @testset "type stability: $(typeof(d))" begin
        @inferred to_vec(d)
        @inferred from_vec(d)
        ffwd = to_vec(d)
        frvs = from_vec(d)
        @inferred ffwd(x)
        y = ffwd(x)
        @inferred frvs(y)
    end
    @testset "type stability (linked): $(typeof(d))" begin
        @inferred to_linked_vec(d)
        @inferred from_linked_vec(d)
        ffwd = to_linked_vec(d)
        frvs = from_linked_vec(d)
        @inferred ffwd(x)
        y = ffwd(x)
        @inferred frvs(y)
    end
end

end # module VectBijectorsTestExt
