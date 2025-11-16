module VectBijectorsTestExt

using Test
using VectBijectors
using Distributions

_name(d::Distribution) = nameof(typeof(d))

function VectBijectors.TestUtils.test_all(d::Distribution)
    @testset "$(_name(d))" begin
        VectBijectors.TestUtils.test_roundtrip(d)
        VectBijectors.TestUtils.test_type_stability(d)
        VectBijectors.TestUtils.test_vec_lengths(d)
        VectBijectors.TestUtils.test_allocations(d)
    end
end

function VectBijectors.TestUtils.test_roundtrip(d::Distribution)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip: $(_name(d))" begin
        for _ in 1:1000
            x = rand(d)
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @test x ≈ frvs(ffwd(x))
        end
    end
    @testset "roundtrip (linked): $(_name(d))" begin
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
    @testset "type stability: $(_name(d))" begin
        @inferred to_vec(d)
        @inferred from_vec(d)
        ffwd = to_vec(d)
        frvs = from_vec(d)
        @inferred ffwd(x)
        y = ffwd(x)
        @inferred frvs(y)
    end
    @testset "type stability (linked): $(_name(d))" begin
        @inferred to_linked_vec(d)
        @inferred from_linked_vec(d)
        ffwd = to_linked_vec(d)
        frvs = from_linked_vec(d)
        @inferred ffwd(x)
        y = ffwd(x)
        @inferred frvs(y)
    end
end

function VectBijectors.TestUtils.test_vec_lengths(d::Distribution)
    @testset "vector lengths: $(_name(d))" begin
        for _ in 1:10
            y = to_vec(d)(rand(d))
            @test length(y) == vec_length(d)
        end
    end
    @testset "vector lengths (linked): $(_name(d))" begin
        for _ in 1:10
            y = to_linked_vec(d)(rand(d))
            @test length(y) == linked_vec_length(d)
        end
    end
end

function VectBijectors.TestUtils.test_allocations(d::Distribution)
    # For univariates, to_vec and to_linked_vec always cause allocations because they have
    # to create a new vector.
    # TODO: Generalise to multivariates etc
    x = rand(d)
    @testset "allocations: $(_name(d))" begin
        yvec = to_vec(d)(x)
        frvs = from_vec(d)
        frvs(yvec)
        @test (@allocations frvs(yvec)) == 0
    end
    @testset "allocations (linked): $(_name(d))" begin
        yvec = to_linked_vec(d)(x)
        frvs = from_linked_vec(d)
        frvs(yvec)
        @test (@allocations frvs(yvec)) == 0
    end
end

end # module VectBijectorsTestExt
