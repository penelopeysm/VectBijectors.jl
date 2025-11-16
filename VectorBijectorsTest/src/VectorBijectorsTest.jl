module VectorBijectorsTest

using ADTypes
using Test
using VectorBijectors
using Distributions
using LinearAlgebra: logabsdet
import DifferentiationInterface as DI
import ForwardDiff

# TODO: Would like to use FiniteDifferences, but very easy to run into issues with https://juliadiff.org/FiniteDifferences.jl/latest/#Dealing-with-Singularities
const ref_adtype = AutoForwardDiff()

_name(d::Distribution) = nameof(typeof(d))

function test_all(d::Distribution)
    @info "Testing $(_name(d))"
    @testset "$(_name(d))" begin
        VectorBijectorsTest.test_roundtrip(d)
        VectorBijectorsTest.test_roundtrip_inverse(d)
        VectorBijectorsTest.test_type_stability(d)
        VectorBijectorsTest.test_vec_lengths(d)
        VectorBijectorsTest.test_allocations(d)
        VectorBijectorsTest.test_logjac(d)
    end
end

function test_roundtrip(d::Distribution)
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

function test_roundtrip_inverse(d::Distribution)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip inverse: $(_name(d))" begin
        len = vec_length(d)
        for _ in 1:100
            y = randn(len)
            frvs = from_vec(d)
            ffwd = to_vec(d)
            @test y ≈ ffwd(frvs(y))
        end
    end
    @testset "roundtrip inverse (linked): $(_name(d))" begin
        len = linked_vec_length(d)
        for _ in 1:100
            y = randn(len)
            ffwd = to_linked_vec(d)
            frvs = from_linked_vec(d)
            @test y ≈ ffwd(frvs(y))
        end
    end
end

function test_type_stability(d::Distribution)
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

function test_vec_lengths(d::Distribution)
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

function test_allocations(d::Distribution)
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

function test_logjac(d::Distribution; atol=1e-13)
    # Vectorisation logjacs should be zero because they are just reshapes.
    @testset "logjac: $(_name(d))" begin
        for _ in 1:100
            x = rand(d)
            ffwd = to_vec(d)
            @test iszero(last(with_logabsdet_jacobian(ffwd, x)))
            y = ffwd(x)
            frvs = from_vec(d)
            @test iszero(last(with_logabsdet_jacobian(frvs, y)))
        end
    end

    # Link logjacs will not be zero, so we need to check against finite differences Because
    # Jacobians need to map from vector to vector, here we test the transformation of the
    # vectorised form to the linked vectorised form via the original sample.
    #
    # TODO: generalising this to the case where xvec and ffwd(xvec) have different
    # dimensions is tricky as we learnt with LKJChol!
    @testset "logjac (linked): $(_name(d))" begin
        for _ in 1:100
            xvec = to_vec(d)(rand(d))
            ffwd = to_linked_vec(d) ∘ from_vec(d)
            vbt_logjac = last(with_logabsdet_jacobian(ffwd, xvec))
            ad_logjac = first(logabsdet(DI.jacobian(ffwd, ref_adtype, xvec)))
            @test vbt_logjac ≈ ad_logjac atol=atol

            yvec = to_linked_vec(d)(rand(d))
            frvs = to_vec(d) ∘ from_linked_vec(d)
            vbt_logjac = last(with_logabsdet_jacobian(frvs, yvec))
            ad_logjac = first(logabsdet(DI.jacobian(frvs, ref_adtype, yvec)))
            @test vbt_logjac ≈ ad_logjac atol=atol
        end
    end
end

end # module VectorBijectorsTest
