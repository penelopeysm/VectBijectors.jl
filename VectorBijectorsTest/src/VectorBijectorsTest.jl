module VectorBijectorsTest

using ADTypes
using Test
using VectorBijectors
using Distributions
using LinearAlgebra: logabsdet
import DifferentiationInterface as DI
import ForwardDiff

# Would like to use FiniteDifferences, but very easy to run into issues with
# https://juliadiff.org/FiniteDifferences.jl/latest/#Dealing-with-Singularities
const ref_adtype = AutoForwardDiff()

const default_adtypes = [
    AutoReverseDiff(),
    AutoReverseDiff(; compile=true),
    AutoMooncake(),
    AutoMooncakeForward(),
]

# Pretty-printing distributions. Otherwise things like MvNormal are super ugly.
_name(d::Distribution) = nameof(typeof(d))
_name(d::Distributions.Censored) = "censored $(_name(d.uncensored)) [$(d.lower),$(d.upper)]"
_name(d::Distributions.Truncated) = "truncated $(_name(d.untruncated)) [$(d.lower),$(d.upper)]"

# AD will give nonsense results at the limits of censored distributions (since the gradient
# is not well-defined), so we avoid generating samples that are exactly at the limits.
_rand_safe_ad(d::Distribution) = rand(d)
_rand_safe_ad(d::Distributions.Censored) = begin
    a, b = d.lower, d.upper
    while true
        x = rand(d)
        if x != a && x != b
            return x
        end
    end
end

function test_all(d::Distribution; adtypes=default_adtypes, ad_atol=1e-12, test_allocs=true)
    @info "Testing $(_name(d))"
    @testset "$(_name(d))" begin
        VectorBijectorsTest.test_roundtrip(d)
        VectorBijectorsTest.test_roundtrip_inverse(d)
        VectorBijectorsTest.test_type_stability(d)
        VectorBijectorsTest.test_vec_lengths(d)
        if test_allocs
            VectorBijectorsTest.test_allocations(d)
        end
        VectorBijectorsTest.test_logjac(d; atol=ad_atol)
        VectorBijectorsTest.test_ad(d, adtypes; atol=ad_atol)
    end
end

function test_roundtrip(d::Distribution)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip: $(_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                frvs = from_vec(d)
                @test x ≈ frvs(ffwd(x))
            end
        end
    end
    @testset "roundtrip (linked): $(_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_linked_vec(d)
                frvs = from_linked_vec(d)
                @test x ≈ frvs(ffwd(x))
            end
        end
    end
end

function test_roundtrip_inverse(d::Distribution)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip inverse: $(_name(d))" begin
        len = vec_length(d)
        for _ in 1:100
            @testset let y = randn(len), d = d
                frvs = from_vec(d)
                ffwd = to_vec(d)
                @test y ≈ ffwd(frvs(y))
            end
        end
    end
    @testset "roundtrip inverse (linked): $(_name(d))" begin
        len = linked_vec_length(d)
        for _ in 1:100
            @testset let y = randn(len), d = d
                ffwd = to_linked_vec(d)
                frvs = from_linked_vec(d)
                @test y ≈ ffwd(frvs(y))
            end
        end
    end
end

function test_type_stability(d::Distribution)
    x = rand(d)
    @testset "type stability: $(_name(d))" begin
        @testset let x = x, d = d
            @inferred to_vec(d)
            @inferred from_vec(d)
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
    @testset "type stability (linked): $(_name(d))" begin
        @testset let x = x, d = d
            @inferred to_linked_vec(d)
            @inferred from_linked_vec(d)
            ffwd = to_linked_vec(d)
            frvs = from_linked_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
end

function test_vec_lengths(d::Distribution)
    @testset "vector lengths: $(_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_vec(d)(x)
                @test length(y) == vec_length(d)
            end
        end
    end
    @testset "vector lengths (linked): $(_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_linked_vec(d)(x)
                @test length(y) == linked_vec_length(d)
            end
        end
    end
end

function test_allocations(d::Distribution)
    # For univariates, to_vec and to_linked_vec always cause allocations because they have
    # to create a new vector.
    # TODO: Generalise to multivariates etc
    x = rand(d)
    @testset "allocations: $(_name(d))" begin
        @testset let x = x, d = d
            yvec = to_vec(d)(x)
            frvs = from_vec(d)
            frvs(yvec)
            @test (@allocations frvs(yvec)) == 0
        end
    end
    @testset "allocations (linked): $(_name(d))" begin
        @testset let x = x, d = d
            yvec = to_linked_vec(d)(x)
            frvs = from_linked_vec(d)
            frvs(yvec)
            @test (@allocations frvs(yvec)) == 0
        end
    end
end

function test_logjac(d::Distribution; atol=1e-12)
    # Vectorisation logjacs should be zero because they are just reshapes.
    @testset "logjac: $(_name(d))" begin
        for _ in 1:100
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                @test iszero(last(with_logabsdet_jacobian(ffwd, x)))
                y = ffwd(x)
                frvs = from_vec(d)
                @test iszero(last(with_logabsdet_jacobian(frvs, y)))
            end
        end
    end

    # Link logjacs will not be zero, so we need to check against a chosen backend. Because
    # Jacobians need to map from vector to vector, here we test the transformation of the
    # vectorised form to the linked vectorised form via the original sample.
    #
    # TODO: generalising this to the case where xvec and ffwd(xvec) have different
    # dimensions is tricky as we learnt with LKJChol!
    @testset "logjac (linked): $(_name(d))" begin
        for _ in 1:100
            x = _rand_safe_ad(d)
            @testset let x = x, d = d
                xvec = to_vec(d)(x)
                ffwd = to_linked_vec(d) ∘ from_vec(d)
                vbt_logjac = last(with_logabsdet_jacobian(ffwd, xvec))
                ad_logjac = first(logabsdet(DI.jacobian(ffwd, ref_adtype, xvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol
            end

            @testset let x = x, d = d
                yvec = to_linked_vec(d)(x)
                frvs = to_vec(d) ∘ from_linked_vec(d)
                vbt_logjac = last(with_logabsdet_jacobian(frvs, yvec))
                ad_logjac = first(logabsdet(DI.jacobian(frvs, ref_adtype, yvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol
            end
        end
    end
end

function test_ad(d::Distribution, adtypes::Vector{<:AbstractADType}; atol=1e-12)
    # Test that AD backends can differentiate the conversions to and from vector
    # and linked vector forms.
    @testset "AD forward: $(_name(d))" begin
        x = _rand_safe_ad(d)
        xvec = to_vec(d)(x)
        ffwd = to_linked_vec(d) ∘ from_vec(d)
        ref_jac = DI.jacobian(ffwd, ref_adtype, xvec)
        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(ffwd, adtype, xvec)
                @test ref_jac ≈ ad_jac atol = atol
            end
        end
    end
    @testset "AD reverse: $(_name(d))" begin
        x = _rand_safe_ad(d)
        yvec = to_linked_vec(d)(x)
        frvs = to_vec(d) ∘ from_linked_vec(d)
        ref_jac = DI.jacobian(frvs, ref_adtype, yvec)
        for adtype in adtypes
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(frvs, adtype, yvec)
                @test ref_jac ≈ ad_jac atol = atol
            end
        end
    end
end

end # module VectorBijectorsTest
