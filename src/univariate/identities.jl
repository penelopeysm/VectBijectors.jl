# This file provides bijectors for unconstrained continuous univariate distributions, as
# well as discrete univariate distributions.

"""
    Only

Callable struct, defined such that `(::Only)(x) = x[]`.

!!! warning
    This does not check whether the input has exactly one element.
"""
struct Only end
(::Only)(x) = x[]
with_logabsdet_jacobian(::Only, x::AbstractVector{T}) where {T<:Number} = (x[], zero(T))
with_logabsdet_jacobian(::Only, x::AbstractVector) = (x[], zero(Float64))
inverse(::Only) = Vect()

"""
   Vect

Callable struct, defined such that `(::Vect)(x) = [x]`.

!!! warning
    This does not check whether the input is a scalar.
"""
struct Vect end
(::Vect)(x) = [x]
with_logabsdet_jacobian(::Vect, x::Number) = ([x], zero(x))
with_logabsdet_jacobian(::Vect, x) = ([x], zero(Float64))
inverse(::Vect) = Only()

# For all univariate distributions, from_vec and to_vec are simple
VectBijectors.from_vec(::D.UnivariateDistribution) = Only()
VectBijectors.to_vec(::D.UnivariateDistribution) = Vect()

# For discrete univariate distributions, we really can't transform the 'support'
VectBijectors.from_linked_vec(::D.DiscreteUnivariateDistribution) = Only()
VectBijectors.to_linked_vec(::D.DiscreteUnivariateDistribution) = Vect()

# These continuous distributions have support over the entire real line.
for dist_type in [
    D.Cauchy,
    D.Chernoff,
    D.Gumbel,
    D.JohnsonSU,
    D.Laplace,
    D.Logistic,
    D.NoncentralT,
    D.Normal,
    D.NormalCanon,
    D.NormalInverseGaussian,
    D.PGeneralizedGaussian,
    D.SkewedExponentialPower,
    D.SkewNormal,
    D.TDist
]
    @eval begin
        VectBijectors.from_linked_vec(::$dist_type) = Only()
        VectBijectors.to_linked_vec(::$dist_type) = Vect()
    end
end
