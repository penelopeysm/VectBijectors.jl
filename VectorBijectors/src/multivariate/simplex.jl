# This handles distributions over probability vectors / unit simplices, i.e., distributions
# `d` for which `x = rand(d)` is a vector where all(x -> x >= 0, x) && sum(x) == 1.

# The following is a copy of what Stan does, see e.g.
# https://mc-stan.org/docs/reference-manual/transforms.html#simplex-transform.section
# https://mc-stan.org/docs/reference-manual/transforms.html#sum-to-zero-vector-transform
#
# Note that this differs from Bijectors.jl, which uses a stick-breaking transform (which
# is also what Stan used prior to v2.37).
# 
# There are also claims of a paper here comparing different simplex transforms:
# https://discourse.mc-stan.org/t/where-to-find-stick-breaking-transform-and-jacobian-code/35116/4
# However, I can't find any paper / preprint. This looks like the relevant code:
# https://github.com/mjhajharia/transforms, but I'm not sure if Jacobian code is there.

# The code below is a VERY naive implementation of what's given in the Stan docs.
#
# The link transform is (sum_to_zero_transform ∘ inverse_softmax)
# The inverse link transform is (softmax ∘ sum_to_zero_inverse_transform)
#
# TODO: Optimise the implementation.
# TODO: Find out if this is actually better than Bijectors' stick-breaking. I can't
# understand why it should be? It looks super inefficient.
# TODO: If so, implement logjac.

struct SimplexInvlink end
function (s::SimplexInvlink)(y::AbstractVector{T}) where {T<:Real}
    first(with_logabsdet_jacobian(s, y))
end
function with_logabsdet_jacobian(::SimplexInvlink, y::AbstractVector{T}) where {T<:Real}
    # This is a line-by-line translation of Stan code:
    # https://github.com/stan-dev/math/blob/b82d68ced2e73c8188f3bbf287c1321033103986/stan/math/prim/constraint/simplex_constrain.hpp#L73-L90
    # The only adjustment needed for Julia is to use 1-based indexing.
    N = length(y)
    N == 0 && return ones(T, 1)
    z = zeros(T, N + 1)
    sum_w = zero(T)
    d = zero(T)
    max_val = zero(T)
    max_val_old = -Inf
    for i = N:-1:1
        w = y[i] / sqrt(i * (i + 1))
        sum_w += w
        z[i] += sum_w
        z[i+1] -= w * i
        max_val = max(max_val_old, z[i+1])
        d = d * exp(max_val_old - max_val) + exp(z[i+1] - max_val)
        max_val_old = max_val
    end
    max_val = max(max_val_old, z[1])
    d = d * exp(max_val_old - max_val) + exp(z[1] - max_val)
    z = exp.(z .- max_val) ./ d
    logjac = -((N + 1) * (max_val + log(d))) + (0.5 * log(N + 1))
    return z, logjac
    # VectorBijectors (this function):
    #
    # julia> @be vb_inv($vb_y)
    # Benchmark: 2950 samples with 245 evaluations
    #  min    104.592 ns (5 allocs: 224 bytes)
    #  median 106.633 ns (5 allocs: 224 bytes)
    #  mean   128.849 ns (5 allocs: 224 bytes, 0.23% gc time)
    #  max    16.186 μs (5 allocs: 224 bytes, 98.42% gc time)
    #
    # Bijectors (b_inv = inverse(bijector(dist))):
    #
    # julia> @be $b_inv($y)
    # Benchmark: 2809 samples with 368 evaluations
    #  min    75.519 ns (2 allocs: 96 bytes)
    #  median 84.353 ns (2 allocs: 96 bytes)
    #  mean   91.699 ns (2 allocs: 96 bytes, 0.17% gc time)
    #  max    9.486 μs (2 allocs: 96 bytes, 98.36% gc time)
    #
    # So not THAT much difference, but Bijectors is still faster.
end
inverse(::SimplexInvlink) = SimplexLink()

struct SimplexLink end
function (::SimplexLink)(x::AbstractVector{T}) where {T<:Real}
    # TODO: optimise
    return (_sum_to_zero_transform ∘ _inverse_softmax)(x)
end

function _simplex_invlink_slow(y::AbstractVector{T}) where {T<:Real}
    return _softmax(_sum_to_zero_inverse_transform(y))
end

# converts an unconstrained vector of length N to a sum-to-zero vector of length N+1
function _sum_to_zero_inverse_transform(y::AbstractVector{T}) where {T<:Real}
    N = length(y)
    x = similar(y, N + 1)
    x[1] = sum(i -> y[i] / sqrt(i * (i + 1)), 1:N)
    for n = 1:N
        x[n+1] =
            sum(i -> y[i] / sqrt(i * (i + 1)), (n+1):N; init=zero(T)) -
            (n * y[n] / sqrt(n * (n + 1)))
    end
    return x
end

# converts a sum-to-zero vector of length N+1 to an unconstrained vector of length N
# https://mc-stan.org/docs/reference-manual/transforms.html#sum-to-zero-vector-transform
function _sum_to_zero_transform(x::AbstractVector{T}) where {T<:Real}
    N = length(x) - 1
    y = similar(x, N)
    S_N = zero(T)
    y[N] = -x[N+1] * sqrt(1 + (1 / N))
    for n = N-1:-1:1
        w_np1 = y[n+1] / sqrt((n + 1) * (n + 2))
        S_N = S_N + w_np1
        y[n] = (S_N - x[n+1]) * sqrt(n * (n + 1)) / n
    end
    return y
end

# converts a sum-to-zero vector into a unit simplex
function _softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    ex = map(xi -> exp(xi - xmax), x)  # for numerical stability
    return ex / sum(ex)
end

# converts a unit simplex into a sum-to-zero vector
function _inverse_softmax(y::AbstractVector{T}) where {T<:Real}
    K = length(y)
    z = similar(y, K)
    for k = 1:K
        z[k] = log(y[k]) - (sum(log.(y)) / K)
    end
    return z
end

to_linked_vec(::D.Dirichlet) = (_sum_to_zero_transform ∘ _inverse_softmax)
from_linked_vec(::D.Dirichlet) = SimplexInvlink()
linked_vec_length(d::D.Dirichlet) = length(d) - 1
