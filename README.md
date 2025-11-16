# VectorBijectors.jl

A lightweight reimplementation of Bijectors.jl's functionality, but specifically focused on transformations to and from vectors.

In particular, it guarantees that for any distribution `d`, `to_linked_vec(d)` is a vector.
This is not always true in Bijectors.jl.

This is intended primarily for use with probabilistic programming, e.g. DynamicPPL.jl.
Note that Bijectors.jl is a more general package that contains far more functionality than this.
