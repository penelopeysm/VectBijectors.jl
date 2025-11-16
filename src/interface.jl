"""
    VectBijectors.from_vec(d::Distribution)

Returns a function that can be used to convert a vectorised sample from `d` back to its original form.
"""
function from_vec end

"""
    VectBijectors.to_vec(d::Distribution)

Returns a function that can be used to vectorise a sample from `d`.
"""
function to_vec end

"""
    VectBijectors.from_linked_vec(d::LinkedDistribution)

Returns a function that can be used to convert an unconstrained vector back to a sample from `d`.
"""
function from_linked_vec end

"""
    VectBijectors.to_linked_vec(d::LinkedDistribution)

Returns a function that can be used to convert a sample from `d` to an unconstrained vector.
"""
function to_linked_vec end
