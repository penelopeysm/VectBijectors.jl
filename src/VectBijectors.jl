module VectBijectors

using Distributions: Distributions
const D = Distributions
import ChangesOfVariables: with_logabsdet_jacobian
import InverseFunctions: inverse

include("interface.jl")
include("univariate/identities.jl")
include("univariate/positive.jl")
include("univariate/truncated.jl")

# Overloaded in TestExt
module TestUtils
function test_roundtrip end
function test_type_stability end
end

export from_vec
export to_vec
export from_linked_vec
export to_linked_vec
# re-exports
export with_logabsdet_jacobian
export inverse

end # module VectBijectors
