dists = [
    Arcsine(0, 1),
    Beta(2, 2),
    # BetaPrime(1, 2), pos
    Biweight(1, 2),
    # Cauchy(-2, 1), # iden
    # Chernoff(), # iden
    # Chi(1), # pos
    # Chisq(3), # pos
    Cosine(0, 1),
    Epanechnikov(0, 1),
    # Erlang(7, 0.5), # pos
    # Exponential(0.5), # pos
    # FDist(10, 1), # pos
    Frechet(1, 1),
    # Gamma(7.5, 1), # pos
    GeneralizedExtremeValue(0, 1, 1),
    GeneralizedPareto(0, 1, 1),
    # Gumbel(0, 1), # iden
    # InverseGamma(3, 0.5), # pos
    # InverseGaussian(1, 1), # pos
    # JohnsonSU(0.0, 1.0, 0.0, 1.0), # iden
    # Kolmogorov(), # pos
    KSDist(5),
    KSOneSided(5),
    Kumaraswamy(2, 5),
    # Laplace(0, 4), # iden
    Levy(0, 1),
    # Lindley(1.5), # pos
    # Logistic(2, 1), # iden
    LogitNormal(0, 1),
    # LogNormal(0, 1), # pos
    LogUniform(1, 10),
    NoncentralBeta(2, 3, 1),
    # NoncentralChisq(2, 3), # pos
    # NoncentralF(2, 3, 1), # pos
    # NoncentralT(2, 3), # iden
    # Normal(0, 1), # iden
    # NormalCanon(0, 1), # iden
    # NormalInverseGaussian(0, 0.5, 0.2, 0.1), # iden
    Pareto(1, 1),
    PGeneralizedGaussian(0.2), # iden
    # Rayleigh(0.5), # pos
    # Rician(0.5, 1), # pos
    Semicircle(1),
    # SkewedExponentialPower(0, 1, 0.7, 0.7), # iden
    # SkewNormal(0, 1, -1), # iden
    # StudentizedRange(2, 2), # pos
    SymTriangularDist(0, 1),
    # TDist(5), # iden
    TriangularDist(0, 1.5, 0.5),
    Triweight(1, 1),
    Uniform(0, 1),
    VonMises(0.5),
    # Weibull(0.5, 1) # pos
]

@testset "$d" for d in dists
    x = rand(d)
    ffwd = to_linked_vec(d)
    frvs = from_linked_vec(d)
    @test x ≈ frvs(ffwd(x))
end


using Distributions, VectBijectors, Test
d = Beta(2, 2)
for _ in 1:10000
    x = rand(d)
    ffwd = to_linked_vec(d)
    frvs = from_linked_vec(d)
    @test x ≈ frvs(ffwd(x))
end
