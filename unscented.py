import numpy as np
from category import *

# sP X := X * R * R
class SigmaPoint(Functor):
    def __init__(self, position, mean_weight, covariance_weight):
        self.position = position
        self.mean_weight = mean_weight
        self.covariance_weight = covariance_weight

    def __str__(self): return str(self.position)

    # sP : (X -> Y) -> (sP X -> sP Y)
    @staticmethod
    def lift(f):
        def spf(sp):
            x = sp.position
            return SigmaPoint(f(x), sp.mean_weight, sp.covariance_weight)
        return spf

# sR X := [sP X]
class SigmaRepresentation(Functor):
    def __init__(self, sigma_points):
        self.sigma_points = sigma_points

    def __str__(self):
        points = "\n".join([str(sp.position) for sp in self.sigma_points])
        return "My sigma points are:\n" + points

    # sR : (X -> Y) -> (sR X -> sR Y)
    @staticmethod
    def lift(f):
        def sRf(sigma_rep):
            sps = sigma_rep.sigma_points
            new_sps = list(map(SigmaPoint.lift(f), sps))
            return SigmaRepresentation(new_sps)
        return sRf

# unscent : N X -> sR X
def unscent(gaussian):
    x = gaussian.mean
    P = gaussian.covariance
    n = x.shape[0]

    a = 1
    b = 2
    k = 3-n
    l = a**2 * (n + k) - n

    w0m = l/(n + l)
    w0c = w0m + (1-a**2 + b)
    sigma_points = [SigmaPoint(x, w0m, w0c)]

    wi = 0.5/(n+l)
    L = np.linalg.cholesky(P)
    for i in range(n):
        Li = L[:,i]
        offset = np.sqrt(n+l)*Li
        sigma_points += [SigmaPoint(x+offset, wi, wi)]
        sigma_points += [SigmaPoint(x-offset, wi, wi)]
    return SigmaRepresentation(sigma_points)

# scent : sR X -> N X
def scent(sigma_rep):
    mean = sum([sp.mean_weight * sp.position for sp in sigma_rep.sigma_points])
    n = len(mean)
    covariance = np.zeros((n,n))

    def tensorsquare(x):
        return np.tensordot(x,x,axes=0)

    for sp in sigma_rep.sigma_points:
        ts = tensorsquare(sp.position - mean)
        covariance += sp.covariance_weight * ts

    return Gaussian(mean, covariance)

