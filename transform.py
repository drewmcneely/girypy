class LinearTransform:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, x): return self.matrix @ x

    def push(self, P):
        F = self.matrix
        return F@P@F.T


## represent_matrix : (X -> Y) -> M X Y
#def represent_matrix(f):
#    nx = 2
#    ones = np.ones(nx)
#    test = f(ones)
#    if isinstance(test, Gaussian):
#        mapped_basis = [f(ei).mean for ei in list(np.identity(nx))]
#    else:
#        mapped_basis = [f(ei) for ei in list(np.identity(nx))]
#    return np.vstack(mapped_basis).T
#



## unrepresent_matrix : M X Y -> (X -> Y)
#def unrepresent_matrix(F): return lambda x: F@x
#
#
## characterize_noise : (X -> N Y) -> Pair (M X Y) (S+ Y)
#def characterize_noise(f):
#    nx = 2
#    F = represent_matrix(lambda x: f(x).mean)
#    Q = f(np.zeros(nx)).covariance
#    return (F,Q)
#
## uncharacterize_noise : Pair (M X Y) (S+ Y) -> (X -> N Y)
#def uncharacterize_noise(F,Q): return lambda x: Gaussian(F@x, Q)
#
