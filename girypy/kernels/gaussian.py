import numpy as np
from scipy.linalg import block_diag
from girypy.strictmarkov import StrictMarkov

class Gaussian(StrictMarkov):
    def __init__(self, matrix, **kwargs):
        (target, source) = matrix.shape

        self.matrix = matrix
        allowed_keys = {"mean", "covariance"}
        self.mean = np.zeros(target)
        self.covariance = np.zeros((target, target))
        self.__dict__.update((k,v) for (k,v) in kwargs.items() if k in allowed_keys)

        assert (self.mean.shape == (target,))
        assert (self.covariance.shape == (target,target))
        eigs, evecs = np.linalg.eig(self.covariance)
        #assert all([eig >= 0 for eig in eigs])

    def __str__(self):
        F = str(self.matrix)
        x = str(self.mean.reshape(self.target,1))
        P = str(self.covariance)

        num_blanks = self.target - 1
        beginning = ['N( '] + ['   ']*num_blanks
        middle = ['    ']*num_blanks + [' , ']
        ending = ['']*num_blanks + [' )']

        splt_lines = zip(beginning,\
                x.split('\n'),\
                middle,\
                P.split('\n'),\
                ending)

        return '\n'.join(''.join(z) for z in splt_lines)

    def terminal_matrix(n): return np.empty((0,n))
    def elemental_matrix(n): return np.empty((n,0))

    def unit(): return 0
    # Creator Methods
    @classmethod
    def certain_state(cls, state):
        matrix = Gaussian.elemental_matrix(len(state))
        return cls(matrix=matrix, mean=state)

    @classmethod
    def uncertain_state(cls, mean, covariance):
        matrix = Gaussian.elemental_matrix(len(mean))
        return cls(matrix=matrix, mean=mean, covariance=covariance)

    @classmethod
    def affine(cls, matrix, vector):
        return cls(matrix=matrix, vector=vector)

    @classmethod
    def linear(cls, matrix): return cls(matrix=matrix)

    # Category
    @classmethod
    def identity(cls, n): return cls(matrix=np.identity(n))

    def compose(self, other):
        assert (self.source == other.target)

        F = other.matrix
        G = self.matrix
        H = G@F

        x = other.mean
        y = self.mean
        z = G@x + y

        P = other.covariance
        Q = self.covariance
        R = G@P@G.T + Q

        return Gaussian(matrix=H, mean=z, covariance=R)

    @property
    def source(self): return self.matrix.shape[1]
    @property
    def target(self): return self.matrix.shape[0]

    # StrictMonoidal
    def bimap(self,other):
        matrix = block_diag(self.matrix, other.matrix)
        mean = np.concatenate((self.mean, other.mean))
        covariance = block_diag(self.covariance, other.covariance)
        return Gaussian(matrix=matrix, mean=mean, covariance=covariance)

    # Symmetric
    @classmethod
    def swapper(cls, n1, n2):
        z21 = np.zeros((n2,n1))
        z12 = np.zeros((n1,n2))
        i1 = np.identity(n1)
        i2 = np.identity(n2)
        return cls(matrix=np.block([[z21,i2],[i1,z12]]))

    @staticmethod
    def factor(xy, x): return xy - x

    # StrictMarkov
    @classmethod
    def copier(cls, n):
        I = np.identity(n)
        return cls(matrix=np.vstack((I,I)))
    @classmethod
    def discarder(cls, n): return cls(matrix=Gaussian.terminal_matrix(n))

    def condition(self, n):
        M = self.matrix[:n, :]
        N = self.matrix[n:, :]

        Cxx = self.covariance[:n, :n]
        Cxy = self.covariance[:n, n:]
        Cyx = self.covariance[n:, :n]
        Cyy = self.covariance[n:, n:]

        s = self.mean[:n]
        t = self.mean[n:]

        A = Cyx @ np.linalg.pinv(Cxx)

        return Gaussian(
                matrix=np.hstack((A, N-A@M)),
                covariance=(Cyy - A@Cxy),
                mean=(t - A@s))
