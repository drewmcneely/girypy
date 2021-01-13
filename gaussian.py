import numpy as np
from scipy.linalg import block_diag

class Gaussian:
    def __init__(self, matrix, **kwargs):
        (target, source) = matrix.shape
        self.__dict__.update(zip(["target","source"] , matrix.shape))

        self.matrix = matrix
        allowed_keys = {"mean", "covariance"}
        self.mean = np.zeros(target)
        self.covariance = np.zeros((target, target))
        self.__dict__.update((k,v) for (k,v) in kwargs.items() if k in allowed_keys)

        assert (self.mean.shape == (target,))
        assert (covariance.shape == (target,target))
        eigs, evecs = np.linalg.eig(self.covariance)
        assert all([eig >= 0 for eig in eigs])

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

    # Creator Methods
    @classmethod
    def certain_state(cls, state):
        matrix = elemental_matrix(len(state))
        return cls(matrix=matrix, mean=state)

    @classmethod
    def uncertain_state(cls, mean, covariance):
        matrix = elemental_matrix(len(mean))
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
        '''
        (G,y,Q) @ (F,x,P)
        = (Gm + N(y,Q) <- m) @ (Fn + N(x,P) <- n)
        = G(Fn + N(x,P)) + N(y,Q)
        = GFn + G*N(x,P) + N(y,Q)
        = GFn + N(Gx, GPG^) + N(y,Q)
        = GFn + N(Gx + y, GPG^ + Q)
        '''

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

    def __matmul__(self,other): return compose(self,other)

    # Monoidal
    def bimap(self,other):
        matrix = block_diag(self.matrix, other.matrix)
        mean = np.concatenate(self.mean, other.mean)
        covariance = block_diag(self.covariance, other.covariance)
        return Gaussian(matrix=matrix, mean=mean, covariance=covariance)

    def __mul__(self,other): return bimap(self,other)

    # Symmetric
    @classmethod
    def swap(cls, n1, n2):
        z21 = np.zeros((n2,n1))
        z12 = np.zeros((n1,n2))
        i1 = np.identity(n1)
        i2 = np.identity(n2)
        # def block_split(P, n):
        #     P11 = P[:n,:n]
        #     P12 = P[:n,n:]
        #     P21 = P[n:,:n]
        #     P22 = P[n:,n:]
        #     return (P11, P12, P21, P22)
        # def block_swap(P,n):
        #     P11, P12, P21, P22 = block_split(P,n)
        #     return np.block([[P22,P21],[P12,P11]])
        # n = n1 - 1
        # x = self.mean(:n)
        # y = self.mean(n:)
        # return Gaussian(matrix=block_swap(self.matrix),
        #         mean=np.concatenate((y,x)),
        #         covariance=block_swap(self.covariance))
        return cls(matrix=np.block([[z21,i2],[i1,z12]])

    # Comonoidal
    @classmethod
    def copier(cls, n):
        I = np.identity(n)
        return cls(matrix=np.vstack((I,I)))
    @classmethod
    def discarder(cls, n): return cls(matrix=terminal_matrix(n))

    # Semi-Cartesian
    @classmethod
    def proj1(cls, n1, n2): return cls.identity(n1) * cls.discarder(n2)
    @classmethod
    def proj2(cls, n1, n2): return cls.proj1(n2,n1) @ cls.swap(n1,n2)

    # Markov
    def integrator1(self):
        nx = self.source
        ny = self.target
        return (self * Gaussian.identity(nx) ) @ Gaussian.copier(nx)
    def integrator2(self):
        nx = self.source
        ny = self.target
