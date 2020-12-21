from category import *

# N X := X * (S+ X)
class Gaussian(Monad):
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    @property
    def dimension(self): return self.mean.shape[0]

    def __str__(self):
        x = str(self.mean.reshape(self.dimension,1))
        P = str(self.covariance)

        num_blanks = self.dimension - 1
        beginning = ['N( '] + ['   ']*num_blanks
        middle = ['    ']*num_blanks + [' , ']
        ending = ['']*num_blanks + [' )']

        splt_lines = zip(beginning,\
                x.split('\n'),\
                middle,\
                P.split('\n'),\
                ending)

        return '\n'.join(''.join(z) for z in splt_lines)

    @staticmethod
    def lift(f):
        return lambda g: Gaussian(f(g.mean), f.push(g.covariance))

    def join(nnx):
        nx = nnx.mean
        Q = nnx.covariance

        x = nx.mean
        P = nx.covariance
        return Gaussian(x, P+Q)

    def unit(x):
        dim = len(x)
        return Gaussian(x,np.zeros((dim,dim)))

    @classmethod
    def add_noise(cls, f, Q): return lambda x: cls(f(x), Q)
