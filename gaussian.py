import numpy as np
from scipy.linalg import block_diag
from category import StrictMarkov
import csv

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

    def recovery_from1(self, nx):
        assert (self.source == 0)
        print("Gaussian Conditional")

        nxy = self.target
        ny = nxy - nx

        M = self.covariance

        px = self.proj1(nx)
        x = px.mean
        P = px.covariance
        Pinv = np.linalg.inv(P)

        py = self.proj2(ny)
        y = py.mean
        R = py.covariance
        Q = M[nx:,:nx]

        F = Q @ Pinv
        z = y - Q @ Pinv @ x
        S = R - Q @ Pinv @ Q.T
        

        return Gaussian(matrix=F, mean=z, covariance=S)

def classic_update(prior, dynamics, instrument, measurement):
    xhat = prior.mean
    P = prior.covariance
    F = dynamics.matrix
    Q = dynamics.covariance
    H = instrument.matrix
    R = instrument.covariance
    z = measurement.mean
    xbar = F@xhat
    Pbar = F@P@F.T + Q

    ybar = H@xbar
    ytilde = z - ybar
    S = H@Pbar@H.T + R
    K = Pbar@H.T@np.linalg.inv(S)

    xhat_new = xbar + K@ytilde
    P_new = Pbar - K@H@Pbar

    return Gaussian.uncertain_state(xhat_new, P_new)

if __name__ == "__main__":
    # The following code block sets up a simple continuous dynamics model for simulation
    A = np.array([[0,1],[-1,-1]]) # Continuous dynamics matrix
    dynamics_model = lambda t, x: A@x
    t0 = 0
    tf = 10
    t_span = (t0, tf)
    x0 = np.array([3,2])
    dt = 0.5
    t_eval = np.arange(t0, tf, dt)

    # Simulate the system and get a sequence of states
    from scipy.integrate import solve_ivp
    solution = solve_ivp(dynamics_model, t_span, x0, t_eval=t_eval)
    y = solution.y
    xs = [y[:,k] for k in range(y.shape[1])]

    # Simulate a sequence of measurements
    from numpy.random import normal
    H = np.array([[1,0]])
    std = 0.5
    ys = [H@x + normal(scale=std) for x in xs]
    ys = ys[1:]
    measurements = [Gaussian.certain_state(y) for y in ys]

    
    # Model to be used in the filter
    from scipy.linalg import expm
    F = expm(A*dt)
    P0 = np.identity(2)
    initial_state = Gaussian.uncertain_state(x0, P0)
    dynamics = Gaussian.linear(F)
    R = np.array([[std**2]])
    w = np.zeros(1)
    instrument = Gaussian(matrix=H, mean=w, covariance=R)

    posteriors = [initial_state]
    priors = []
    predicted_measurements = []
    states = xs

    for meas in measurements:
        cat_est = posteriors[-1]

        next_prior = dynamics @ cat_est 
        priors += [next_prior]

        pred_meas = instrument @ next_prior
        predicted_measurements += [pred_meas]

        posteriors += [cat_est.update(dynamics, instrument, meas)]

    def get_mean_2d(state):
        mean = state.mean
        mx = mean[0]
        my = mean[1]
        return [mx, my]

    def get_cov_2d(state):
        cov = state.covariance
        Pxx = cov[0,0]
        Pxy = cov[0,1]
        Pyy = cov[1,1]
        return [Pxx, Pxy, Pyy]

    def get_mean_1d(state):
        return [state.mean[0]]
    def get_cov_1d(state):
        return [state.covariance[0,0]]

    with open('posterior_means.csv', 'w') as m, open('posterior_covariances.csv', 'w') as p:
        mhead = ['mx', 'mv']
        writerm = csv.writer(m)
        writerm.writerow(mhead)

        phead = ['Pxx', 'Pxv', 'Pvv']
        writerp = csv.writer(p)
        writerp.writerow(phead)

        for state in posteriors:
            writerm.writerow(get_mean_2d(state))
            writerp.writerow(get_cov_2d(state))

    with open('prior_means.csv', 'w') as m, open('prior_covariances.csv', 'w') as p:
        mhead = ['mx', 'my']
        writerm = csv.writer(m)
        writerm.writerow(mhead)

        phead = ['Pxx', 'Pxv', 'Pvv']
        writerp = csv.writer(p)
        writerp.writerow(phead)

        for state in priors:
            writerm.writerow(get_mean_2d(state))
            writerp.writerow(get_cov_2d(state))


    with open('predicted_measurement_means.csv', 'w') as m, open('predicted_measurement_covariances.csv', 'w') as p:
        mhead = ['mean']
        writerm = csv.writer(m)
        writerm.writerow(mhead)

        phead = ['covariance']
        writerp = csv.writer(p)
        writerp.writerow(phead)

        for state in predicted_measurements:
            writerm.writerow(get_mean_1d(state))
            writerp.writerow(get_cov_1d(state))

    with open('actual_measurements.csv', 'w') as m:
        mhead = ['measured_position']
        writerm = csv.writer(m)
        writerm.writerow(mhead)

        for meas in measurements:
            writerm.writerow(get_mean_1d(meas))

    with open('actual_states.csv', 'w') as m:
        mhead = ['position', 'velocity']
        writerm = csv.writer(m)
        writerm.writerow(mhead)

        for state in states:
            writerm.writerow(list(state))
