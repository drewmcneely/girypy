#from girypy.kernels import gaussian as gs
import csv
import numpy as np
import scipy

import sys
sys.path.append('../girypy')

from girypy.kernels.gaussian import Gaussian
from girypy.algorithms.filter import update

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

    posteriors += [update(cat_est, dynamics, instrument, meas)]

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
