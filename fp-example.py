import numpy as np
from transform import *
from gaussian import *

def main():
    xbar = np.array([2,3])
    P = np.array([[2,1],[1,4]])

    prev_N = Gaussian(xbar, P)
    f = LinearTransform(np.array([[1,-1],[1,1]]))
    print(type(f))

    Q = np.array([[3,0],[0,2]])
    phi = Gaussian.add_noise(f, Q)

    print(prev_N)
    print(prev_N.push_through(f))
    print(prev_N.bind(phi))

    #h = LinearTransform(np.array([1,0]))
    #R = 0.5

    #f0 = unrepresent_matrix(F)
    #f = uncharacterize_noise(F,Q)
    #h0 = unrepresent_matrix(H)
    #h = uncharacterize_noise(H,R)

    #propagate0 = Gaussian.lift(compose(h0,f0))
    #propagate = Monad.kleisli(f,h)


    #prev_sr = unscent(prev_N)
    #prior_sr = prev_sr.push_through(f)
    #prior_sr_lift = SigmaRepresentation.lift(f)(prev_sr)

    #print(scent(prior_sr))
    #print(scent(prior_sr_lift))

if __name__ == "__main__":
    main()
