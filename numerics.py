import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(funcName)s:%(message)s")




def solveODE(f, inits, interval, tol=1e-8, order=4, solver="ABE"):
    pass

class ODE:
    def __init__(self, f, init_val, interval):
        self.f = f
        self.y0 = init_val
        self.a = interval[0]
        self.b = interval[1]
    
    def func(self, t, y):
        if isinstance(t, (int, float)):
            if isinstance(y, (int, float)):
                return self.f(t, y)
            else:
                return np.array([self.f(t, yval) for yval in y])
        else:
            if isinstance(y, (int, float)):
                return np.array([self.f(tval, y) for tval in t])
            else:
                return np.array([[self.f(tval, yval) for yval in y] for tval in t])

    def solve(self, solver="ABE", tol=1e-8, steps=10000):
        pass

    def



def main():
    # do logging
    pass

if __name__ == "__main__":
    main()
