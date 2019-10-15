import numpy as np


def FrankeFunction(x,y):
    term = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term += 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term += 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term += -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term
