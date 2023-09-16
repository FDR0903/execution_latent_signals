import numpy as np
from scipy.integrate import solve_ivp

def solveLotkaVolterra(phi, beta, c, T, k, varphi, nb_t=1000):
    ts         = np.linspace(0, T, nb_t)
    
    Gt         = lambda t, s: -np.array([-beta*c - phi + s[2]      + s[0] * (s[0] + c*s[1]) / 2 / k ,
                                         -beta - beta * s[1]       + s[1] * (s[0] + c*s[1]) / 2 / k ,
                                         beta*beta*c - beta * s[2] + s[2] * (s[0] + c*s[1]) / 2 / k])

    sol        = solve_ivp(Gt, 
                           [T, 0], 
                           np.array([c - varphi, 0, 0]),  #c * beta
                           t_eval = ts[::-1])
    
    Gt         = sol.y
    
    g2, g3, g4 = Gt[0, ::-1], Gt[1, ::-1], Gt[2, ::-1]

    return Gt, ts, g2, g3, g4


def solveLotkaVolterra_matrix_form(_phi, _beta, _c, _T, _k, _nb_t=10000):
    _ts         = np.linspace(0, _T, _nb_t)
    _a          = np.array([_phi, -_beta, -_beta*_beta*_c])
    
    _b          = np.array([0, -_beta, _beta])
    
    _A          = np.array([[-1, _c, 1/_beta],
                           [-1, _c, 1/_beta],
                           [-1, _c, 1/_beta]])/2/_k
    
    _Gt         = lambda t, s: -_a + -np.diag(s) @ (_b + _A@s)    
    
    _sol        = solve_ivp(_Gt, 
                           [_T, 0], 
                           np.array([-_c, 0, 0]), 
                           t_eval = _ts[::-1])
    _Gt         = _sol.y
    
    return _Gt, _ts

