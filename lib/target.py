import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from .lotka import solveLotkaVolterra
from .utils import simulate_ou_signal_price


###############################################
# Solving the problem when the signal is OU
###############################################
def solve_g1_OU_closedform(q_target, g2, g3, k, c, phi, sigma_signal, r_signal, alpha, varphi, q_target_T, T, nb_t):
    dt = T / (nb_t - 1)
    ds = dt
    du = dt
    
    # Compute g1 with closed-form     
    ts        = np.linspace(0, T, nb_t)    
    expterm1  = np.array([sum((np.exp(np.sum(g2[it:iss] * du + c*g3[it:iss] * du)/2/k - r_signal*(ts[iss] - ts[it])) * ds for iss in range(it+1, nb_t)) ) for it in range(nb_t)])
    expterm2  = np.array([sum((q_target[iss] * np.exp(np.sum(g2[it:iss] * dt + c*g3[it:iss]*dt)/2/k  ) * dt  for iss in range(it+1, nb_t)) ) for it in range(nb_t) ])
    
    # this is because target trading speed is deterministic
    d_qtarget = np.diff(q_target)
    expterm4  = np.array([ np.exp(np.sum(g2[it:nb_t] * ds + c * g3[it:nb_t]*ds)/2/k)  for it in range(nb_t)])
    
    g1       = phi * expterm2 + alpha[:,0] * expterm1  + q_target_T * varphi * expterm4

    return g1

def solve_g1_OU_montecarlo(q_target, g2, g3, k, c, phi, r_signal, sigma_signal, varphi, q_target_T, N, T, nb_t):
    dt    = T / (nb_t - 1)
    g1_mc = np.zeros(nb_t)
    for n in range(N):
        if (n%10==0): print(n) 
        W  = np.random.randn(nb_t, 1)
        mc_alpha        = np.empty((nb_t, 1))
        mc_alpha[0]     = 0
        for i in range(1, nb_t):
            mc_alpha[i] = mc_alpha[i-1]  - r_signal * mc_alpha[i-1] * dt + sigma_signal * W[i] * np.sqrt(dt)
        
        g1_mc += np.array([sum(((mc_alpha[iss,0] + phi * q_target[iss]) * np.exp(np.sum(g2[it:iss] * dt + c*g3[it:iss]*dt)/2/k) * dt 
                                for iss in range(it, nb_t-1))) 
                           for it in range(nb_t)])

    # this is because target trading speed is deterministic
    d_qtarget = np.diff(q_target)
    #expterm3 = np.array([sum((np.exp(np.sum(g2[it:iss] * dt + c*g3[it:iss]*dt)/2/k) * d_qtarget[iss]  for iss in range(it, nb_t-1)) ) for it in range(nb_t) ])

    expterm4 = np.array([ np.exp(np.sum( g2[it:nb_t] * dt + c*g3[it:nb_t]*dt)/2/k)  for it in range(nb_t)])
    
    return g1_mc/N - varphi*q_target_T*expterm4

def get_strat_simulation(nb_t, r_signal, c, k, phi, sigma_signal, 
                       beta, sigma_transient, sigma_price, q_target, 
                       F0, Q0, T, varphi):
    
    # solve the ODE system
    Gt, ts, g2, g3, g4 = solveLotkaVolterra(phi  = phi, beta = beta, c = c, T = T, nb_t = nb_t, k = k, varphi = varphi)

    # generate signal & fundamental price
    alpha, F = simulate_ou_signal_price(r_signal = r_signal, sigma_signal = sigma_signal, 
                                        sigma_price = sigma_price, F0 = F0, T = T, nb_t = nb_t)

    # Solve g1
    g1       = solve_g1_OU_closedform(q_target, g2, g3, k, c, phi, sigma_signal, alpha, varphi, T, nb_t)
    
    # Solve the FBSDE system
    nu, Q, I, Y = solve_FBSDE(Q0, g1, g2, g3, g4, beta, c, k, sigma_transient, T, nb_t)
    
    return ts, nu, Q, Y, I, alpha, F, g1, g2, g3, g4

###############################################
# Solving the problem win the general case
###############################################
def solve_FBSDE(Q0, g1, g2, g3, g4, beta, c, k, sigma_transient, T, nb_t):
    dt       = T / (nb_t - 1)

    Q        = np.empty((nb_t, 1))
    I        = np.empty((nb_t, 1))
    Y        = np.empty((nb_t, 1))
    nu       = np.empty((nb_t, 1))

    Q[0]     = Q0
    I[0]     = 0
    Y[0]     = 0
    
    M2 = np.random.randn(nb_t, 1) * sigma_transient * np.sqrt(dt)
    
    # FBSDE system
    for i in range(0, nb_t-1):
        nu[i]   = (g1[i] + g2[i] * Q[i] + g3[i] * I[i] + g4[i] * Y[i]) / 2 / k
        Q[i+1]  = Q[i] + nu[i] * dt    
        Y[i+1]  = Y[i] + (Q[i] - beta * Y[i]) * dt
        I[i+1]  = I[i] + c * nu[i] * dt - beta * I[i] * dt + M2[i]

    return nu, Q, I, Y



###############################################
# Others
###############################################
def solve_g1_OU_closedform_numpy(q_target, g2, g3, k, c, phi, sigma_signal, r_signal, alpha, varphi, q_target_T, T, nb_t):
        dt = T / (nb_t - 1)
        ds = dt
        du = dt

        # Compute g1 with closed-form     
        ts        = np.linspace(0, T, nb_t)    
        g2_       = interp1d(ts, g2)
        g3_       = interp1d(ts, g3)
        q_target_ = interp1d(ts, q_target)

        constant_speed = (q_target_T - q_target[0]) / T

        expint1   = lambda t, s: np.exp( integrate.quad(lambda u: g2_(u) + c*g3_(u) , t, s,  epsabs=1e-5)[0] /2/k - r_signal*(s-t) ) 
        expint2   = lambda t, s: np.exp( integrate.quad(lambda u: g2_(u) + c*g3_(u) , t, s,  epsabs=1e-5)[0] /2/k ) 

        expterm1 = np.array([ integrate.quad(lambda s: expint1(t, s), t, T,  epsabs=1e-5)[0] for t in ts])
        expterm2 = np.array([ integrate.quad(lambda s: q_target_(s) * expint2(t, s), t, T,  epsabs=1e-5)[0] for t in ts])
        expterm3 = np.array([ integrate.quad(lambda s: expint2(t, s) * constant_speed, t, T,  epsabs=1e-5)[0]  for t in ts])
        expterm4 =  np.array([expint2(t, T) for t in ts])

        g1       = phi * expterm2 + alpha[:,0] * expterm1 - 2*varphi*expterm3 + q_target_T*expterm4*varphi

        return g1