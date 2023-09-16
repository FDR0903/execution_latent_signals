import numpy as np
import matplotlib.pyplot as plt

####################################
# Simulation utilities
####################################
def simulate_ou_signal_price(r_signal, sigma_signal, sigma_price, F0, T, nb_t):

    dt = T / (nb_t - 1)
    
    W  = np.random.randn(nb_t, 1)
    M1 = np.random.randn(nb_t, 1) * sigma_price * np.sqrt(dt)

    # Simulate a signal:  OU process
    alpha        = np.empty((nb_t, 1))
    alpha[0]     = 0
    for i in range(1, nb_t):
        alpha[i] = alpha[i-1]  - r_signal * alpha[i-1] * dt + sigma_signal * W[i] * np.sqrt(dt)

    # Simulate the fundamental price
    F        = np.empty((nb_t, 1))
    F[0]     = F0
    for i in range(1, nb_t):
        F[i] = F[i-1]  + alpha[i-1] * dt + M1[i]
        
    return alpha, F 


####################################
# Plotting with LateX utilities
####################################
def updatePLT(W, l=4, w=3, fontsize=10):
    plt.rcParams.update({
        'figure.figsize': (W, W/(l/w)),     # 4:3 aspect ratio
        'font.size' : fontsize,                   # Set font size to 11pt
        'axes.labelsize': fontsize,               # -> axis labels
        'legend.fontsize': fontsize,              # -> legends
        'font.family': 'lmodern',
        'text.usetex': True,
        'text.latex.preamble': (            # LaTeX preamble
            r'\usepackage{lmodern}'
            # ... more packages if needed
        )
    })

def horizental_plot(ts, series, titles):
    updatePLT(W=2*len(series), l=3*len(series), w=3, fontsize=10)
    fig, axes = plt.subplots(1, len(series), constrained_layout=True, sharex=True)
    for (i, serie) in enumerate(series):
        axes[i].plot(ts, serie, color='k', lw=2.5, ls='-.')
        axes[i].set_title(titles[i])
    plt.show()
    return axes

def trading_simulation_plot(ts, nu, Q, q_target, I, Y, F, alpha):
    updatePLT(W=2*5, l=3*5, w=3, fontsize=10)
    fig, axes = plt.subplots(1, 5, constrained_layout=True, sharex=True)
    
    axes[0].plot(ts, nu, color='k', lw=2.5, ls='-.'); axes[0].set_title('Trading speed')
    
    axes[1].plot(ts, Q, color='k', lw=1.5, ls='-')
    axes[1].plot(ts, q_target, color='blue', lw=2.5, ls='-.')
    axes[1].legend(['Tracking', 'Target']); axes[1].set_title('Inventory')
    
    axes[2].plot(ts, Y, color='k', lw=2.5, ls='-.'); axes[2].set_title('Y')
    
    axes[3].plot(ts, alpha, color='k', lw=2.5, ls='-.'); axes[3].set_title('Signal')
    
    axes[4].plot(ts, F, color='k', lw=1, ls='-')
    axes[4].plot(ts, I+F, color='blue', lw=1, ls='-')
    axes[4].legend(['Fundamental', 'mid-price']); axes[4].set_title('Prices')
    
    plt.show()
    
    return axes

def simulations_plot(ts, q_target, trajectories, savefigname, legends=None, W=5):
    updatePLT(W=W, l=10, w=3, fontsize=10)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].plot(ts, q_target,  color='b', lw=2.5, ls='-.')

    for n in trajectories.keys():
        axes[0].plot(ts, trajectories[n][0],  color='lightcoral', lw=0.8, ls='-', alpha=0.6)
        axes[1].plot(ts, trajectories[n][1],  color='lightcoral', lw=0.8, ls='-.')

    
    axes[0].legend(['Target'])
    axes[0].set_title(r'Inventories')
    axes[1].set_title(r'Signal: $\alpha_t$')      

    plt.savefig(savefigname)

def parameter_study_plot(ts, q_target, trajectories, alpha, param_name, savefigname, W=5):
    updatePLT(W=W, l=10, w=3, fontsize=10)

    colors  = ('lightblue', 'lightcoral', 'tan', 'grey')
    isTuple = False
    
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].plot(ts, q_target,  color='b', lw=2.5, ls='-.')

    for (paramvalue, clr) in zip(trajectories.keys(), colors):
        if type(trajectories[paramvalue]) is tuple:
            axes[0].plot(ts, trajectories[paramvalue][0],  color=clr, lw=0.8, ls='-', alpha=1)
            axes[1].plot(ts, trajectories[paramvalue][1],  color=clr, lw=0.8, ls='-', alpha=1)
            isTuple = True
        else:
            axes[0].plot(ts, trajectories[paramvalue],  color=clr, lw=0.8, ls='-', alpha=1)

    axes[0].legend( [f'{param_name}={phi_}' for phi_ in trajectories.keys()] + ['Traget'], handlelength=0.3, framealpha=0.3 )
    
    axes[0].set_title(r'Inventories')
    axes[1].set_title(r'Signal: $\alpha_t$')      

    if not isTuple:     axes[1].plot(ts, alpha,     color='k', lw=2.5, ls='-.');  axes[1].set_title(r'$\alpha$') 

    plt.savefig(savefigname)
    plt.show()