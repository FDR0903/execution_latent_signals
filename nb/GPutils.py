import itertools
import torch
import gpytorch
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def solveLotkaVolterra(_phi, _beta, _c, _T, _k, _nb_t=200000):
    _ts         = np.linspace(0, _T, _nb_t)
    _a          = np.array([_phi, -_beta, -_beta*_beta*_c])
    
    _b          = np.array([0, -_beta, _beta])
    
    _A          = np.array([[-1, _c, 1/_beta],
                           [-1, _c, 1/_beta],
                           [-1, _c, 1/_beta]])/2/_k
    
    _Gt         = lambda t, s: -_a + -np.diag(s) @ (_b + _A@s)    
    
    _sol        = solve_ivp(_Gt, 
                           [_T, 0], 
                           np.array([0, 0, _beta * _c]), 
                           t_eval = _ts[::-1])
    _Gt         = _sol.y
    
    return _Gt, _ts

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, training_iterations):
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(l=0.01))
        self.likelihood = likelihood
        self.reward_observation_times = []
        self.train_x = train_x
        self.train_y = train_y
        self.training_iterations = training_iterations
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def trainn(self):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        #
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            
def add_fwd_cols(all_fwd_cols, _LOB_features):
    for col in all_fwd_cols:
        fwd_w = int(col.split('_')[-1])
        _LOB_features[col] = _LOB_features['mid_price'].diff(fwd_w).shift(-fwd_w)
        
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
    
