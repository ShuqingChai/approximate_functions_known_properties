'''
Classes and methods to sample test data and data at each training iteration.
'''

import numpy as np
from scipy.stats import special_ortho_group

def generate_task_random(dim=4, beta=2):
    # Generate coefficients of the function
    initializer = np.random.uniform(0, 1, dim)
    Theta = beta*initializer
    sigma = np.random.uniform(0, beta)

    return Theta, sigma

def generate_feature_random(xy_bd, alpha=1000, beta=2):
    assert len(xy_bd) == 4 # xy_bd = [x1, x2, y1, y2]

    real_bds = np.array(xy_bd, dtype=float)
    initializer = np.random.uniform(0, 1, 4)
    x = real_bds[0]+(real_bds[1]-real_bds[0])*initializer[0]
    y = real_bds[2]+(real_bds[3]-real_bds[2])*initializer[1]
    v = beta * (initializer[2:] - 0.5)
    
    u = 0
    while u % 3 == 0:
        u = np.random.randint(1, alpha)
    u *= 5

    return np.array([x, y, v[0], v[1], u])

def equation(model, features, xy_bd, Theta, noncvx):

    bds = np.array(xy_bd, dtype=float)
    if features[0] < bds[0]: 
        x = bds[0]
    elif features[0] > bds[1]:
        x = bds[1]
    else:
        x = features[0]

    if features[1] < bds[2]:
        y = bds[2]
    elif features[1] > bds[3]:
        y = bds[3]
    else:
        y = features[1]

    # assert type(features[-1]) == int

    if model == "linear":
        y_sign = (y - bds[3])**2
        if noncvx:
            y_sign = y_sign if features[-1] % 2 == 0 else -y_sign
        response = np.dot([x, y_sign, 1, 1], Theta)
    elif model == "multiply":
        y_sign = Theta[1] / (y - bds[2] + (bds[3]-bds[2])*1e-1)
        if noncvx:
            y_sign = y_sign if features[-1] % 2 == 0 else -y_sign
            x = x if features[-1] % 2 == 0 else -x
        x = np.exp(x * Theta[0]) 
        response = x * y_sign
    else:
        raise NotImplementedError

    return response

def generate_data(model, num_data, xy_bd, Theta, sigma, noncvx):

    features = np.zeros((num_data, 5))
    responses = np.zeros((num_data, 1))

    for j in range(num_data):
        features[j,:] = generate_feature_random(xy_bd)
        responses[j] = equation(model, features[j,:], xy_bd, Theta, noncvx)

    return features, responses

class OnlineSampler(object):

    def __init__(self, model, num_data, xy_bd, noise="no_noise", noncvx=False):

        self.model = model
        self.num_data = num_data
        self.xy_bd = xy_bd
        self.params = generate_feature_random(xy_bd)
        self.Theta, self.sigma = generate_task_random()
        self.noise = noise
        self.noncvx = noncvx

    def sample(self):

        features, responses = generate_data(self.model, self.num_data, self.xy_bd, self.Theta, self.sigma, self.noncvx)
        return features, responses

    def get_fresh_values(self, params=None):

        if params is None:
            params = self.params

        response = equation(self.model, params, self.xy_bd, self.Theta, self.noncvx)
        if self.noise == "no_noise":
            pass
        elif self.noise == "fixed_variance":
            response += np.random.normal(0, self.sigma/2)
        else:
            raise NotImplementedError

        return response

    def loss(self, epsilons, c, antithetic):
        '''In this problem, the loss actually refers to the value of the function'''
        detailed_losses = np.zeros((epsilons.shape[0], 2))

        if antithetic:
            for l in range(epsilons.shape[0]):
                detailed_losses[l,0] = self.get_fresh_values(self.params+c*epsilons[l,:])
                detailed_losses[l,1] = self.get_fresh_values(self.params -c*epsilons[l,:])
        else:
            for l in range(epsilons.shape[0]):
                detailed_losses[l,0] = self.get_fresh_values(self.params+c*epsilons[l,:])
                detailed_losses[l,1] = self.get_fresh_values()
        
        return detailed_losses

    def pollute_observation(self, data):
        if self.noise == "no_noise":
            return data
        elif self.noise == "fixed_variance":
            return data + np.random.normal(0, self.sigma/2, data.shape)
        else:
            raise NotImplementedError
