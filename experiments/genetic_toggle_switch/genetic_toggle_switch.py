import gillespy2
import numpy as np

from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from sciope.utilities.priors.uniform_prior import UniformPrior

from approximate_summary_stats.simulators import *
from approximate_summary_stats.nets.cnn import CNN
from approximate_summary_stats.nets.mlp import MLP
import torch

class genetic_toggle_switch(gillespy2.Model):
    
    def __init__(self, 
                 parameter_values=None,
                 start_time = 0,
                 dt = 1, 
                 end_time = 200):
        # initialize Model
        gillespy2.Model.__init__(self, name="toggle_switch")
        
        params = {
            'alpha1' : 1,
            'alpha2' : 1,
            'beta' : 2,
            'gamma' : 2,
            'mu' : 1,
            'U' : 10,
            'V' : 10
        }
        
        if parameter_values is not None:
            params.update(parameter_values)

        # parameters
        self.add_parameter(Parameter(name='alpha1', expression = params['alpha1']))
        self.add_parameter(Parameter(name='alpha2', expression = params['alpha2']))
        self.add_parameter(Parameter(name='beta', expression = params['beta']))
        self.add_parameter(Parameter(name='gamma', expression = params['gamma']))
        self.add_parameter(Parameter(name='mu', expression = params['mu']))

        # Species
        self.add_species(Species(name='U', initial_value = params['U'], mode = 'discrete'))
        self.add_species(Species(name='V', initial_value = params['V'], mode = 'discrete'))

        # reactions
        self.add_reaction(Reaction(name="r1", reactants={}, products = {'U' : 1},
                                propensity_function="alpha1/(1 + pow(V, beta))"))
        self.add_reaction(Reaction(name="r2", reactants={}, products = {'V' : 1},
                                propensity_function="alpha2/(1 + pow(U, gamma))"))
        self.add_reaction(Reaction(name="r3", reactants={'U' : 1}, products = {}, rate = self.listOfParameters['mu']))
        self.add_reaction(Reaction(name="r4", reactants={'V' : 1}, products = {}, rate = self.listOfParameters['mu']))
                                
        self.timespan(np.arange(start_time, end_time, dt))



parameter_names = ['alpha1','alpha2','beta','gamma','mu']
lower_bounds = [0,0,0,0,0]
upper_bounds = [6,6,6,6,6]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))

m = genetic_toggle_switch

def parameter_transform(param):
    return param

def output_transform(res):
    return np.vstack([res['U'],res['V']]).reshape((1,2,200))

def fail_transform(res):
    return np.inf * np.ones((1,2,200))

model = SKMInferenceModel(
            model = m,
            prior = prior,
            param_names = parameter_names,
            param_transform = parameter_transform,
            output_transform = output_transform,
            fail_transform = fail_transform,
            timeout = 3
)

true_params = np.array([1,1,2,2,1])
obs_data = model.simulate_ssa(true_params)

def result_filter(res):
    return not np.all(res == np.inf) and not np.any(np.isnan(res)) and np.all(res >= -1)

summary_stat_encoder = CNN((2,200), 5)

class RatioEstimator(torch.nn.Module):

    def __init__(self, input_shape, param_shape, sample_encoder, param_encoder, output_layer):
        super().__init__()
        self.input_shape = input_shape
        self.param_shape = param_shape
        self.sample_encoder = sample_encoder
        self.param_encoder = param_encoder
        self.output_layer = output_layer

    def forward(self, x):
        t,p = x[:,:-self.param_shape].view(-1,self.input_shape[0], self.input_shape[1]), x[:,-self.param_shape:]

        encoded_t = self.sample_encoder(t)
        # B x D1
        encoded_p = self.param_encoder(p)
        # B x D2
        out = self.output_layer(torch.cat((encoded_t, encoded_p),1))
        return out

ratio_estimator = RatioEstimator((2, 200), 5,
                                 CNN((2, 200), 5, con_len = 3, con_layers = [10, 10], dense_layers = [25, 25]),
                                 MLP(5, 5, [10,10]),
                                 MLP(10, 1, [10, 5]))
approx_simulator = model.simulate_tau
