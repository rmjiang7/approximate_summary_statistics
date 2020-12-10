import gillespy2
import numpy as np

from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from sciope.utilities.priors.uniform_prior import UniformPrior

from approximate_summary_stats.simulators import *
from approximate_summary_stats.nets.cnn import CNN
from approximate_summary_stats.nets.mlp import MLP

import torch

class pure_birth(gillespy2.Model):
    
    def __init__(self, 
                 parameter_values=None,
                 start_time = 0,
                 dt = 0.2, 
                 end_time = 19.8):
        # initialize Model
        gillespy2.Model.__init__(self, name="pure_birth")
        
        params = {
            'k1' : 30,
            'S' : 300
        }
        
        if parameter_values is not None:
            params.update(parameter_values)

        # parameters
        self.add_parameter(Parameter(name='k1', expression = params['k1']))

        # Species
        self.add_species(Species(name='S', initial_value = params['S'], mode = 'discrete'))

        # reactions
        self.add_reaction(Reaction(name="birth", reactants={}, products = {'S' : 1},
                                rate=self.listOfParameters['k1']))

        self.timespan(np.arange(start_time, end_time, dt))


parameter_names = ['k1']
lower_bounds = [0]
upper_bounds = [10000]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))

m = pure_birth

def parameter_transform(param):
    return param

def output_transform(res):
    return np.vstack(res['S']).reshape((1,1,99))

def fail_transform(res):
    return np.inf * np.ones((1,1,99))

model = SKMInferenceModel(
        model = m,
        prior = prior,
        param_names = parameter_names,
        param_transform = parameter_transform,
        output_transform = output_transform,
        fail_transform = fail_transform,
        timeout = 3
)


true_params = np.array([2432])
obs_data = model.simulate_ssa(true_params)

def result_filter(res):
    return not np.all(res == np.inf) and not np.any(np.isnan(res)) and np.all(res >= -1)

summary_stat_encoder = CNN((1,99), 1, con_len = 3, con_layers = [5,5], dense_layers = [25, 25])

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

ratio_estimator = RatioEstimator((1, 99), 1,
                                 CNN((1, 99), 5, con_len = 3, con_layers = [5, 5], dense_layers = [25]),
                                 MLP(1, 5, [10, 10]),
                                 MLP(10, 1, [10, 5]))
approx_simulator = model.simulate_tau
