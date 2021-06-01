import gillespy2
import numpy as np

from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from sciope.utilities.priors.uniform_prior import UniformPrior

from approximate_summary_stats.simulators import *
from approximate_summary_stats.nets.cnn import CNN
from approximate_summary_stats.nets.mlp import MLP

class lotka_volterra(gillespy2.Model):
    
    def __init__(self, 
                 parameter_values=None,
                 start_time = 0,
                 dt = 0.2,
                 end_time = 30):
        # initialize Model
        gillespy2.Model.__init__(self, name="lotka_volterra")
       

        # parameters
        params = {
            'k1' : 0.01,
            'k2' : 0.5,
            'k3' : 1.0,
            'k4' : 0.01,
            'predator' : 50,
            'prey' : 100
        }
        
        if parameter_values is not None:
            params.update(parameter_values)
        
        self.add_parameter(Parameter(name='k1', expression = params['k1']))
        self.add_parameter(Parameter(name='k2', expression = params['k2']))
        self.add_parameter(Parameter(name='k3', expression = params['k3']))
        self.add_parameter(Parameter(name='k4', expression = params['k4']))

        # Species
        self.add_species(Species(name='predator', initial_value = params['predator'], mode = 'discrete'))
        self.add_species(Species(name='prey', initial_value = params['prey'], mode = 'discrete'))

        # reactions
        self.add_reaction(Reaction(name="r1", reactants={'predator': 1, 'prey': 1}, products = {'predator' : 2},
                                rate=self.listOfParameters['k1']))

        self.add_reaction(Reaction(name="r2", reactants={'predator': 1}, products = {}, rate = self.listOfParameters['k2']))

        self.add_reaction(Reaction(name="r3", reactants={'prey': 1}, products = {'prey' : 2}, rate = self.listOfParameters['k3']))
        self.add_reaction(Reaction(name="r4", reactants={'prey': 1, 'predator': 1}, products = {'predator': 1}, rate = self.listOfParameters['k4']))

        self.timespan(np.arange(start_time, end_time, dt))

parameter_names = ['k1', 'k2', 'k3', 'k4']
lower_bounds = [-6, -6, -6, -6]
upper_bounds = [ 2,  2,  2,  2]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))

m = lotka_volterra

def parameter_transform(param):
    param_update = param.copy()
    param_update['k1'] = np.exp(param['k1'])
    param_update['k2'] = np.exp(param['k2'])
    param_update['k3'] = np.exp(param['k3'])
    param_update['k4'] = np.exp(param['k4'])
    return param_update

def output_transform(res):
    return np.vstack([res['prey'], res['predator']]).reshape(1,2,-1)

def fail_transform(res):
    return np.inf * np.ones((1,2,150))

model = SKMInferenceModel(
        model = m,
        prior = prior,
        param_names = parameter_names,
        param_transform = parameter_transform,
        output_transform = output_transform,
        fail_transform = fail_transform,
        timeout = 10
)

true_params = np.log([0.01, 0.5, 1.0, 0.01])
obs_data = model.simulate_ssa(true_params)

def result_filter(res):
    return not np.all(res == np.inf) and not np.any(np.isnan(res)) and np.all(res >= -1)

summary_stat_encoder = CNN((2,150), 4, con_len = 3, con_layers = [10,10], dense_layers = [25, 25])
ratio_estimator = MLP(304, 1, [100,100])
approx_simulator = model.simulate_ode
