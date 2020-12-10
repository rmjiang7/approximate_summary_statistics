import gillespy2
import numpy as np

from gillespy2 import Model, Species, Reaction, Parameter, RateRule, AssignmentRule, FunctionDefinition
from sciope.utilities.priors.uniform_prior import UniformPrior

from approximate_summary_stats.simulators import *
from approximate_summary_stats.nets.cnn import CNN
from approximate_summary_stats.nets.mlp import MLP
import torch

class vilar_oscillator(gillespy2.Model):
    def __init__(self, 
                 parameter_values = None,
                 start_time = 0,
                 dt = 1.,
                 end_time = 99.0):
        # initialize Model
        gillespy2.Model.__init__(self, name="vilar_oscillator")
        
        params = {
            'alpha_a' : 50,
            'alpha_a_prime' : 500,
            'alpha_r' : 0.01,
            'alpha_r_prime' : 50.0,
            'beta_a' : 50.0,
            'beta_r' : 5.0,
            'delta_ma' : 10.0,
            'delta_mr' : 0.5,
            'delta_a' : 1.0,
            'delta_r' : 0.2,
            'gamma_a' : 1.0,
            'gamma_r' : 1.0,
            'gamma_c' : 2.0,
            'Theta_a' : 50.0,
            'Theta_r' : 100.0,
            
            'Da' : 1,
            'Da_prime' : 0,
            'Ma' : 0,
            'Dr' : 0,
            'Dr_prime' : 1,
            'Mr' : 0,
            'C' : 10,
            'A' : 10,
            'R' : 10
        }
        
        if parameter_values is not None:
            params.update(parameter_values)
           
        # parameters
        alpha_a = gillespy2.Parameter(name='alpha_a', expression=params['alpha_a'])
        alpha_a_prime = gillespy2.Parameter(name='alpha_a_prime', expression=params['alpha_a_prime'])
        alpha_r = gillespy2.Parameter(name='alpha_r', expression=params['alpha_r'])
        alpha_r_prime = gillespy2.Parameter(name='alpha_r_prime', expression=params['alpha_r_prime'])
        beta_a = gillespy2.Parameter(name='beta_a', expression=params['beta_a'])
        beta_r = gillespy2.Parameter(name='beta_r', expression=params['beta_r'])
        delta_ma = gillespy2.Parameter(name='delta_ma', expression=params['delta_ma'])
        delta_mr = gillespy2.Parameter(name='delta_mr', expression=params['delta_mr'])
        delta_a = gillespy2.Parameter(name='delta_a', expression=params['delta_a'])
        delta_r = gillespy2.Parameter(name='delta_r', expression=params['delta_r'])
        gamma_a = gillespy2.Parameter(name='gamma_a', expression=params['gamma_a'])
        gamma_r = gillespy2.Parameter(name='gamma_r', expression=params['gamma_r'])
        gamma_c = gillespy2.Parameter(name='gamma_c', expression=params['gamma_c'])
        Theta_a = gillespy2.Parameter(name='Theta_a', expression=params['Theta_a'])
        Theta_r = gillespy2.Parameter(name='Theta_r', expression=params['Theta_r'])

        self.add_parameter([alpha_a, alpha_a_prime, alpha_r, alpha_r_prime, beta_a, beta_r, delta_ma, delta_mr,
                            delta_a, delta_r, gamma_a, gamma_r, gamma_c, Theta_a, Theta_r])

        # Species
        Da = gillespy2.Species(name='Da', initial_value=params['Da'])
        Da_prime = gillespy2.Species(name='Da_prime', initial_value=params['Da_prime'])
        Ma = gillespy2.Species(name='Ma', initial_value=params['Ma'])
        Dr = gillespy2.Species(name='Dr', initial_value=params['Dr'])
        Dr_prime = gillespy2.Species(name='Dr_prime', initial_value=params['Dr_prime'])
        Mr = gillespy2.Species(name='Mr', initial_value=params['Mr'])
        C = gillespy2.Species(name='C', initial_value=params['C'])
        A = gillespy2.Species(name='A', initial_value=params['A'])
        R = gillespy2.Species(name='R', initial_value=params['R'])

        self.add_species([Da, Da_prime, Ma, Dr, Dr_prime, Mr, C, A, R])

        # reactions
        # Reversible binding of the promoter for activator A
        #s_Da = gillespy2.Reaction(name="s_Da", reactants={Da_prime: 1}, products={Da: 1}, rate=Theta_a)
        s_Da_prime = gillespy2.Reaction(name="s_Da_prime", reactants={Da: 1, A: 1}, products={Da_prime: 1},
                                        rate=gamma_a)
        
        # Reversible binding of the promoter for creation of repressor R
        # s_Dr = gillespy2.Reaction(name="s_Dr", reactants={Dr_prime: 1}, products={Dr: 1}, rate=Theta_r)
        s_Dr_prime = gillespy2.Reaction(name="s_Dr_prime", reactants={Dr: 1, A: 1}, products={Dr_prime: 1},
                                        rate=gamma_r)
        
        # Transcription of mRNA for A at the activated rate
        s_Ma1 = gillespy2.Reaction(name="s_Ma1", reactants={Da_prime: 1}, products={Da_prime: 1, Ma: 1},
                                   rate=alpha_a_prime)
        
        # Transcription of mRNA for A at the basal rate
        s_Ma2 = gillespy2.Reaction(name="s_Ma2", reactants={Da: 1}, products={Da: 1, Ma: 1}, rate=alpha_a)
        
        # Spontaneous degradation rate of mRNA for A 
        a_Ma = gillespy2.Reaction(name="a_Ma", reactants={Ma: 1}, products={}, rate=delta_ma)
        
        # Translation of mRNA for A to protein A
        s_A1 = gillespy2.Reaction(name="s_A1", reactants={Ma: 1}, products={A: 1, Ma: 1}, rate=beta_a)
        
        # Unbinding of  of a A and R at the promoter site
        s_A2 = gillespy2.Reaction(name="S_A2", reactants={Da_prime: 1}, products={Da: 1, A: 1}, rate=Theta_a)
        s_A3 = gillespy2.Reaction(name="S_A3", reactants={Dr_prime: 1}, products={Dr: 1, A: 1}, rate=Theta_r)
        
        # Spontaneous degradation of protein A
        a_A = gillespy2.Reaction(name="a_A", reactants={A: 1}, products={}, rate=delta_a)
        
        # Production of complex C
        s_C = gillespy2.Reaction(name="s_C", reactants={A: 1, R: 1}, products={C: 1}, rate=gamma_c)
         
        # Transcription of mRNA for R at the activated rate
        S_Mr1 = gillespy2.Reaction(name="S_Mr1", reactants={Dr_prime: 1}, products={Dr_prime: 1, Mr: 1},
                                   rate=alpha_r_prime)
        
        # Transcription of mRNA for R at the basal rate
        S_Mr2 = gillespy2.Reaction(name="S_Mr2", reactants={Dr: 1}, products={Dr: 1, Mr: 1}, rate=alpha_r)
        
        # Spontaneous degradation of mRNA for R
        a_Mr = gillespy2.Reaction(name="a_Mr", reactants={Mr: 1}, products={}, rate=delta_mr)
        
        # Translation of mRNA for R to protein R
        s_R1 = gillespy2.Reaction(name="s_R1", reactants={Mr: 1}, products={Mr: 1, R: 1}, rate=beta_r)
        
        # Spontaneous degradation of protein R
        a_R = gillespy2.Reaction(name="a_R", reactants={R: 1}, products={}, rate=delta_r)
        
        # Breakdown of C into protein R
        s_r2 = gillespy2.Reaction(name="s_r2", reactants={C: 1}, products={R: 1}, rate=delta_a)

        self.add_reaction([s_Da_prime, s_Dr_prime, s_Ma1, s_Ma2, a_Ma, s_A1, s_A2, s_A3, a_A, s_C,
                           S_Mr1, S_Mr2, a_Mr, s_R1, a_R, s_r2])

        self.timespan(np.arange(start_time, end_time, dt))

parameter_names =  ['alpha_a', 'alpha_a_prime', 'alpha_r', 'alpha_r_prime', 'beta_a', 'beta_r', 'delta_ma', 'delta_mr', 'delta_a', 'delta_r', 'gamma_a', 'gamma_r', 'gamma_c', 'Theta_a', 'Theta_r']
lower_bounds = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
upper_bounds = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]
prior = UniformPrior(np.array(lower_bounds), np.array(upper_bounds))

m = vilar_oscillator

def parameter_transform(param):
    return param

def output_transform(res):
    return np.vstack([res['C'], res['A'], res['R']]).reshape(1,3,-1)

def fail_transform(res):
    return np.inf * np.ones((1,3,99))

model = SKMInferenceModel(
            model = m,
            prior = prior,
            param_names = parameter_names,
            param_transform = parameter_transform,
            output_transform = output_transform,
            fail_transform = fail_transform,
            timeout = 20000
)

true_params = np.array([50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0])
obs_data = model.simulate_ssa(true_params)

def result_filter(res):
    return not np.all(res == np.inf) and not np.any(np.isnan(res)) and np.all(res >= -1)

summary_stat_encoder = CNN((3, 99), 15)

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

ratio_estimator = RatioEstimator((3, 99), 15,
                                 CNN((3, 99), 5),
                                 MLP(15, 5, [10,10]),
                                 MLP(10, 1, [10, 5]))
approx_simulator = model.simulate_ode
