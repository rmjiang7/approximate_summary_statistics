import numpy as np
import dask
from dask.diagnostics import ProgressBar

from gillespy2.solvers.cpp.variable_ssa_c_solver import VariableSSACSolver
from gillespy2.solvers.numpy import ODESolver, TauLeapingSolver, NumPySSASolver, TauHybridSolver
from gillespy2.core.gillespyError import SimulationError

class SKMInferenceModel(object):

    def __init__(self, model, prior,
                 param_names,
                 param_transform = lambda x: x,
                 output_transform = lambda x: x,
                 fail_transform = lambda _ : None,
                 timeout = None,
                 silent_error = False,
                 end_time = None,
                 dt = None):

        self._model = model
        self._prior = prior
        self._param_names = param_names
        self._timeout = timeout
        self._silent_error = silent_error
        self._end_time = end_time
        self._dt = dt

        self._param_transform = param_transform
        self._output_transform = output_transform
        self._fail_transform = fail_transform

        if end_time != None:
            if dt is not None:
                self._compiled_m = self._model(end_time = self._end_time, dt = self._dt)
            else:
                self._compiled_m = self._model(end_time = self._end_time)
        else:
            if dt is not None:
                self._compiled_m = self._model(dt = dt)
            else:
                self._compiled_m = self._model()
        self._compiled_solver = VariableSSACSolver(self._compiled_m)

    def _simulate_solver(self, param, solver, transform = True, n_trajectories = 1, seed = None, **kwargs):

        pv = {self._param_names[i] : param[i] for i in range(len(self._param_names))}
        pv = self._param_transform(pv)

        try:
            if self._end_time is not None:
                if self._dt is not None:
                    mv = self._model(pv, dt = self._dt, end_time = self._end_time)
                else:
                    mv = self._model(pv, end_time = self._end_time)
            else:
                if self._dt is not None:
                    mv = self._model(pv, dt = self._dt)
                else:
                    mv = self._model(pv)

            if solver == ODESolver:
                res = mv.run(solver = solver,
                     show_labels = True,
                     timeout = self._timeout,
                     **kwargs)
            else:
                res = mv.run(solver = solver,
                     number_of_trajectories = n_trajectories,
                     show_labels = True,
                     seed = seed if seed else np.random.randint(int(1e8)),
                     timeout = self._timeout,
                     **kwargs)

            if res.rc == 33:
                return self._fail_transform(res)

            if transform:
                return self._output_transform(res)
            else:
                return res
        except:
            if self._silent_error:
                return None
            else:
                raise

    def simulate_ssa(self, param, transform = True, n_trajectories = 1, seed = None, **kwargs):

        pv = {self._param_names[i] : param[i] for i in range(len(self._param_names))}
        pv = self._param_transform(pv)
        try:
            res = self._compiled_m.run(solver = self._compiled_solver,
                                   number_of_trajectories = n_trajectories,
                                   show_labels = True,
                                   variables = pv,
                                   seed = seed if seed else np.random.randint(int(1e8)),
                                   timeout = self._timeout)
        except SimulationError:
            if self._silent_error:
                return None
            else:
                raise

        if res.rc == 33:
            return self._fail_transform(res)

        if transform:
            return self._output_transform(res)
        else:
            return res

    def simulate_ssa_n(self, param, transform = True, n_trajectories = 1, seed = None, **kwargs):
        return self._simulate_solver(param, NumPySSASolver, transform, n_trajectories, **kwargs)

    def simulate_ode(self, param, transform = True, n_trajectories = 1, seed = None, **kwargs):
        return self._simulate_solver(param, ODESolver, transform, n_trajectories, **kwargs)

    def simulate_tau(self, param, transform = True, n_trajectories = 1, seed = None, **kwargs):
        return self._simulate_solver(param, TauLeapingSolver, transform, n_trajectories, **kwargs)

    def simulate_hybrid(self, param, transform = True, n_trajectories = 1, seed = None, **kwargs):
        return self._simulate_solver(param, TauHybridSolver, transform, n_trajectories, **kwargs)

    def generate_samples(self, simulators, N, result_filter = lambda x: True):

        params = np.vstack(dask.compute(self._prior.draw(N))[0])
        sim_res = []
        for simulator in simulators:
            indices, params, res = self._draw_samples(simulator, params, result_filter)
            for i in range(len(sim_res)):
                sim_res[i] = sim_res[i][indices]
            sim_res.append(res)

        return params, sim_res

    def _draw_samples(self, simulator, params, result_filter = lambda x: True):
        # Draw

        @dask.delayed
        def sim_reject(param):
            res = simulator(param)
            if result_filter(res):
                return res
            else:
                return None

        results = []
        for i in range(params.shape[0]):
            results.append(sim_reject(params[i,:]))

        with ProgressBar():
            computed_results, = dask.compute(results)

        # Filter the responses
        computed_results_filtered = [(i,c) for i,c in enumerate(computed_results) if c is not None]
        indices = [i for i,c in computed_results_filtered]
        computed_results_f = [c for i,c in computed_results_filtered]

        ts = np.asarray(computed_results_f)
        return indices, params[indices], ts
