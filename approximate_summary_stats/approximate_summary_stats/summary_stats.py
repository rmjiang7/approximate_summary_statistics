import torch
from sciope.utilities.summarystats.summary_base import SummaryBase
import numpy as np

class PyTorchANN_Statistics(SummaryBase):
    """
    The thetas predicted by ANN models act as summary statistics
    """

    def __init__(self, model_eval, mean_trajectories=False, use_logger=False, device = 'cpu'):
        self.name = 'PyTorchANN_Statistics'
        self.model_eval = model_eval
        self.device_name = device
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        super(PyTorchANN_Statistics, self).__init__(self.name, mean_trajectories, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("PyTorchANN_Statistics summary statistic initialized")

    def compute(self, data_arr):
        """
        Calculate the value(s) of the summary statistic(s)

        Parameters
        ----------
        data : [type]
            simulated or data set in the form N x S X T - num data points x num species x num time steps

        Returns
        -------
        [type]
            computed statistic value

        """
        assert len(data_arr.shape) == 3, "required input shape is (n_points, n_species, n_timepoints)"

        if self.device_name == 'cpu':
            res = self.model_eval(data_arr).detach().numpy()
            #res = self.model_eval(torch.tensor(data_arr.astype(np.float32))).detach().numpy()
        else:
            res = self.model_eval(data_arr.to(self.device)).detach().cpu().numpy()
            #res = self.model_eval(torch.tensor(data_arr.astype(np.float32)).to(self.device)).detach().cpu().numpy()

        if self.mean_trajectories:
            res = np.asarray(np.mean(res, axis=0))  # returns a scalar, so we cast it as an array

        if self.use_logger:
            self.logger.info("ANN_Statistics summary statistic: processed data matrix of shape {0} and generated summaries"
                             " of shape {1}".format(data.shape, res.shape))
        return res
    
class SciopeANN_Statistics(SummaryBase):
    """
    The thetas predicted by ANN models act as summary statistics
    """

    def __init__(self, model_eval, mean_trajectories=False, use_logger=False):
        self.name = 'PyTorchANN_Statistics'
        self.model_eval = model_eval
        super(SciopeANN_Statistics, self).__init__(self.name, mean_trajectories, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("PyTorchANN_Statistics summary statistic initialized")

    def compute(self, data):
        """
        Calculate the value(s) of the summary statistic(s)

        Parameters
        ----------
        data : [type]
            simulated or data set in the form N x S X T - num data points x num species x num time steps

        Returns
        -------
        [type]
            computed statistic value

        """
        data_arr = np.array(data)
        assert len(data_arr.shape) == 3, "required input shape is (n_points, n_species, n_timepoints)"

        res = self.model_eval.predict(data_arr)

        if self.mean_trajectories:
            res = np.asarray(np.mean(res, axis=0))  # returns a scalar, so we cast it as an array

        if self.use_logger:
            self.logger.info("ANN_Statistics summary statistic: processed data matrix of shape {0} and generated summaries"
                             " of shape {1}".format(data.shape, res.shape))
        return res
