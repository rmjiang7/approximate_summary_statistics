import numpy as np
import torch
from torch.nn import functional as F
from approximate_summary_stats.summary_stats import PyTorchANN_Statistics
from approximate_summary_stats.nets.trainers import DataLoaderTrainer

from sklearn.metrics import mean_absolute_error

def normalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    return (data - dmin)/(dmax - dmin)

def denormalize_data(data, dmin, dmax):
    dmin = np.array(dmin)
    dmax = np.array(dmax)
    return data * (dmax - dmin) + dmin


def train_summary_statistic(nn_model, sims, params,
                            seed = 42, batch_size = 100, train_pct = 0.9,
                            lr = 1e-3, device = 'cpu', patience = 5, scheduler_rate = None):

    train_loader, val_loader = DataLoaderTrainer.generate_train_valid(sims, params, batch_size, train_pct, seed = seed)

    def loss(output, y):
        return F.mse_loss(output, y, reduction = 'mean')

    optimizer = torch.optim.Adam(nn_model.parameters(), lr = lr)
    
    if scheduler_rate is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_rate)
    else:
        scheduler = None
    
    epoch_losses = DataLoaderTrainer.train(
        nn_model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        scheduler,
        patience = patience,
        n_epochs = 3000,
        device = device
    )

    cnn_stat = PyTorchANN_Statistics(nn_model.eval(), device = device)
    return cnn_stat

def train_ratio_estimator(nn_model, sims1, sims2, params,
                          seed = 42, batch_size = 100, train_pct = 0.9,
                          lr = 1e-3, device = 'cpu', 
                          patience = 5):

    x = np.hstack([np.vstack([sims1, sims2]).reshape(sims1.shape[0] * 2, -1), np.vstack([params, params])])
    labels = np.hstack([np.ones(sims1.shape[0]), np.zeros(sims1.shape[0])]).reshape(-1,1)

    train_loader, val_loader = DataLoaderTrainer.generate_train_valid(x, labels, batch_size, train_pct, seed = seed)

    def loss(output, y):
        return F.binary_cross_entropy_with_logits(output, y, reduction = 'mean')

    optimizer = torch.optim.Adam(nn_model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    epoch_losses = DataLoaderTrainer.train(
        nn_model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        scheduler,
        patience = patience,
        n_epochs = 3000,
        device = device
    )

def evaluate_error_metrics(summary_statistic, ref_sims, ref_params, lower_bounds, upper_bounds):
    summary_stats_ref = summary_statistic.compute(ref_sims)

    mae = mean_absolute_error(denormalize_data(summary_stats_ref, lower_bounds, upper_bounds), ref_params)
    e_pct = np.mean((4 / (np.array(upper_bounds) - np.array(lower_bounds))) * np.mean(np.abs(denormalize_data(summary_stats_ref, lower_bounds, upper_bounds) - ref_params), axis = 0))

    return mae, e_pct
