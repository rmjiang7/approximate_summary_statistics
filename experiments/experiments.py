import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("model_name", help = "model name to run", type = str, choices = ['lv', 'gts', 'pb', 'vilar'])
parser.add_argument("task", help = "task to run", type = str, choices = ['approx_only', 'full_only', 'mixed', 'mixed_saved'])

parser.add_argument("--M", type = int, help = "number of ratio estimator samples to gather", default = 5000)
parser.add_argument("--N", type = int, help = "number of total samples to use", default = 100000)
parser.add_argument("--rho", type = float, help = "one-sided rho cutoff", default = 0.05)
parser.add_argument("--rho_quantile", type = float, help = "proportion of N to resample from full model", default = -1)
parser.add_argument("--saved", type = bool, help = "use previous simulations instead", default = False)
parser.add_argument("--saved_test", type = bool, help = "use previous simulations in test", default = True)
parser.add_argument("--save_summary_statistic", type = bool, help = "save summary statistic after training", default = False)
parser.add_argument("--load_summary_statistic", type = bool, help = "use previous summary statistic", default = False)
parser.add_argument("--test", type = bool, help = "evaluate on hold out test set", default = False)
parser.add_argument("--N_test", type = int, help = "number of test samples to use for evaluation", default = 300000)
parser.add_argument("--ss_patience", type = int, help = "patience for training the summary statistic", default = 10)
parser.add_argument("--ratio_patience", type = int, help = "patience for training the ratio estimator", default = 15)
parser.add_argument("--save_simulations", type = bool, help = "save simulation trajectories", default = False)
args = parser.parse_args()

import pickle
import torch
torch.set_num_threads(5)

import dask
dask.config.set(scheduler = 'processes', workers = 5)

from approximate_summary_stats.run_experiments import *


if args.model_name == 'lv':
    from lotka_volterra.lk import *
    base_dir = './lotka_volterra'
elif args.model_name == 'gts':
    from genetic_toggle_switch.genetic_toggle_switch import *
    base_dir = './genetic_toggle_switch'
elif args.model_name == 'pb':
    from pure_birth.pb import *
    base_dir = './pure_birth'
elif args.model_name == 'vilar':
    from vilar_oscillator.vilar import *
    base_dir = './vilar_oscillator'
    
import scipy.special as sc
olderr = sc.seterr(all='ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

N = args.N
M = args.M
N_test = args.N_test
saved = args.saved
save_simulations = args.save_simulations
save_summary_statistic = args.save_summary_statistic
load_summary_statistic = args.load_summary_statistic
saved_test = args.saved_test
task = args.task
test = args.test
rho = args.rho
rho_quantile = args.rho_quantile
ss_patience = args.ss_patience
re_patience = args.ratio_patience

if saved:
    if not os.path.isdir(base_dir + "/saved_simulations"):
        print("No saved simulations exist.")
        saved = False 

if save_simulations and not os.path.isdir(base_dir + "/saved_simulations"):
    os.mkdir(base_dir + "/saved_simulations")

if task == 'approx_only':

    if not load_summary_statistic or (load_summary_statistic and not os.path.isfile(base_dir + "/saved_simulations/approx_only_summary_statistics.pkl")):
        if saved:
            if os.path.isfile(base_dir + "/saved_simulations/approx_sims_train.npy"):
                params = np.load(base_dir + "/saved_simulations/params_train.npy")
                approx_sims = np.load(base_dir + "/saved_simulations/approx_sims_train.npy")
                if N < params.shape[0]:
                    tidces = np.random.choice(params.shape[0], N, replace = False)
                    params = params[tidces]
                    approx_sims = approx_sims[tidces]
            else:
                print("No saved simulations exist.")
                sys.exit(1)
        else:
            print("Generating samples...")
            params, approx_sims = model.generate_samples([approx_simulator], N, result_filter, batch_size = 50000)
            approx_sims = approx_sims[0][:,0,:,:]
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_train.npy", params)
                np.save(base_dir + "/saved_simulations/approx_sims_train.npy", approx_sims)


        normalized_params = normalize_data(params, lower_bounds, upper_bounds)
        if args.model_name == 'vilar':
            approx_sims[approx_sims < 1] = 0
            approx_sims = np.exp(np.log(approx_sims + 0.0001) + np.random.normal(0, 0.4, size = approx_sims.shape))
        summary_statistic = train_summary_statistic(summary_stat_encoder, approx_sims, normalized_params, lr = 1e-3, batch_size = 512, seed = None, patience = ss_patience)

        if save_summary_statistic:
            f = open(base_dir + "/saved_simulations/approx_only_summary_statistics.pkl", "wb")
            pickle.dump(summary_statistic, f)
    else:
        print("Loading summary statistic...")
        f = open(base_dir + "/saved_simulations/approx_only_summary_statistics.pkl", "rb")
        summary_statistic = pickle.load(f)
        print("Done")

    if test:
        if saved_test:
            if os.path.isfile(base_dir + "/saved_simulations/params_test.npy"):
                test_params = np.load(base_dir + "/saved_simulations/params_test.npy")
                test_full_sims = np.load(base_dir + "/saved_simulations/full_sims_test.npy")
            else:
                print("No saved test simulations.  Simulating...")
                test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
                test_full_sims = test_full_sims[0][:,0,:,:]
                if save_simulations:
                    np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                    np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)

        else:
            test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
            test_full_sims = test_full_sims[0][:,0,:,:]
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)
        
        if args.model_name == 'vilar':
            test_full_sims = np.exp(np.log(test_full_sims + 0.0001) + np.random.normal(0, 0.4, size = test_full_sims.shape))

        test_full_sims = torch.tensor(test_full_sims.astype(np.float32))
        mae, e = evaluate_error_metrics(summary_statistic, test_full_sims, test_params, lower_bounds, upper_bounds)
        print("MAE : {}, E : {}".format(mae, e))
        with open('test_error_approx_{}.csv'.format(args.model_name), 'a+') as f:
            f.write('{}, {}\n'.format(mae, e))


elif task == 'full_only':

    if not load_summary_statistic or (load_summary_statistic and not os.path.isfile(base_dir + "/saved_simulations/full_only_summary_statistics.pkl")):
        if saved:
            if os.path.isfile(base_dir + "/saved_simulations/full_sims_train.npy"):
                params = np.load(base_dir + "/saved_simulations/params_train.npy")
                full_sims = np.load(base_dir + "/saved_simulations/full_sims_train.npy")
                if N < params.shape[0]:
                    tidces = np.random.choice(params.shape[0], N, replace = False)
                    params = params[tidces]
                    full_sims = full_sims[tidces]
            else:
                print("No saved simulations exist.")
                sys.exit(1)

        else:
            params, full_sims = model.generate_samples([model.simulate_ssa_n], N, result_filter, batch_size = 50000)
            full_sims = full_sims[0][:,0,:,:]
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_train.npy", params)
                np.save(base_dir + "/saved_simulations/full_sims_train.npy", full_sims)
        
        if args.model_name == 'vilar':
            full_sims = np.exp(np.log(full_sims + 0.0001) + np.random.normal(0, 0.4, size = full_sims.shape))

        normalized_params = normalize_data(params, lower_bounds, upper_bounds)
        summary_statistic = train_summary_statistic(summary_stat_encoder, full_sims, normalized_params, lr = 1e-3, batch_size = 512, seed = None, patience = ss_patience)
        
        if save_summary_statistic:
            f = open(base_dir + "/saved_simulations/full_only_summary_statistics.pkl", "wb")
            pickle.dump(summary_statistic, f)
    else:
        print("Loading summary statistic...")
        f = open(base_dir + "/saved_simulations/full_only_summary_statistics.pkl", "rb")
        summary_statistic = pickle.load(f)
        print("Done")

    if test:
        if saved_test:
            if os.path.isfile(base_dir + "/saved_simulations/params_test.npy"):
                test_params = np.load(base_dir + "/saved_simulations/params_test.npy")
                test_full_sims = np.load(base_dir + "/saved_simulations/full_sims_test.npy")
            else:
                print("No saved test simulations.  Simulating...")
                test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
                test_full_sims = test_full_sims[0][:,0,:,:]
                if save_simulations:
                    np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                    np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)

        else:
            test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
            test_full_sims = test_full_sims[0][:,0,:,:]
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)
        
        if args.model_name == 'vilar':
            test_full_sims = np.exp(np.log(test_full_sims + 0.0001) + np.random.normal(0, 0.4, size = test_full_sims.shape))

        test_full_sims = torch.tensor(test_full_sims.astype(np.float32))
        mae, e = evaluate_error_metrics(summary_statistic, test_full_sims, test_params, lower_bounds, upper_bounds)
        print("MAE : {}, E : {}".format(mae, e))
        with open('test_error_full_{}.csv'.format(args.model_name), 'a+') as f:
            f.write('{}, {}\n'.format(mae, e))

elif task == 'mixed':
    
    if not load_summary_statistic or (load_summary_statistic and not os.path.isfile(base_dir + "/saved_simulations/approx_summary_statistics.pkl")):
        ratio_params, ratio_sims = model.generate_samples([model.simulate_ssa_n, approx_simulator], M, result_filter, batch_size = 50000)
        ratio_full_sims, ratio_approx_sims = ratio_sims
        ratio_full_sims = ratio_full_sims[:,0,:,:]
        ratio_approx_sims = ratio_approx_sims[:,0,:,:]
        
        if args.model_name == 'vilar':
            ratio_full_sims = np.exp(np.log(ratio_full_sims + 0.0001) + np.random.normal(0, 0.4, size = ratio_full_sims.shape))
            ratio_approx_sims[ratio_approx_sims < 1] = 0
            ratio_approx_sims = np.exp(np.log(ratio_approx_sims + 0.0001) + np.random.normal(0, 0.4, size = ratio_approx_sims.shape))
        
        normalized_ratio_params = normalize_data(ratio_params, lower_bounds, upper_bounds)
        
        # train ratio estimator 
        train_ratio_estimator(ratio_estimator, ratio_full_sims, ratio_approx_sims, normalized_ratio_params, lr = 5e-4, batch_size = 64, patience = re_patience)
        
        # build approximate dataset
        print("Constructing initial approximate dataset")
        params, sims = model.generate_samples([approx_simulator], N - M, result_filter, batch_size = 50000)
        approx_sims = sims[0][:,0,:,:]

        if args.model_name == 'vilar':
            approx_sims[approx_sims < 1] = 0
            approx_sims = np.exp(np.log(approx_sims + 0.0001) + np.random.normal(0, 0.4, size = approx_sims.shape))

        normalized_params = normalize_data(params, lower_bounds, upper_bounds)
        
        probs_approx = torch.sigmoid(ratio_estimator.eval()(torch.tensor(np.hstack([approx_sims.reshape(approx_sims.shape[0], -1), normalized_params]).astype(np.float32)))).detach().numpy()[:,0]
        sorted_probs = list(np.argsort(probs_approx))
        
        if rho_quantile > 0:
            one_sided = int((rho_quantile * params.shape[0])/2)
            resample_indices = sorted_probs[:one_sided] + sorted_probs[-one_sided:]
            approx_indces = sorted_probs[one_sided:len(sorted_probs) - one_sided]
        else:
            resample_indices = [i for i in sorted_probs if probs_approx[i] < rho or probs_approx[i] > (1 - rho)]
            approx_indces = [i for i in sorted_probs if probs_approx[i] >= rho and probs_approx[i] <= (1 - rho)]
        
        # resample from full model
        print("Resampling from full")
        _, full_params, full_sims = model._draw_samples(model.simulate_ssa_n, params[resample_indices], result_filter, batch_size = 50000)
        full_sims = full_sims[:,0,:,:]
        
        if args.model_name == 'vilar':
            full_sims = np.exp(np.log(full_sims + 0.0001) + np.random.normal(0, 0.4, size = full_sims.shape))

        mixed_params = [ratio_params]
        mixed_sims = [ratio_full_sims]
        
        mixed_params.append(params[approx_indces])
        mixed_sims.append(approx_sims[approx_indces])
        
        mixed_params.append(full_params)
        mixed_sims.append(full_sims)

        mixed_params = np.vstack(mixed_params)
        mixed_sims = np.vstack(mixed_sims)
        
        print("Training summary statistic with {} full_samples".format(full_params.shape[0] + M))
        normalized_mixed_params = normalize_data(mixed_params, lower_bounds, upper_bounds)
        summary_statistic = train_summary_statistic(summary_stat_encoder, mixed_sims, normalized_mixed_params, lr = 1e-3, batch_size = 512, seed = None, patience = ss_patience)
        
        if save_summary_statistic:
            f = open(base_dir + "/saved_simulations/approx_summary_statistics.pkl", "wb")
            pickle.dump(summary_statistic, f)
    else:
        f = open(base_dir + "/saved_simulations/approx_summary_statistics.pkl", "rb")
        summary_statistic = pickle.load(f)

    if test:
        if saved_test:
            if os.path.isfile(base_dir + "/saved_simulations/params_test.npy"):
                test_params = np.load(base_dir + "/saved_simulations/params_test.npy")
                test_full_sims = np.load(base_dir + "/saved_simulations/full_sims_test.npy")
            else:
                print("No saved test simulations.  Simulating...")
                test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
                test_full_sims = test_full_sims[0][:,0,:,:]
                if save_simulations:
                    np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                    np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)

        else:
            test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)
        
        if args.model_name == 'vilar':
            test_full_sims = np.exp(np.log(test_full_sims + 0.0001) + np.random.normal(0, 0.4, size = test_full_sims.shape))

        test_full_sims = torch.tensor(test_full_sims.astype(np.float32))
        mae, e = evaluate_error_metrics(summary_statistic, test_full_sims, test_params, lower_bounds, upper_bounds)
        print("MAE : {}, E : {}".format(mae, e))
        with open('test_error_mixed_{}.csv'.format(args.model_name), 'a+') as f:
            f.write('{}, {}\n'.format(mae, e))

elif task == 'mixed_saved':
    
    if os.path.isfile(base_dir + "/saved_simulations/approx_sims_train.npy") and os.path.isfile(base_dir + "/saved_simulations/full_sims_train.npy"):
        params = np.load(base_dir + "/saved_simulations/params_train.npy")
        approx_sims = np.load(base_dir + "/saved_simulations/approx_sims_train.npy")
        full_sims = np.load(base_dir + "/saved_simulations/full_sims_train.npy")
        if N < params.shape[0]:
            tidces = np.random.choice(params.shape[0], N, replace = False)
            params = params[tidces]
            approx_sims = approx_sims[tidces]
            full_sims = full_sims[tidces]
    else:
        print("No saved simulations exist.")
        sys.exit(1)

    normalized_params = normalize_data(params, lower_bounds, upper_bounds)
    
    if args.model_name == 'vilar':
        full_sims = np.exp(np.log(full_sims + 0.0001) + np.random.normal(0, 0.4, size = full_sims.shape))
        approx_sims[approx_sims < 1] = 0
        approx_sims = np.exp(np.log(approx_sims + 0.0001) + np.random.normal(0, 0.4, size = approx_sims.shape))

    # train ratio estimator 
    print("Training ratio estimator with M = {}".format(M))
    re_indices = np.random.choice(params.shape[0], M, replace = False)
    train_ratio_estimator(ratio_estimator, full_sims[re_indices], approx_sims[re_indices], normalized_params[re_indices], lr = 5e-4, batch_size = 256, patience = re_patience)
    
    # build approximate dataset
    probs_approx = torch.sigmoid(ratio_estimator.eval()(torch.tensor(np.hstack([approx_sims.reshape(approx_sims.shape[0], -1), normalized_params]).astype(np.float32)))).detach().numpy()[:,0]
    sorted_probs = list(np.argsort(probs_approx))

    if rho_quantile > 0:
       one_sided = int((rho_quantile * params.shape[0])/2)
       resample_indices = sorted_probs[:one_sided] + sorted_probs[-one_sided:]
    else:
       resample_indices = [i for i in sorted_probs if probs_approx[i] < rho or probs_approx[i] > (1 - rho)]

    mixed_sims = approx_sims.copy()
    for j in resample_indices:
        mixed_sims[j,:,:] = full_sims[j,:,:].copy()

    for j in re_indices:
        mixed_sims[j,:,:] = full_sims[j,:,:].copy()
    
    # train summary statistic
    print("Training summary statistic with {} full samples".format(len(resample_indices) + len(re_indices)))
    summary_statistic = train_summary_statistic(summary_stat_encoder, mixed_sims, normalized_params, lr = 1e-3, batch_size = 512, seed = None, patience = ss_patience)

    if test:
        if saved_test:
            if os.path.isfile(base_dir + "/saved_simulations/params_test.npy"):
                test_params = np.load(base_dir + "/saved_simulations/params_test.npy")
                test_full_sims = np.load(base_dir + "/saved_simulations/full_sims_test.npy")
            else:
                print("No saved test simulations.  Simulating...")
                test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
                test_full_sims = test_full_sims[0][:,0,:,:]
                if save_simulations:
                    np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                    np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)

        else:
            test_params, test_full_sims = model.generate_samples([model.simulate_ssa_n], N_test, result_filter, batch_size = 50000)
            if save_simulations:
                np.save(base_dir + "/saved_simulations/params_test.npy", test_params)
                np.save(base_dir + "/saved_simulations/full_sims_test.npy", test_full_sims)
        
        if args.model_name == 'vilar':
            test_full_sims = np.exp(np.log(test_full_sims + 0.0001) + np.random.normal(0, 0.4, size = test_full_sims.shape))

        test_full_sims = torch.tensor(test_full_sims.astype(np.float32))
        mae, e = evaluate_error_metrics(summary_statistic, test_full_sims, test_params, lower_bounds, upper_bounds)
        print("MAE : {}, E : {}".format(mae, e))
        with open('test_error_mixed_{}.csv'.format(args.model_name), 'a+') as f:
            f.write('{}, {}\n'.format(mae, e))
