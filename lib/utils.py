####################################################
# SurvLatent ODE
# Author : Intae Moon
#
# Partially adpated from Latent ODEs for Irregularly-Sampled Time Series (Rubanova et al. 2019)
####################################################

import os
import logging
import pickle
import warnings
import random

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

from lifelines.utils import concordance_index
from lib.likelihood_eval import *
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc, integrated_brier_score
from sklearn.utils import check_consistent_length, check_array
from sklearn.impute import SimpleImputer
from sklearn import model_selection

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import itertools
import ipdb # for debugging

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger

def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def make_dataset(dataset_type = "spiral",**kwargs):
	if dataset_type == "spiral":
		data_path = "data/spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	elif dataset_type == "chiralspiral":
		data_path = "data/chiral-spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	else:
		raise Exception("Unknown dataset type " + dataset_type)
	return dataset, chiralities


def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res

def init_network_weights(net, std = 0.1, mode = None):
	for idx, m in enumerate(net.modules()):
		if isinstance(m, nn.Linear):
			if mode == 'Cox': # or mode == 'Softmax':
				nn.init.normal_(m.weight, mean=0, std=std)
			else:
				nn.init.normal_(m.weight, mean=0, std=std)
				nn.init.constant_(m.bias, val=0)


def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))

def subsample_timepoints(data, time_steps, mask, n_tp_to_sample = None):
	# n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
	if n_tp_to_sample is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)


	if n_tp_to_sample > 1:
		# Subsample exact number of points
		assert(n_tp_to_sample <= n_tp_in_batch)
		n_tp_to_sample = int(n_tp_to_sample)

		for i in range(data.size(0)):
			missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

			data[i, missing_idx] = 0.
			if mask is not None:
				mask[i, missing_idx] = 0.
	
	elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
		# Subsample percentage of points from each time series
		percentage_tp_to_sample = n_tp_to_sample
		for i in range(data.size(0)):
			# take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
			current_mask = mask[i].sum(-1).cpu()
			non_missing_tp = np.where(current_mask > 0)[0]
			n_tp_current = len(non_missing_tp)
			n_to_sample = int(n_tp_current * percentage_tp_to_sample)
			subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
			tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

			data[i, tp_to_set_to_zero] = 0.
			if mask is not None:
				mask[i, tp_to_set_to_zero] = 0.

	return data, time_steps, mask

def cut_out_timepoints(data, time_steps, mask, n_points_to_cut = None):
	# n_points_to_cut: number of consecutive time points to cut out
	if n_points_to_cut is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)

	if n_points_to_cut < 1:
		raise Exception("Number of time points to cut out must be > 1")

	assert(n_points_to_cut <= n_tp_in_batch)
	n_points_to_cut = int(n_points_to_cut)

	for i in range(data.size(0)):
		start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)

		data[i, start : (start + n_points_to_cut)] = 0.
		if mask is not None:
			mask[i, start : (start + n_points_to_cut)] = 0.

	return data, time_steps, mask

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma, n_latent_traj = None, random_seed = 0):
	device = get_device(mu)
	dist = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	if n_latent_traj > 1:
		torch.manual_seed(random_seed)
		r = dist.sample([n_latent_traj, mu.size()[1], mu.size()[2]]).squeeze(-1)
		return r * sigma.float().expand(n_latent_traj, sigma.size()[1], sigma.size()[2]) + mu.float().expand(n_latent_traj, mu.size()[1], mu.size()[2])
	else:
		torch.manual_seed(random_seed)
		r = dist.sample(mu.size()).squeeze(-1)
		return r * sigma.float() + mu.float()

def split_train_test(data, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]
	return data_train, data_test

def split_train_test_data_and_time(data, time_steps, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]

	assert(len(time_steps.size()) == 2)
	train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
	test_time_steps = time_steps[:, int(n_samples * train_fraq):]

	return data_train, data_test, train_time_steps, test_time_steps

def get_next_batch(dataloader):
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template() # create an empty template
	for key, vals in data_dict.items():
		batch_dict[key] = vals

	batch_dict['non_missing_tp'] = torch.sum(batch_dict["observed_data"],(0,2)) != 0.
	batch_dict['non_missing_tp_pred'] = torch.sum(batch_dict["data_to_predict"],(0,2)) != 0.
	return batch_dict

def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path, map_location = device)
	ckpt_args = checkpt['params_dic']
	ckpt_args['min_max_data_tuple'] = checkpt['min_max_data_tuple']
	ckpt_args['max_obs_time'] = checkpt['max_obs_time']
	ckpt_args['events_info_train_tuple'] = checkpt['events_info_train_tuple']
	if 'best_epoch' in checkpt.keys():
		ckpt_args['best_epoch'] = checkpt['best_epoch']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	model_dict.update(state_dict) 
	model.load_state_dict(state_dict)
	model.to(device)
	return ckpt_args


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def get_item_from_pickle(pickle_file, item_name):
	from_pickle = load_pickle(pickle_file)
	if item_name in from_pickle:
		return from_pickle[item_name]
	return None

def get_dict_template():
	return {"sample_ids": None,
			"observed_data": None,
			"observed_tp": None,
			"observed_tp_unnorm": None,
			"data_to_predict": None,
			"tp_to_predict": None,
			"tp_to_predict_unnorm": None,
			"pred_horizon_idx": None,
			"observed_mask": None,
			"mask_predicted_data": None,
			"mask_surv": None,
			"labels": None,
			"event_time_idx":None,
			"event_times":None,
			"end_of_obs_idx":None,
			"data_extra_info":None,
			"feat_names":None
			}

def normalize_data(data):
	reshaped = data.reshape(-1, data.size(-1))

	att_min = torch.min(reshaped, 0)[0]
	att_max = torch.max(reshaped, 0)[0]

	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / att_max
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm, att_min, att_max

def normalize_masked_data(data, mask, att_min, att_max, extra = False, scale_param = 1.0, device = None):
	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if extra:
		data_norm = []
		if (att_max != 0.).all():
			for data_per_sample, mask_per_sample in zip(data, mask):
				# breakpoint()
				try:
					data_norm_per_sample = (data_per_sample.to(device) - att_min.to(device))/(att_max.to(device) - att_min.to(device)) * scale_param #normalize_masked_data
				except:
					data_norm_per_sample = (data_per_sample.to(device) - att_min)/(att_max - att_min) * scale_param #normalize_masked_data
				# set masked out elements back to zero 
				# breakpoint()
				data_norm_per_sample[mask_per_sample == 0] = 0
				data_norm.append(data_norm_per_sample)
		else:
			raise Exception("Zero!")

		return data_norm, att_min, att_max
	else:
		if (att_max != 0.).all():
			try:
				data_norm = (data - att_min.to(device)) / (att_max.to(device) - att_min.to(device)) * scale_param #- att_min.to(device)
			except:
				data_norm = (data - torch.tensor(att_min).to(device)) / (torch.tensor(att_max).to(device) - torch.tensor(att_min).to(device)) * scale_param #- att_min.to(device)	
		else:
			raise Exception("Zero!")

		if torch.isnan(data_norm).any():
			raise Exception("nans in the normalized data; check your min and max data vectors")

		# set masked out elements back to zero 
		data_norm[mask == 0] = 0

		# get normalized event time with respect to time points
		# if event_times is not None:
		# 	event_times = (event_times - att_min[1])/att_max[1] # wrt time points
		# 	return data_norm, event_times, att_min, att_max
		# else:
		return data_norm, att_min, att_max

def shift_outputs(outputs, first_datapoint = None):
	outputs = outputs[:,:,:-1,:]

	if first_datapoint is not None:
		n_traj, n_dims = first_datapoint.size()
		first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
		outputs = torch.cat((first_datapoint, outputs), 2)
	return outputs

def split_data_extrap(data_dict, dataset = ""):
	device = get_device(data_dict["data"])

	n_observed_tp = data_dict["data"].size(1) // 2
	if dataset == "hopper":
		n_observed_tp = data_dict["data"].size(1) // 3

	split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
				"observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
				"data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
				"tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone()}

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if ("mask" in data_dict) and (data_dict["mask"] is not None):
		split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		split_dict["labels"] = data_dict["labels"].clone()

	split_dict["mode"] = "extrap"
	return split_dict

def add_mask(data_dict):
	data = data_dict["observed_data"]
	mask = data_dict["observed_mask"]

	if mask is None:
		mask = torch.ones_like(data).to(get_device(data))

	data_dict["observed_mask"] = mask
	return data_dict

def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
	# n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
	# n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points

	if n_tp_to_sample is not None:
		# Randomly subsample time points
		data, time_steps, mask = subsample_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_tp_to_sample = n_tp_to_sample)

	if n_points_to_cut is not None:
		# Remove consecutive time points
		data, time_steps, mask = cut_out_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_points_to_cut = n_points_to_cut)

	new_data_dict = {}
	for key in data_dict.keys():
		new_data_dict[key] = data_dict[key]

	new_data_dict["observed_data"] = data.clone()
	new_data_dict["observed_tp"] = time_steps.clone()
	new_data_dict["observed_mask"] = mask.clone()

	if n_points_to_cut is not None:
		# Cut the section in the data to predict as well
		# Used only for the demo on the periodic function
		new_data_dict["data_to_predict"] = data.clone()
		new_data_dict["tp_to_predict"] = time_steps.clone()
		new_data_dict["mask_predicted_data"] = mask.clone()

	return new_data_dict

def compute_loss_all_batches(model, train_dataloader, valid_dataloader, params_dic, n_batches_train, device,
	n_latent_traj = 1, kl_coef = 1., 
	max_samples_for_eval = None, plot_survival_curves = True, itr = None, 
	plot_concord_ibs_across_epoch = True, filename_suffix = None,
	surv_est = None, dataset = None, bootstrap = False, feat_names = None, max_pred_window = None, min_max_data_tuple = None, survival_loss_scale = 10, optimizer = None, n_events = 1, valid_batch_size = 200):

	curr_epoch = itr // n_batches_train
	# print('curr_epoch : ', curr_epoch)
	batch_dict_train = get_next_batch(train_dataloader)

	tp_res = 1
	min_event_time = 1
	max_event_time = max_pred_window if max_pred_window is not None else 750 # 1000 hours

	### TODO : provide global non_missing_tp and non_missing_tp_pred for entire validation set

	## split validation similar to how you would do in Get Surv!
	batch_dict = remove_timepoints_wo_obs(get_next_batch(valid_dataloader)) 

	batch_total_observed_data = divide_list(batch_dict["observed_data"], valid_batch_size)
	batch_total_observed_mask = divide_list(batch_dict["observed_mask"], valid_batch_size)
	batch_total_end_obs_idx = divide_list(batch_dict["end_of_obs_idx"], valid_batch_size)

	for i in tqdm(range(len(batch_total_observed_data)), desc = 'Loading validation set...'):
		validation = True 
		

		pred_y_mult_traj, hazards_y_mult_traj, info = model.get_reconstruction_survival(batch_dict["tp_to_predict"], 
																					   batch_total_observed_data[i], batch_dict["observed_tp"], batch_total_end_obs_idx[i], 
																					   mask = batch_total_observed_mask[i], n_latent_traj = n_latent_traj)
		# merge batch dict and result dict
		if i > 0:
			# concatenate pred_y_mult_traj, hazards_y_mult_traj, info
			pred_y_mult_traj = torch.cat((prev_pred_y_mult_traj, pred_y_mult_traj), 1)
			if n_events == 1:
				hazards_y_mult_traj = torch.cat((prev_hazards_y_mult_traj, hazards_y_mult_traj), 1)
			else:
				hazards_y_mult_traj = torch.cat((prev_hazards_y_mult_traj, hazards_y_mult_traj), 1)
			for key, data in info.items():
				if key == 'first_point':
					info[key] = (torch.cat((prev_info[key][0], data[0]), 1), torch.cat((prev_info[key][1], data[1]), 1), torch.cat((prev_info[key][2], data[2]), 1))
				elif key == 'latent_hazard' or key == 'first_point_ext':
					pass
				else:
					info[key] = torch.cat((prev_info[key], data), 1)

		prev_batch_dict = batch_dict.copy()

		prev_pred_y_mult_traj = pred_y_mult_traj.detach()
		prev_info = info.copy()

		if n_events == 1:
			prev_hazards_y_mult_traj = hazards_y_mult_traj.detach()
		else:
			prev_hazards_y_mult_traj = hazards_y_mult_traj.detach()

	reconstr_info = (pred_y_mult_traj, hazards_y_mult_traj, info)
	results = model.compute_all_losses(batch_dict, n_latent_traj = n_latent_traj, kl_coef = kl_coef, surv_est = surv_est, survival_loss_scale = survival_loss_scale, reconstr_info = reconstr_info)
	reconstr_loss = float(results["likelihood"].cpu().numpy()); survival_loss = float(results["survival_loss"].cpu().numpy())
	print('\n')
	print('============ Validation set performance ============')
	print('survival log-likelihood : ', np.round(survival_loss,4))
	print('reconstr. likelihood : ', np.round(reconstr_loss, 4))
	remaining_time_to_event, _, perf_dic, quantile_perf_dic = get_performance_results(model, results, batch_dict_train, batch_dict, curr_epoch = curr_epoch, dataset = dataset, surv_est = surv_est, bootstrap = bootstrap, filename_suffix = filename_suffix, event_time_horizon = (min_event_time, max_event_time, tp_res), plot_survival_curves = plot_survival_curves, validation = validation, feat_names = feat_names, n_events = n_events)
	print('====================================================')
	print('\n')
	
	if plot_concord_ibs_across_epoch:
		# load previous records 
		model_performance_total = []
		for event_idx in range(n_events):
			if n_events > 1:
				c_idx, ibs, auc, mean_auc = perf_dic[event_idx]['mean_c_idx'], perf_dic[event_idx]['ibs'], perf_dic[event_idx]['auc'], perf_dic[event_idx]['mean_auc'] #quantile_perf_dic['point_ests']['mean_auc']
			else:
				c_idx, ibs, auc, mean_auc = perf_dic['c_idx'], perf_dic['ibs'], perf_dic['auc'], perf_dic['mean_auc'] #quantile_perf_dic['point_ests']['mean_auc']

			try:
				with open('model_performance/' + filename_suffix + '/' + 'model_performance_' + str(event_idx) + '_' + filename_suffix + '.npy', 'rb') as f:
					model_performance = np.load(f)
				if bootstrap: # once bootstrap set to true, ibs and auc are dic
					model_performance = [np.append(model_performance[0], c_idx), np.append(model_performance[1], ibs), np.append(model_performance[2], auc), np.append(model_performance[3], reconstr_loss), np.append(model_performance[4], survival_loss)] # concodrance, ibs
				else:
					model_performance = [np.append(model_performance[0], c_idx), np.append(model_performance[1], ibs), np.append(model_performance[2], mean_auc), np.append(model_performance[3], reconstr_loss), np.append(model_performance[4], survival_loss)] # concodrance, ibs
				# export model performance as df
				df_result = pd.DataFrame(model_performance, index = ['c_idx', 'ibs', 'mean_auc', 'reconstr_loss', 'survival_loss'])
				df_result.to_csv('model_performance/' + filename_suffix + '/' + 'df_model_performance_'  + str(event_idx) + '_' + filename_suffix + '.csv')
			except:
				model_performance = [[], [], [], [], []]
				model_performance[0].append(c_idx) # concodrance
				model_performance[3].append(reconstr_loss)
				model_performance[4].append(survival_loss)
				if bootstrap: 
					model_performance[1].append(ibs) # ibs
					model_performance[2].append(auc) # mean auc
				else:
					model_performance[1].append(ibs) # ibs
					model_performance[2].append(mean_auc) # mean auc
			# locally store current performance
			with open('model_performance/' + filename_suffix + '/' + 'model_performance_' + str(event_idx) + '_' + filename_suffix + '.npy', 'wb') as f:
				np.save(f, model_performance)
			# plot_performance(model_performance, surv_est = surv_est, bootstrap = bootstrap, filename_suffix = filename_suffix, event_idx = event_idx)

			model_performance_total.append(model_performance)

			# store quantile test dict as well
			try:
				with open('model_performance/' + filename_suffix + '/' + 'model_performance_quantile.npy', 'rb') as f:
					model_performance_quantile = np.load(f)
				model_performance_quantile.append(quantile_perf_dic)
			except:
				model_performance_quantile = []
				model_performance_quantile.append(quantile_perf_dic)

			with open('model_performance/' + filename_suffix + '/' + 'model_performance_quantile.npy', 'wb') as f:
				np.save(f, model_performance_quantile)

		model_performance_oi = model_performance_total[0] # choose the primary event of interest
		best_auc_index = np.argmax(model_performance_oi[2])
		print('Best iteration so far wrt. mean AUC : ', best_auc_index + 1)
		# store the latest model for a general checkpoint
		print('Storing the latest model...')
		path = 'model_performance/' + filename_suffix + '/latest_model.pt'
		print(path)

		events_info_train_tuple = (batch_dict_train['event_times'], batch_dict_train['labels'], batch_dict_train['remaining_time_to_event'], batch_dict_train['end_of_obs_idx'])
		torch.save({'params_dic': params_dic, 'max_obs_time' : batch_dict_train['max_obs_time'], 'min_max_data_tuple':min_max_data_tuple, 'events_info_train_tuple' : events_info_train_tuple, 'state_dict': model.state_dict(), 'itr' : itr, 'optimizer_state_dict': optimizer.state_dict()}, path)
		print('\n')
		if best_auc_index + 1 == curr_epoch and curr_epoch >= 3: # start saving the best performance model after 3 epochs
			# print('Plotting reconst traj from the best model...')
			# model.get_reconstruction_traj(results, batch_dict, filename_suffix = filename_suffix, curr_epoch = curr_epoch, feat_names = feat_names, min_event_time = min_event_time)
			print('Storing the best model...')
			path = 'model_performance/' + filename_suffix + '/best_model.pt'
			events_info_train_tuple = (batch_dict_train['event_times'], batch_dict_train['labels'], batch_dict_train['remaining_time_to_event'], batch_dict_train['end_of_obs_idx'])
			torch.save({'params_dic': params_dic, 'max_obs_time' : batch_dict_train['max_obs_time'], 'min_max_data_tuple':min_max_data_tuple, 'events_info_train_tuple' : events_info_train_tuple, 'state_dict': model.state_dict(), 'best_epoch' : curr_epoch}, path)

	return model_performance_oi

def check_mask(data, mask):
	#check that "mask" argument indeed contains a mask for data
	n_zeros = torch.sum(mask == 0.).cpu().numpy()
	n_ones = torch.sum(mask == 1.).cpu().numpy()

	# mask should contain only zeros and ones
	assert((n_zeros + n_ones) == np.prod(list(mask.size())))

	# all masked out elements should be zeros
	assert(torch.sum(data[mask == 0.] != 0.) == 0)

def perform_bootstrap_quantile(test_stat, quant_to_event_oi_train_test_surv_dict, boot_iter = 5000, alpha = 0.05, ef_surv = False, max_pred_window = None, cidx_bootstrap = False):
	"""
	Obtain confidence interval of median via bootstrap
	"""
	sampled_stats_dic = {}; sampled_ses_dic = {}; delta_conf_int_dic = {}
	if ef_surv:
		sampled_stats_dic['bs'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		sampled_ses_dic['bs'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
	else:
		sampled_stats_dic['auc'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		sampled_ses_dic['auc'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		delta_conf_int_dic['auc'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		sampled_stats_dic['bs'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		sampled_ses_dic['bs'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		# sampled_stats_dic['c_idx'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)
		# sampled_ses_dic['c_idx'] = create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict)

	skipped_iter = 0
	for i in tqdm(range(boot_iter), desc = 'Bootstrapping...'):
		# for each quantile
		for quant, (event_oi_train, event_oi, surv_metric_oi, quant_time, surv_metric_oi_cum, last_observed_points_oi) in quant_to_event_oi_train_test_surv_dict.items():
			resampled_indicies = np.random.choice(np.arange(len(event_oi)), len(event_oi), replace = True)

			event_oi_sampled = event_oi[resampled_indicies]
			metric_oi_sampled = surv_metric_oi[resampled_indicies]
			if not ef_surv:
				last_observed_points_oi_sampled = last_observed_points_oi[resampled_indicies]
			if ef_surv:
				try:
					bs = brier_score(event_oi_train, event_oi_sampled, metric_oi_sampled, quant_time)[1][0]
				except:
					print(f"Invalid time window. Skipping the current bootstrap iteration: {i}")
					skipped_iter += 1
					continue
				sampled_stats_dic['bs'][quant].append(bs)
			else:
				try:
					auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi_sampled, metric_oi_sampled, quant_time)
				except:
					print(f"Invalid time window. Skipping the current bootstrap iteration: {i}")
					skipped_iter += 1
					continue
				sampled_stats_dic['auc'][quant].append(auc[0])
				bs = brier_score(event_oi_train, event_oi_sampled, 1-metric_oi_sampled, quant_time)[1][0]
				sampled_stats_dic['bs'][quant].append(bs)
				# breakpoint()
	for metric, sub_dict in sampled_stats_dic.items():
		for quant, metric_vals in sub_dict.items():
			sampled_ses_dic[metric][quant] = np.round(math.sqrt(1/(boot_iter-skipped_iter) * sum((np.asarray(metric_vals) - np.mean(metric_vals))**2)), 4)
	return sampled_ses_dic

def perform_bootstrap_v2(test_stat, event_oi_train, event_oi, surv_metric_oi, surv_metric_oi_mean, test_quantile_times, boot_iter = 5000, alpha = 0.05):
	"""
	Obtain confidence interval of median via bootstrap
	"""

	# random_seed = 1991 # orig random seed : 1991
	# np.random.seed(random_seed)
	# breakpoint()
	
	sampled_stats_dic = {}; sampled_ses_dic = {}; delta_conf_int_dic = {}
	# quantiles_dict = {}; 
	# for quant, (_, _, _, _, _, _) in quant_to_event_oi_train_test_surv_dict.items():
	#   quantiles_dict[quant] = []
	sampled_stats = []
	# if mode == 'quantiles':
	sampled_stats_dic = {'mean_auc' : [], 'ibs' : [], 'auc' : {25 : [], 50 : [], 75 : []}, 'bs' : {25 : [], 50 : [], 75 : []}, 'c_idx' : {25 : [], 50 : [], 75 : []}}

	# breakpoint()
	for i in tqdm(range(boot_iter), desc = 'Bootstrapping...'):
		# for each quantile
		# for quant, (event_oi_train, event_oi, surv_metric_oi, quant_time, surv_metric_oi_cum, last_observed_points_oi) in quant_to_event_oi_train_test_surv_dict.items():
		np.random.seed(i)
		resampled_samples = np.random.choice(np.arange(len(event_oi)), len(event_oi), replace = True)

		event_oi_sampled = event_oi[resampled_samples]
		metric_oi_sampled = surv_metric_oi[resampled_samples]
		metric_oi_sampled_mean = surv_metric_oi_mean[resampled_samples]

		bs = brier_score(event_oi_train, event_oi_sampled, metric_oi_sampled, test_quantile_times)[1]
		ibs = integrated_brier_score(event_oi_train, event_oi_sampled, metric_oi_sampled_mean, np.arange(test_quantile_times[0], test_quantile_times[-1]))
		auc, _ = cumulative_dynamic_auc(event_oi_train, event_oi_sampled, 1-metric_oi_sampled, test_quantile_times)
		_, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi_sampled, 1-metric_oi_sampled_mean, np.arange(test_quantile_times[0], test_quantile_times[-1]))

		# breakpoint()
		for bs_, auc_, quantile in zip(bs, auc, [25, 50, 75]):
			sampled_stats_dic['auc'][quantile].append(auc_)
			sampled_stats_dic['bs'][quantile].append(bs_)
		sampled_stats_dic['mean_auc'].append(mean_auc)
		sampled_stats_dic['ibs'].append(ibs)
		# sampled_stats_dic['auc'][quant].append(auc[0])

		# print(sampled_stats_dic['auc'][quant])
		# print('len(sampled_stats_dic) : ', len(sampled_stats_dic['auc'][quant]))
		# breakpoint()

	sampled_ses_dic = {'auc' : {25 : 0.0, 50 : 0.0, 75 : 0.0}, 'bs' : {25 : 0.0, 50 : 0.0, 75 : 0.0}, 'c_idx' : {25 : 0.0, 50 : 0.0, 75 : 0.0}}
	delta_conf_int_dic = {'auc' : {25 : (), 50 : (), 75 : ()}, 'bs' : {25 : (), 50 : (), 75 : ()}, 'c_idx' : {25 : (), 50 : (), 75 : ()}}
	for quantile in [25, 50, 75]:
		sampled_ses_dic['auc'][quantile] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['auc'][quantile]) - np.mean(sampled_stats_dic['auc'][quantile]))**2))
		sampled_ses_dic['bs'][quantile] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['bs'][quantile]) - np.mean(sampled_stats_dic['bs'][quantile]))**2))
		# sampled_ses_dic['c_idx'][quantile] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_ses_dic['c_idx'][quantile]) - np.mean(sampled_ses_dic['c_idx'][quantile]))**2))
		for metric_oi in ['auc', 'bs']:
			lower_bound = 2*test_stat[metric_oi][quantile] - np.percentile(sampled_stats_dic[metric_oi][quantile], 100*(1-alpha/2))# prevent negative survival time
			upper_bound = 2*test_stat[metric_oi][quantile] - np.percentile(sampled_stats_dic[metric_oi][quantile], 100*alpha/2)
			delta_conf_int = (np.round(lower_bound,6), np.round(upper_bound,6))
			delta_conf_int_dic[metric_oi][quantile] = delta_conf_int

	# for ibs and auc_mean
	sampled_ses_dic['mean_auc'] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['mean_auc']) - np.mean(sampled_stats_dic['mean_auc']))**2))
	sampled_ses_dic['ibs'] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['ibs']) - np.mean(sampled_stats_dic['ibs']))**2))
	
	# create bootstrap info df and add it to sampled_ses_dic
	df_bootstrap_records = pd.DataFrame([], index = np.arange(boot_iter), columns = ['auc_25', 'auc_50', 'auc_75', 'bs_25', 'bs_50', 'bs_75', 'ibs', 'mean_auc'])
	for metric, vals in sampled_stats_dic.items():
		if metric in ['mean_auc', 'ibs']:
			df_bootstrap_records[metric] = vals
		elif metric != 'c_idx':
			for quantile, subvals in vals.items():
				# if :
				df_bootstrap_records[metric + '_' + str(quantile)] = subvals
	# breakpoint()
	sampled_ses_dic['df_bootstrap_records'] = df_bootstrap_records
	return sampled_ses_dic, delta_conf_int_dic

def perform_bootstrap(test_stat, len_samples_oi, func = None, func_args = None, colname = None, boot_iter = 10000, alpha = 0.05, mode = None):
	
	times_orig = func_args['times'].copy()
	estimate_orig = func_args['estimate'].copy()
	survival_test_orig = func_args['survival_test'].copy()

	sampled_stats = []
	if mode == 'quantiles':
		sampled_stats_dic = {'auc' : {25 : [], 50 : [], 75 : []}, 'bs' : {25 : [], 50 : [], 75 : []}, 'c_idx' : {25 : [], 50 : [], 75 : []}}

	for i in tqdm(range(boot_iter), desc = 'Bootstrapping...'):
		resampled_samples = np.random.choice(np.arange(len_samples_oi), len_samples_oi, replace = True)
		if type(colname) == tuple:
			for col in colname:
				# breakpoint()
				func_args[col] = func_args[col][resampled_samples]
		else:
			func_args[colname] = func_args[colname][resampled_samples]
		# adjust prediction window based on test dataset
		if not mode == 'quantiles':
			times_oi = func_args['times']
			tp_res = times_oi[1] - times_oi[0]
			min_event_time = func_args['survival_test']['time'].min()
			min_event_time = times_oi.min() * (min_event_time < times_oi.min()) + min_event_time * (min_event_time >= times_oi.min())

			max_event_time = func_args['survival_test']['time'].max()
			max_event_time = times_oi.max() * (max_event_time >= times_oi.max()) + max_event_time * (max_event_time < times_oi.max())

			func_args['times'] = np.arange(min_event_time, max_event_time, tp_res)
			times_oi_est = (func_args['times']/tp_res).astype(int)

			func_args['estimate'] = func_args['estimate'][:,times_oi_est]
		try: # in case of all samples being censored
			if mode == 'auc':
				sampled_stats.append(func(**func_args)[1])
			elif mode == 'ibs':
				sampled_stats.append(func(**func_args))
			elif mode == 'quantiles':
				bs = brier_score(**func_args)[1]
				func_args['estimate'] = func_args['estimate'] * -1
				auc = cumulative_dynamic_auc(**func_args)[0]

				for bs_, auc_, quantile in zip(bs, auc, [25, 50, 75]):
					sampled_stats_dic['auc'][quantile].append(auc_)
					sampled_stats_dic['bs'][quantile].append(bs_)
		except:
			print('bootstrap error: skipping iteration...')
			pass

		func_args['times'] = times_orig
		func_args['estimate'] = estimate_orig
		func_args['survival_test'] = survival_test_orig

	if mode == 'quantiles':
		sampled_ses_dic = {'auc' : {25 : 0.0, 50 : 0.0, 75 : 0.0}, 'bs' : {25 : 0.0, 50 : 0.0, 75 : 0.0}, 'c_idx' : {25 : 0.0, 50 : 0.0, 75 : 0.0}}
		delta_conf_int_dic = {'auc' : {25 : (), 50 : (), 75 : ()}, 'bs' : {25 : (), 50 : (), 75 : ()}, 'c_idx' : {25 : (), 50 : (), 75 : ()}}
		for quantile in [25, 50, 75]:
			sampled_ses_dic['auc'][quantile] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['auc'][quantile]) - np.mean(sampled_stats_dic['auc'][quantile]))**2))
			sampled_ses_dic['bs'][quantile] = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats_dic['bs'][quantile]) - np.mean(sampled_stats_dic['bs'][quantile]))**2))
			for metric_oi in ['auc', 'bs']:
				lower_bound = 2*test_stat[metric_oi][quantile] - np.percentile(sampled_stats_dic[metric_oi][quantile], 100*(1-alpha/2))# prevent negative survival time
				upper_bound = 2*test_stat[metric_oi][quantile] - np.percentile(sampled_stats_dic[metric_oi][quantile], 100*alpha/2)
				delta_conf_int = (np.round(lower_bound,6), np.round(upper_bound,6))
				delta_conf_int_dic[metric_oi][quantile] = delta_conf_int
		return sampled_ses_dic, delta_conf_int_dic
	else:
		sampled_se = math.sqrt(1/boot_iter * sum((np.asarray(sampled_stats) - np.mean(sampled_stats))**2))
		lower_bound = 2*test_stat - np.percentile(sampled_stats, 100*(1-alpha/2))# prevent negative survival time
		upper_bound = 2*test_stat - np.percentile(sampled_stats, 100*alpha/2)
		delta_conf_int = (np.round(lower_bound,6), np.round(upper_bound,6))
		return sampled_se, delta_conf_int

def compute_dynamic_auc(surv_prob, data_dict_train, data_dict_test, eval_info_dic_train, eval_info_dic_train_test, event_time_horizon = None):
	"""
	Compute AUC in real-time
	Censor patients once their next observatons become available
	"""
	min_event_time = event_time_horizon[0]
	max_event_time = event_time_horizon[1]
	tp_res = event_time_horizon[2]
	perf_per_round_dic = {}
	for round_, surv_tuple_ in surv_prob.items():
		eligible_indices = surv_tuple_[0]
		eligible_indices_train = [idx for idx, val in enumerate(eval_info_dic_train['observed_tp_per_round'].T[round_]) if not np.isnan(val)]
		# test
		if round_ < len(surv_prob) - 1:
			eligible_indices_next = surv_prob[round_ + 1][0]
		else:
			eligible_indices_next = []
		# train; this could only work if # of rounds for test == # of rounds for train
		if round_ < len(surv_prob) - 1:
			eligible_indices_next_train = [val for val in eval_info_dic_train['observed_tp_per_round'].T[round_ + 1] if not np.isnan(val)]
		else:
			eligible_indices_next_train = []

		surv_prob_ = surv_tuple_[1]
		event_oi = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=len(eligible_indices))
		event_oi_train = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=len(eligible_indices_train))

		# popluate structured arrays for test data
		event_ind_list = []; event_time_list = []
		for idx_, event_ind_ in enumerate(data_dict_test['labels'][eligible_indices]):
			if idx_ in eligible_indices_next:
				event_ind_list.append(False)
			else:	
				event_ind_list.append(bool(event_ind_ == 1))				

		event_oi['event'] = event_ind_list
		event_oi['time'] = [float(event_time_) for event_time_ in eval_info_dic_train_test['event_time_per_round'].T[round_][eligible_indices]]
		print('\n')
		print('Round : ', round_)
		# popluate structured arrays for train data
		event_ind_list = []; event_time_list = []
		for idx_, event_ind_ in enumerate(data_dict_train['labels'][eligible_indices_train]):
			if idx_ in eligible_indices_next_train:
				event_ind_list.append(False)
			else:	
				event_ind_list.append(bool(event_ind_ == 1))				

		event_oi_train['event'] = event_ind_list
		event_oi_train['time'] = [float(event_time_) for event_time_ in eval_info_dic_train['event_time_per_round'].T[round_][eligible_indices_train]]
		
		print('training set')
		print('num events : ', np.sum(event_ind_list))
		print('num samples : ', len(eligible_indices_train))
		print('test set')
		print('num events : ', np.sum(event_ind_list))
		print('num samples : ', len(eligible_indices))
		times_oi = np.arange(min_event_time, max_event_time, tp_res)
		surv_prob_oi = -1 * surv_prob_[:, times_oi]
		auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, surv_prob_oi, times_oi)
		perf_per_round_dic[(round_, 'auc')] = (auc, mean_auc)
		print('mean AUC from round ' + str(round_) + ' : ', np.round(mean_auc, 3))

		# get brier scores as well
		ibs = integrated_brier_score(event_oi_train, event_oi, -1*surv_prob_oi, times_oi)
		times, bs = brier_score(event_oi_train, event_oi, -1*surv_prob_oi, times_oi)

		assert (times == times_oi).any(), "BS-specific times different from event horizon"
		perf_per_round_dic[(round_, 'ibs')] = (bs, ibs)
		print('IBS from round ' + str(round_) + ' : ', np.round(ibs, 3))
			# print('\n')
	return perf_per_round_dic

def get_remaining_time_to_event(data_dict, num_samples, torch_ver = False):
	if torch_ver:
		remaining_time_to_event = torch.zeros(num_samples)#.to(device)
		for j in range(num_samples):
			last_observed_point = data_dict['observed_tp_unnorm'][data_dict['observed_mask'][j].sum(axis = 1) > 0][-1]
			remaining_time_to_event[j] = data_dict['event_times'][j] - last_observed_point
	else:
		remaining_time_to_event = []
		for j in range(num_samples):
			last_observed_point = data_dict['observed_tp_unnorm'][data_dict['observed_mask'][j].sum(axis = 1) > 0][-1]
			remaining_time_to_event.append(data_dict['event_times'][j] - last_observed_point)
	# breakpoint()
	return remaining_time_to_event

def compute_auc_from_last_obs(surv_prob, data_train_tuple, data_test_tuple, event_time_horizon = None, bootstrap = False):
	"""
	Compute AUC from last observation points
	data_train_tuple[0] --> label
	data_train_tuple[1] --> remaining_time_to_event
	"""
	min_event_time = event_time_horizon[0]
	max_event_time = event_time_horizon[1]
	tp_res = event_time_horizon[2]

	# test set
	num_samples = len(data_test_tuple[0])
	event_oi = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=num_samples)
	remaining_time_to_event = data_test_tuple[1] #get_remaining_time_to_event(data_dict_test, num_samples)
	# breakpoint()

	# event indicator
	event_oi['event'] = [bool(event_ind_ == 1) for event_ind_ in data_test_tuple[0]]
	event_oi['time'] = [float(event_time_) for event_time_ in remaining_time_to_event]

	# train set 
	num_samples = len(data_train_tuple[0])
	event_oi_train = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=num_samples)
	remaining_time_to_event = data_train_tuple[1]

	event_oi_train['event'] = [bool(event_ind_ == 1) for event_ind_ in data_train_tuple[0]]
	event_oi_train['time'] = [float(event_time_) for event_time_ in remaining_time_to_event]

	# get AUC
	# up to 90% quantile
	times_oi = np.arange(min_event_time, int(np.quantile(event_oi['time'], q = 0.9)), tp_res)
	surv_prob_oi = 1 - surv_prob
	# get mean AUC from 2 days to up until 90% of test survival times 
	times_oi = np.arange(min_event_time, np.quantile(event_oi['time'], q = 0.9), tp_res) # times oi has to be within the range of follow-up time in test set
	# breakpoint()
	surv_prob_oi = surv_prob_oi[:, (times_oi).astype(int)]
	# breakpoint()
	"""
	TODO : 
	1. Get eligible samples for each time quantile
		- Here, eligible means their observations end before the qunatile time
	2. Perform evauation based on those eligible samples across quantiles
		- 15%, 25%, 50%, 75%, 85% 
	3. Save the dictionary (mapping between quantile and eligible samples) locally
	"""

	auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, surv_prob_oi, times_oi)
	if bootstrap:
		perf_dic = {}
		sampled_se, conf_int_mean_auc = perform_bootstrap(mean_auc, len(surv_prob_oi), mode = 'auc', colname = ('estimate', 'survival_test'), func = cumulative_dynamic_auc, func_args = {'survival_train' : event_oi_train, 'survival_test' : event_oi, 'estimate' : surv_prob, 'times' : times_oi})
		perf_dic['auc'] = auc
		perf_dic['mean_auc'] = mean_auc
		perf_dic['auc_se'] = sampled_se
		perf_dic['auc_conf_int'] = conf_int_mean_auc
		return auc, perf_dic
	else:
		return auc, mean_auc

def compute_auc(surv_prob, data_train_tuple, data_test_tuple, eval_info_dic_train, eval_info_dic_train_test, cif = False, event_time_horizon = None, filename_suffix = None, curr_epoch = None, plot = False, bootstrap = False):
	"""
	Compute dynamic AUC 
	See https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.metrics.cumulative_dynamic_auc.html#sksurv.metrics.cumulative_dynamic_auc for details

	"""
	# if real_time_eval:
	# 	# start - stop time for each round
	# 	# where x is some pre-defined starting date
	# 	# 1st round : x days - censor them at the next observation (4000 days)
	# 	# 2nd round : x days - censor them at the next observation (4000 days)
	# 	# 3rd round : x days to 4,000 days
	# 	perf_per_round_dic = compute_dynamic_auc(surv_prob, data_dict_train, data_dict_test, eval_info_dic_train, eval_info_dic_train_test, event_time_horizon = event_time_horizon)
	# 	plot_perf_time(None, None, event_time_horizon = event_time_horizon, filename_suffix = filename_suffix, curr_epoch = curr_epoch, perf_per_round_dic = perf_per_round_dic)
	# 	return perf_per_round_dic
	# else:
	auc, mean_auc = compute_auc_from_last_obs(surv_prob, data_train_tuple, data_test_tuple, event_time_horizon = event_time_horizon, bootstrap = bootstrap)
	if plot:
		plot_perf_time(auc, mean_auc, metric = 'auc', event_time_horizon = event_time_horizon, filename_suffix = filename_suffix, curr_epoch = curr_epoch, bootstrap = bootstrap)
		
	return auc, mean_auc

def plot_perf_time(estimate, mean_est, metric = None, event_time_horizon = None, time_spec = None, perf_per_round_dic = None, filename_suffix = None, curr_epoch = None, bootstrap = False, tag = ''):
	"""
	Plot performance 
	If time_spec is provided, we don't use event time horizon
	"""
	min_event_time = event_time_horizon[0]
	max_event_time = event_time_horizon[1]
	tp_res = event_time_horizon[2]

	if time_spec is not None:
		t = time_spec
		t_orig = np.arange(min_event_time, max_event_time, tp_res)
	else:
		t = np.arange(min_event_time, min_event_time + len(estimate), tp_res) # for 90% quantile surv

	if metric == 'auc':
		label = 'AUC from the last observation'
		label_mean = 'Mean AUC'
		baseline_est = 'auc'
		baseline_mean_est = 'mean_auc' 
		save_file_name = "auc_from_last_obs_" + str(curr_epoch) + '_' + filename_suffix + '_' + tag # + ".pdf"
	elif metric == 'ibs':
		label = 'Brier Score from the last observation'
		label_mean = 'Integrated BS'
		baseline_est = 'bs'
		baseline_mean_est = 'ibs'
		save_file_name = "brier_score_from_last_obs_" + str(curr_epoch) + '_' + filename_suffix + '_' + tag # + ".pdf"
	if perf_per_round_dic is None:
		fig, ax = plt.subplots()
		ax.plot(t, estimate, color = 'blue', label = label, alpha = 0.5)
		if metric == 'auc':
			ax.set_ylim([0, 1])
			if bootstrap:
				ax.hlines(mean_est['mean_auc'], min_event_time, max_event_time, color = 'blue', linestyles = 'dashed', label = label_mean)
			else:
				ax.hlines(mean_est, min_event_time, max_event_time, color = 'blue', linestyles = 'dashed', label = label_mean)
		elif metric == 'ibs':
			ax.hlines(mean_est, min_event_time, max_event_time, color = 'blue', linestyles = 'dashed', label = label_mean)
		# plot baseline model
		# if baseline_perf is not None:
		# 	cox_perf = baseline_perf['cox_time_varying'] 
		# 	if time_spec is not None:
		# 		ax.plot(t_orig, cox_perf[baseline_est], color = 'red', label = label, alpha = 0.5)
		# 	else:
		# 		ax.plot(t, cox_perf[baseline_est], color = 'red', label = label, alpha = 0.5)
		# 	ax.hlines(cox_perf[baseline_mean_est], min_event_time, max_event_time, color = 'red', linestyles = 'dashed', label = label_mean)

		ax.set(xlabel='time (days)', ylabel='AUC')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol = 2)
		fig.savefig("model_performance/" + filename_suffix + "/" + save_file_name + ".pdf", bbox_inches='tight')
		result_df = pd.DataFrame([], columns = ['time', metric])
		result_df['time'] = t; result_df[metric] = estimate
		result_df.to_csv("model_performance/" + filename_suffix + "/df_" + save_file_name + ".csv")
		plt.close()
	else:
		# real time evaluation
		# plot auc across time
		perf_dic_real_time = {}
		fig, ax = plt.subplots()
		idx = 0; colors = ['blue', 'red', 'black']
		for round_key_, (est_, mean_est_) in perf_per_round_dic.items():
			if 'auc' in round_key_[1]:
				round_ = round_key_[0]
				result_df = pd.DataFrame([], columns = ['time', 'auc'])
				result_df['time'] = t; result_df['auc'] = est_
				perf_dic_real_time[round_key_] = result_df
				ax.plot(t, est_, color = colors[idx], label = 'AUC (round ' + str(round_) + ')', alpha = 0.5)
				ax.hlines(mean_est_, min_event_time, max_event_time, color = colors[idx], linestyles = 'dashed', label = 'Mean AUC (round ' + str(round_) + ')')
				idx += 1
		ax.set_ylim([0, 1])
		ax.set(xlabel='time (days)', ylabel='AUC')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol = 3)
		fig.savefig("model_performance/" + filename_suffix + "/auc_real_time_" + str(curr_epoch) + '_' + filename_suffix + ".pdf", bbox_inches='tight')
		plt.close()

		# plot brier score across time
		fig, ax = plt.subplots()
		idx = 0; colors = ['blue', 'red', 'black']
		for round_key_, (est_, mean_est_) in perf_per_round_dic.items():
			if 'ibs' in round_key_[1]:
				round_ = round_key_[0]
				result_df = pd.DataFrame([], columns = ['time', 'bs'])
				result_df['time'] = t; result_df['bs'] = est_
				perf_dic_real_time[round_key_] = result_df
				ax.plot(t, est_, color = colors[idx], label = 'Brier Score (round ' + str(round_) + ')', alpha = 0.5)
				ax.hlines(mean_est_, min_event_time, max_event_time, color = colors[idx], linestyles = 'dashed', label = 'IBS (round ' + str(round_) + ')')
				idx += 1
		# ax.set_ylim([0, 1])
		ax.set(xlabel='time (days)', ylabel='Brier Score')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol = 3)
		fig.savefig("model_performance/" + filename_suffix + "/brier_score_real_time_" + str(curr_epoch) + '_' + filename_suffix + ".pdf", bbox_inches='tight')			
		plt.close()

		# export real-time dictionary :
		# with open('model_performance/perf_dic_real_time_' + filename_suffix + '.npy', 'wb') as f:
		# 	np.save(f, model_performance)

		f = open("model_performance/" + filename_suffix + "/perf_dic_real_time" + str(curr_epoch) + '_' + filename_suffix + ".pkl", "wb") # prev : cup_sig_feats_dic_Feb_23rd_2021_binary_wo_duplicates_ph_check, cup_sig_feats_dic_Jan_21th_2021_binary_wo_duplicates_thresh_0_10, cup_sig_feats_dic_Jan_21th_2021_binary_wo_duplicates, cup_sig_feats_dic_Nov_28th_binary_mut_burden, cup_sig_feats_dic_Sep_24th
		pickle.dump(perf_dic_real_time,f)
		f.close()
	return

def compute_integrated_brier_score(surv_prob, data_train_tuple, data_test_tuple, event_time_horizon = None, bootstrap = False, plot = False, filename_suffix = None, curr_epoch = None):
	"""
	Compute integrated brier score based on provided range of time
	https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.metrics.integrated_brier_score.html
	"""
	min_event_time = event_time_horizon[0]
	max_event_time = event_time_horizon[1]
	tp_res = event_time_horizon[2]
	# surv_prob_oi = surv_prob.copy()

	# get test_quantile_times
	horizons = [0.25, 0.5, 0.75]
	test_quantile_times = np.asarray([int(val) for val in np.quantile([t_ for t_, e_ in zip(data_test_tuple[1], data_test_tuple[0]) if e_[-1] == 1], horizons)])

	# train
	num_samples = len(data_train_tuple[0])
	event_oi_train = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=num_samples)
	remaining_time_to_event_train = data_train_tuple[1] #get_remaining_time_to_event(data_dict_train, num_samples)
	event_oi_train['event'] = [bool(event_ind_ == 1) for event_ind_ in data_train_tuple[0]]
	event_oi_train['time'] = [float(event_time_) for event_time_ in remaining_time_to_event_train]
	print('training remaining survival time quantile : ' + ' 5 % : ' + str(np.round(np.quantile(event_oi_train['time'], q = 0.05))) + ', 95 % : ' + str(np.round(np.quantile(event_oi_train['time'], q = 0.95))))

	# test
	event_oi = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=len(data_test_tuple[0]))
	event_oi['event'] = [bool(event_ind_ == 1) for event_ind_ in data_test_tuple[0]]
	event_oi['time'] = [float(event_time_) for event_time_ in data_test_tuple[1]]
	
	times_oi = np.arange(test_quantile_times[0], test_quantile_times[-1]) #np.arange(min_event_time, int(np.quantile(event_oi['time'], q = 0.9)), tp_res) # remaining time to event	
	surv_prob_oi = []
	for surv_prob_ in surv_prob:
		surv_prob_oi.append(surv_prob_[np.arange(test_quantile_times[0], test_quantile_times[-1])])
	
	ibs = integrated_brier_score(event_oi_train, event_oi, surv_prob_oi, times_oi)
	times, bs = brier_score(event_oi_train, event_oi, surv_prob_oi, times_oi)
	
	if bootstrap:
		perf_dic = {}
		sampled_se, conf_int_ibs = perform_bootstrap(ibs, len(surv_prob_oi), mode = 'ibs', colname = ('estimate', 'survival_test'), func = integrated_brier_score, func_args = {'survival_train' : event_oi_train, 'survival_test' : event_oi, 'estimate' : surv_prob, 'times' : times_oi})
		perf_dic['ibs'] = ibs
		perf_dic['ibs_se'] = sampled_se
		perf_dic['ibs_conf_int'] = conf_int_ibs
		return perf_dic
	else:
		return ibs

def get_performance_at_quantiles(surv_metric, data_train_tuple, data_test_tuple, event_idx = None, run_id = None, sample_ids = None, event_time_horizon = None, bootstrap = False, dataset = None, n_events = 1, ef_surv = False, cif = False, time_varying_metric = True, horizons = None):
	"""
	This function evaluates the model's performance for a chosen event of interest

	-- Input --
	surv_metric : relevent metric for evaluation
				  i) when ef_surv (effect-free survival set to true), ef_surv_prob
				  ii) when not ef_surv, cause specific CIF
				  iii) in a single risk, surv_prob

	data_train_tuple : (event_ind, time to event, last observation time)
	data_test_tuple : (event_ind, time to event, last observation time)
	event_time_horizon : (min prediction window, max prediction window, time point resolution)

	-- Output --
	If boostrap set to true
	test_stat_dic : dictionary of scores across different quantiles
	sampled_ses_dic : dictionary of standard errors of scores across different quantiles
	conf_ints_dic : dictionary of 95% confidence intervals of scores across different quantiles

	else
	test_stat_dic
	"""

	# Specify horizons
	# when computing quantiles, get time-to-event from the entry 
	if horizons is None:
		horizons = [0.25, 0.375, 0.5, 0.625, 0.75]
	max_pred_window = event_time_horizon[1]
	# remaining time-to-event quantiles
	# main assumption here is that lst obs measurement is more or less ovelapping with sequence date
	test_quantile_times = np.asarray([int(val) for val in np.quantile([t_ for t_, e_ in zip(data_test_tuple[1], data_test_tuple[0]) if e_[-1] == 1], horizons)])
	print('remaining time-to-event in validation/test set [25%, 37.5%, 50%, 62.5%, 75%] percentiles : ', test_quantile_times)
	
	# create test stat dic
	test_stat_dic = {'num_samples' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0},
					 'auc' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0}, 
					 'bs' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0}, 
					 'bs_censored' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0}, 
					 'bs_uncensored' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0}, 
					 'c_idx' : {0.25 : 0.0, 0.375 : 0.0, 0.5 : 0.0, 0.625 : 0.0, 0.75 : 0.0}}
	
	last_observed_points_train = data_train_tuple[2]
	last_observed_points = data_test_tuple[2]
	quant_to_event_oi_train_test_surv_dict = {}
	
	for quant_time, quant in zip(test_quantile_times, horizons):
		surv_metric_oi = []; surv_metric_oi_cum = []; eligible_indices = []
		for idx, (surv_ind, last_obs_t) in enumerate(zip(surv_metric, data_test_tuple[2])):
			try:
				last_obs_t_int = int(last_obs_t.cpu().numpy())
			except:
				last_obs_t_int = int(last_obs_t)
			if quant_time < len(surv_ind):
				surv_metric_oi.append(surv_ind[quant_time])
				eligible_indices.append(idx)

		surv_metric_oi = np.asarray(surv_metric_oi)
		# surv_metric_oi_cum = np.asarray(surv_metric_oi_cum)

		train_events_oi = data_train_tuple[0]#
		train_remain_tte_oi = data_train_tuple[1]#

		test_events_oi = data_test_tuple[0][eligible_indices]
		test_reamain_tte_oi = data_test_tuple[1][eligible_indices]

		last_observed_points_oi = np.asarray(last_observed_points)[eligible_indices]

		num_samples_train = len(train_events_oi)
		event_oi_train = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=num_samples_train)
		event_oi_train['event'] = [bool(event_ind_ == 1) for event_ind_ in train_events_oi]
		event_oi_train['time'] = [float(event_time_) for event_time_ in train_remain_tte_oi]

		num_samples = len(test_events_oi)
		event_oi = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=num_samples)
		event_oi['event'] = [bool(event_ind_ == 1) for event_ind_ in test_events_oi]
		event_oi['time'] = [float(event_time_) for event_time_ in test_reamain_tte_oi]
		test_stat_dic['num_samples'][quant] = num_samples
		
		test_stat_dic, quant_to_event_oi_train_test_surv_dict = _performance_at_quantiles(test_stat_dic, event_oi_train, event_oi, surv_metric_oi, test_events_oi, test_reamain_tte_oi, last_observed_points_oi, quant_to_event_oi_train_test_surv_dict = quant_to_event_oi_train_test_surv_dict, surv_metric_oi_cum = surv_metric_oi_cum, quant = quant, quant_time = quant_time, max_pred_window = max_pred_window, ef_surv = ef_surv, n_events = n_events)
		
		if quant == 0.75:
			times_oi = np.arange(test_quantile_times[0], test_quantile_times[-1])
			surv_metric_total = []
			for idx, surv_ind_ in enumerate(surv_metric):
				if idx in eligible_indices:
					surv_metric_total.append(surv_ind_[test_quantile_times[0]:test_quantile_times[-1]])
			# 25 - 76 percentiles
			if n_events == 1:
				auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, 1-np.asarray(surv_metric_total), times_oi)
				ibs = integrated_brier_score(event_oi_train, event_oi, np.asarray(surv_metric_total), times_oi)
			else:
				auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, np.asarray(surv_metric_total), times_oi)
				ibs = integrated_brier_score(event_oi_train, event_oi, 1 - np.asarray(surv_metric_total), times_oi)
				# breakpoint()
	# if not ef_surv:
	test_stat_dic['mean_auc'] = mean_auc
	test_stat_dic['ibs'] = ibs
	# else:
	# 	test_stat_dic['mean_bs_ef'] = np.mean([bs for bs in test_stat_dic['bs'].values()])
	
	display_performance_at_quantiles(test_stat_dic, ef_surv = ef_surv, n_events = n_events)
	if bootstrap:
		sampled_ses_dic = perform_bootstrap_quantile(test_stat_dic, quant_to_event_oi_train_test_surv_dict, ef_surv = ef_surv, max_pred_window = event_time_horizon[1])																
		return test_stat_dic, sampled_ses_dic
	else:
		return test_stat_dic, None

def get_performance_results(model, results, batch_dict_train, batch_dict, event_time_horizon = None, curr_epoch = None, dataset = None, surv_est = None, bootstrap = False, filename_suffix = None, plot_survival_curves = False, validation = False, plot = False, feat_names = None, high_prop_feats_idx = None, n_events = 1, compute_cont_metric = False):
	
	min_event_time = event_time_horizon[0]; max_event_time = event_time_horizon[1]; tp_res = event_time_horizon[2]
	num_samples = len(batch_dict['sample_ids'])
	

	last_observed_points = batch_dict['end_of_obs_idx']
	event_ind = [int(val[0]) for val in batch_dict['labels'].cpu().numpy()]
	remaining_time_to_event = batch_dict['remaining_time_to_event']
	
	data_train_tuple = (batch_dict_train['labels'], batch_dict_train['remaining_time_to_event'], batch_dict_train['end_of_obs_idx'])
	data_test_tuple = (batch_dict['labels'], batch_dict['remaining_time_to_event'], last_observed_points)
	
	if surv_est == 'Hazard': # non-parametrics
		if n_events == 1:
			# compuute survival prob :
			surv_prob = compute_survival_curves(results, batch_dict, batch_dict_train, last_observed_points, surv_est = surv_est, n_events = n_events)
			# get performances at quantiles :
			quantile_perf_dic = {}
			test_stat_dic, sampled_ses_dic = get_performance_at_quantiles(surv_prob, data_train_tuple, data_test_tuple, bootstrap = False, event_time_horizon = (min_event_time, max_event_time, tp_res), dataset = dataset)
			
			quantile_perf_dic['point_ests'] = test_stat_dic
			quantile_perf_dic['bootstrap_ses'] = sampled_ses_dic
			# quantile_perf_dic['bootstrap_cis'] = conf_ints_dic

			perf_dic = {}
			perf_dic['auc'] = 0.0 # zeros are put as dummies
			perf_dic['mean_auc'] = test_stat_dic['mean_auc']
			perf_dic['ibs'] = 0.0

			perf_dic['c_idx'] = 0.0
		else:
			# compuute survival prob :
			ef_surv_prob, cs_cif_total = compute_survival_curves(results, batch_dict, batch_dict_train, last_observed_points, surv_est = surv_est, n_events = n_events)

			perf_dic_list = []; quantile_perf_dic_list = []
			for idx_, cs_cif_ in enumerate(cs_cif_total):
				print('----------------------------------------------------')
				print('outcome : ', idx_ + 1)
				# for model evaluation
				data_train_tuple = (batch_dict_train['labels'] == idx_ + 1, batch_dict_train['remaining_time_to_event'], batch_dict_train['end_of_obs_idx'])
				data_test_tuple = (batch_dict['labels'] == idx_ + 1, batch_dict['remaining_time_to_event'], last_observed_points)
				# get performances at quantiles :
				quantile_perf_dic = {}
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					if validation == True:
						test_stat_dic, sampled_ses_dic = get_performance_at_quantiles(cs_cif_, data_train_tuple, data_test_tuple, bootstrap = False, event_time_horizon = (min_event_time, max_event_time, tp_res), dataset = dataset, n_events = n_events)
					else:
						test_stat_dic, sampled_ses_dic = get_performance_at_quantiles(cs_cif_, data_train_tuple, data_test_tuple, bootstrap = True, event_time_horizon = (min_event_time, max_event_time, tp_res), dataset = dataset, n_events = n_events)
				quantile_perf_dic['point_ests'] = test_stat_dic
				quantile_perf_dic['bootstrap_ses'] = sampled_ses_dic
				quantile_perf_dic_list.append(quantile_perf_dic)

				perf_dic = {}
				perf_dic['mean_c_idx'] = 0.0
				perf_dic['mean_auc'] = test_stat_dic['mean_auc']
				perf_dic['auc'] = 0.0 # zeros are put as dummies
				perf_dic['ibs'] = test_stat_dic['ibs']
				perf_dic_list.append(perf_dic)
			# event free survival metric
			# data_train_tuple = (batch_dict_train['labels'] != 0, batch_dict_train['remaining_time_to_event'], batch_dict_train['end_of_obs_idx'])
			# data_test_tuple = (batch_dict['labels'] != 0, batch_dict['remaining_time_to_event'], last_observed_points)
			# get ibs
			# perf_dic = {}
			# if compute_cont_metric:
			# 	ibs, ibs_censored, ibs_uncensored = compute_integrated_brier_score(ef_surv_prob, data_train_tuple, data_test_tuple, event_time_horizon = (min_event_time, max_event_time, tp_res), bootstrap = bootstrap, filename_suffix = filename_suffix, curr_epoch = curr_epoch, plot = plot)
			# 	perf_dic['ibs_ef'] = ibs
			# 	print('\n')
			# 	print('event free survival : integarted brier score from last obs to 90 % quantile (' + str(np.round(np.quantile(remaining_time_to_event, q = 0.9))) + ') : ', np.round(ibs['ibs'] if bootstrap else ibs, 3))
			# else:
			# 	perf_dic['ibs_ef'] = 0.0
			# perf_dic_list.append(perf_dic)
			# with warnings.catch_warnings(): # ignore warnings in a partial section of the code
			# 	warnings.simplefilter("ignore")
			# 	if validation == True:
			# 		ef_test_stat_dic, ef_sampled_ses_dic, ef_conf_ints_dic = get_performance_at_quantiles(ef_surv_prob, data_train_tuple, data_test_tuple, bootstrap = False, event_time_horizon = (min_event_time, max_event_time, tp_res), dataset = dataset, n_events = n_events, ef_surv = True)
			# 	else: # test set; do bootstrap
			# 		ef_test_stat_dic, ef_sampled_ses_dic, ef_conf_ints_dic = get_performance_at_quantiles(ef_surv_prob, data_train_tuple, data_test_tuple, bootstrap = True, event_time_horizon = (min_event_time, max_event_time, tp_res), dataset = dataset, n_events = n_events, ef_surv = True)
			# quantile_perf_dic['point_ests_ef'] = ef_test_stat_dic
			# quantile_perf_dic['bootstrap_ses_ef'] = ef_sampled_ses_dic
			# quantile_perf_dic['bootstrap_cis_ef'] = ef_conf_ints_dic
			# for idx_list in range(len(perf_dic_list)):
			# 	perf_dic_list[idx_list]['mean_bs_ef'] = ef_test_stat_dic['mean_bs_ef']
			# quantile_perf_dic_list.append(quantile_perf_dic)
	else:
		raise NotImplementedError

	# plot survival curves
	if plot_survival_curves:
		labels = batch_dict['labels']
		if n_events == 1:
			func_plot_survival_curves(surv_prob, None, labels, remaining_time_to_event, last_observed_points, n_events = n_events, curr_epoch = curr_epoch, filename_suffix = filename_suffix)
		else:
			func_plot_survival_curves(ef_surv_prob, cs_cif_total, labels, remaining_time_to_event, last_observed_points, n_events = n_events, curr_epoch = curr_epoch, filename_suffix = filename_suffix)
	if n_events == 1:
		return remaining_time_to_event, None, perf_dic, quantile_perf_dic
	else:
		return remaining_time_to_event, None, perf_dic_list, quantile_perf_dic_list

def plot_performance(model_performance, surv_est = None, bootstrap = False, filename_suffix = None, event_idx = 0):
	"""
	Plots performance metrics (c-idx, AUC, IBS) across epochs
	"""
	fig, ax = plt.subplots()
	t = np.arange(0, len(model_performance[0]), 1)

	# if surv_est == 1: # cox
	# 	label_auc = 'mean AUC (ODE-RNN Cox)'; label_cidx = 'C-idx (ODE-RNN Cox)'; label_ibs = 'IBS (ODE-RNN Cox)';
	# elif survival_mode_num == 2:
	# 	label_auc = 'mean AUC (ODE-RNN Weibull)'; label_cidx = 'C-idx (ODE-RNN Weibull)'; label_ibs = 'IBS (ODE-RNN Weibull)';
	if surv_est == 'Softmax' or surv_est == 'Hazard':
		label_auc = 'mean AUC (Latent ODE-Surv)'; label_cidx = 'C-idx (Latent ODE-Surv)'; label_ibs = 'IBS (Latent ODE-Surv)'; label_reconstr = 'Reconstr. Loss (Latent ODE-Surv)'; label_surv = 'Surv. Loss (Latent ODE-Surv)'
	elif surv_est == 'Cox':
		label_auc = 'mean AUC (Latent ODE-Cox)'; label_cidx = 'C-idx (Latent ODE-Cox)'; label_ibs = 'IBS (Latent ODE-Cox)'; label_reconstr = 'Reconstr. Loss (Latent ODE-Cox)'; label_surv = 'Surv. Loss (Latent ODE-Cox)'

	if bootstrap:
		# exclude CI and get observed estimates only 
		mean_aucs = [dic['mean_auc'] for dic in model_performance[2]]
		ibss = [dic['ibs'] for dic in model_performance[1]]
	else:
		mean_aucs = model_performance[2]
		ibss = model_performance[1]

	reconstr_losses = model_performance[3]
	survival_losses = model_performance[4]
	# check if current iteration is the best :
	# if best_auc_index == curr_epoch and curr_epoch >= 5:
		# compute performance of the test 
	best_auc_index = np.argmax(model_performance[2])
	ax.scatter(t, mean_aucs, color = 'b', label = label_auc)
	ax.text(t[best_auc_index], model_performance[2][best_auc_index], str(np.round(model_performance[2][best_auc_index], 3)))#, label = 'Concordance')

	ax.scatter(t, model_performance[0], color = 'r', label = label_cidx)
	best_concord_idx = np.argmax(model_performance[0])
	ax.text(t[best_concord_idx], model_performance[0][best_concord_idx], str(np.round(model_performance[0][best_concord_idx], 3)))#, label = 'Concordance')
	
	ax.scatter(t, ibss, color = 'k', label = label_ibs)
	best_ibs_idx = np.argmin(model_performance[1])
	ax.text(t[best_ibs_idx], model_performance[1][best_ibs_idx], str(np.round(model_performance[1][best_ibs_idx], 3)))#, label = 'Concordance')

	# add conventional Cox time-varying performance
	# if baseline_model_comparison:
	# 	ax.hlines(cox_perf_dic['mean_auc'], 0, len(model_performance[0]), color = 'b', linestyles = 'dashed', label = 'mean AUC (Cox time-varying)')
	# 	ax.hlines(cox_perf_dic['c_idx'], 0, len(model_performance[0]), color = 'r', linestyles = 'dashed', label = 'C-idx (Cox time-varying)')
	# 	ax.hlines(cox_perf_dic['ibs'], 0, len(model_performance[0]), color = 'k', linestyles = 'dashed', label = 'IBS (Cox time-varying)')

	ax.set(xlabel='n_epochs', ylabel='performance')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 3)
	plt.xticks(np.arange(0, len(model_performance[0]) + 1, 3))
	plt.xlim(left=0)
	fig.savefig("model_performance/" + str(filename_suffix) + "/model_performance_" + str(event_idx) + '_' + str(filename_suffix) + ".pdf", bbox_inches='tight')
	plt.close()	

	# reconstr loss
	fig, ax = plt.subplots()
	best_reconst_index = np.argmax(model_performance[3])
	ax.scatter(t, reconstr_losses, color = 'b', label = label_reconstr)
	ax.text(t[best_reconst_index], model_performance[3][best_reconst_index], str(np.round(model_performance[3][best_reconst_index], 3)))	
	ax.set(xlabel='n_epochs', ylabel='performance')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 3)
	plt.xticks(np.arange(0, len(model_performance[0]) + 1, 3))
	plt.xlim(left=0)
	fig.savefig("model_performance/" + str(filename_suffix) + "/model_performance_reconstr_loss_" + str(event_idx) + '_' + str(filename_suffix) + ".pdf", bbox_inches='tight')
	plt.close()	


	# survival loss
	fig, ax = plt.subplots()
	best_surv_loss_index = np.argmin(model_performance[4])
	ax.scatter(t, survival_losses, color = 'b', label = label_surv)
	ax.text(t[best_surv_loss_index], model_performance[4][best_surv_loss_index], str(np.round(model_performance[4][best_surv_loss_index], 3)))
	ax.set(xlabel='n_epochs', ylabel='performance')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 3)
	plt.xticks(np.arange(0, len(model_performance[0]) + 1, 3))
	plt.xlim(left=0)
	fig.savefig("model_performance/" + str(filename_suffix) + "/model_performance_surv_loss_" + str(event_idx) + '_' + str(filename_suffix) + ".pdf", bbox_inches='tight')
	plt.close()		

	return

def variable_time_collate_fn_survival(batch, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, dataset = None, max_pred_window = None, feat_reconstr_idx = None, feat_names = None, check_extrapolation = False, max_obs_time = None, pred_from_sequence = True):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][3].shape[1] # 3 since we need to get data
	# get union of all the time points (ex[1] corresponds to time points for each sample)
	combined_tt_oi, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	# breakpoint()
	combined_tt_oi = combined_tt_oi.to(device)

	# this has to be regularly intervaled with length of the max obs. time
	combined_tt = torch.arange(0, max_obs_time + 1).to(device)

	# time_to_event from baseline at t = 0
	# combined_time_to_event = torch.zeros([len(batch)]).to(device)
	combined_time_to_event_idx = torch.zeros([len(batch)]).to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

	if feat_reconstr_idx is not None:
		combined_vals_reconstr = torch.zeros([len(batch), len(combined_tt), len(feat_reconstr_idx)]).to(device)
		combined_mask_reconstr = torch.zeros([len(batch), len(combined_tt), len(feat_reconstr_idx)]).to(device)

	# for extrapolation
	combined_tt_ext = []
	combined_vals_ext = [] # doesn't need to be torch based
	combined_mask_ext = []
	"""
	Add combined mask surv and get the right shape! 
	"""
	if dataset == 'general':
		if max_pred_window is not None:
			combined_mask_surv = torch.zeros([len(batch), int(max_pred_window)]).to(device)
		else:
			combined_mask_surv = None
	else:
		raise NotImplementedError

	combined_labels = None
	N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)
	
	dur_total = torch.zeros(len(batch)).to(device)
	if check_extrapolation:
		for idx, (_, _, _, _, _, _, _, _, _, dur) in enumerate(batch):
			# in general dataset, get duration from the first observation
			dur_total[idx] = dur[0]
	else:
		for idx, (_, _, _, _, _, _, dur) in enumerate(batch):
			# in general dataset, get duration from the first observation
			dur_total[idx] = dur[0]
	sample_ids = []; pred_horizon_idx = []; remaining_time_to_event = []; end_of_obs_idx = []
	if check_extrapolation:
		# itr_tuple = 
		for b, (record_id, tt, tt_ext, vals, vals_ext, mask, mask_ext, mask_surv, labels, dur) in enumerate(batch):
			sample_ids.append(record_id)
			tt = tt.to(device)
			
			vals = vals.to(device)
			mask = mask.to(device)
			
			mask_surv = mask_surv.to(device) # mask for survival. only use hazard from the latest observation to its time of the event
			assert mask_surv.sum() > 0, f"tte value has to be greater than the lastest observation time for ID: {record_id}"

			if labels is not None:
				labels = labels.to(device)

			indices = tt.long()

			combined_vals[b, indices] = vals
			combined_mask[b, indices] = mask

			end_of_obs_idx.append(tt[-1])

			if check_extrapolation:
				combined_tt_ext.append(tt_ext)

			if feat_reconstr_idx is not None:
				combined_vals_reconstr[b, indices] = vals[:, feat_reconstr_idx]
				combined_mask_reconstr[b, indices] = mask[:, feat_reconstr_idx]
				if check_extrapolation:
					combined_vals_ext.append(vals_ext[:, feat_reconstr_idx])
					combined_mask_ext.append(mask_ext[:, feat_reconstr_idx])
			else:
				if check_extrapolation:
					combined_vals_ext.append(vals_ext)
					combined_mask_ext.append(mask_ext)

			if combined_mask_surv is not None:
				combined_mask_surv[b] = mask_surv

			dur_oi = dur[0] # duration from the very initial observation
			
			remaining_time_to_event.append(dur[-1])
			time_to_event_idx = len(combined_tt[combined_tt < dur_oi]) - 1
			
			pred_start_idx = len(combined_tt[combined_tt < tt[-1]]) 
			if time_to_event_idx == len(combined_tt) - 1:
				combined_time_to_event_idx[b] = time_to_event_idx + len(dur_total[dur_total < float(dur_oi)])
				pred_end_idx = time_to_event_idx + int(dur_oi - combined_tt[-1]) + 1
			else: 
				combined_time_to_event_idx[b] = len(combined_tt[combined_tt < dur_oi]) - 1
				pred_end_idx = len(combined_tt[combined_tt <= dur_oi]) 
			pred_horizon_idx.append((pred_start_idx, pred_end_idx))

			if pred_start_idx > pred_end_idx:
				raise KeyError('pred_start_idx cannot be larger than pred_end_idx')

			if labels is not None:
				combined_labels[b] = labels
	else:
		for b, (record_id, tt, vals, mask, mask_surv, labels, dur) in enumerate(batch):
			sample_ids.append(record_id)
			tt = tt.to(device)
			vals = vals.to(device)
			mask = mask.to(device)
			mask_surv = mask_surv.to(device) # mask for survival. only use hazard from the latest observation to its time of the event
			assert mask_surv.sum() > 0, f"tte value has to be greater than the lastest observation time for ID: {record_id}"

			if labels is not None:
				labels = labels.to(device)

			indices = tt.long()
			combined_vals[b, indices] = vals
			combined_mask[b, indices] = mask

			end_of_obs_idx.append(tt[-1])

			if check_extrapolation:
				combined_tt_ext.append(tt_ext)

			if feat_reconstr_idx is not None:
				# breakpoint()
				combined_vals_reconstr[b, indices] = vals[:, feat_reconstr_idx]
				combined_mask_reconstr[b, indices] = mask[:, feat_reconstr_idx]
				if check_extrapolation:
					combined_vals_ext.append(vals_ext[:, feat_reconstr_idx])
					combined_mask_ext.append(mask_ext[:, feat_reconstr_idx])
			else:
				if check_extrapolation:
					combined_vals_ext.append(vals_ext)
					combined_mask_ext.append(mask_ext)
			if combined_mask_surv is not None:
				combined_mask_surv[b] = mask_surv
			
			dur_oi = dur[0] # duration from the very initial observation
			
			remaining_time_to_event.append(dur[-1])
			time_to_event_idx = len(combined_tt[combined_tt < dur_oi]) - 1
			
			pred_start_idx = len(combined_tt[combined_tt < tt[-1]]) 
			if time_to_event_idx == len(combined_tt) - 1:
				combined_time_to_event_idx[b] = time_to_event_idx + len(dur_total[dur_total < float(dur_oi)])
				pred_end_idx = time_to_event_idx + int(dur_oi - combined_tt[-1]) + 1
			else: 
				combined_time_to_event_idx[b] = len(combined_tt[combined_tt < dur_oi]) - 1
				pred_end_idx = len(combined_tt[combined_tt <= dur_oi]) 
			pred_horizon_idx.append((pred_start_idx, pred_end_idx))
			# breakpoint()
			if pred_start_idx > pred_end_idx:
				raise KeyError('pred_start_idx cannot be larger than pred_end_idx')

			if labels is not None:
				combined_labels[b] = labels

	# normalization
	combined_vals, _, _ = normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max, device = device)

	if feat_reconstr_idx is not None:
		combined_vals_reconstr, _, _ = normalize_masked_data(combined_vals_reconstr, combined_mask_reconstr, 
			att_min = data_min[feat_reconstr_idx], att_max = data_max[feat_reconstr_idx], device = device)

	# for extrapolated values
	if check_extrapolation:
		combined_vals_ext, _, _ = normalize_masked_data(combined_vals_ext, combined_mask_ext, 
			att_min = data_min if feat_reconstr_idx is None else data_min[feat_reconstr_idx], att_max = data_max if feat_reconstr_idx is None else data_max[feat_reconstr_idx], extra = True, device = device)

	if torch.max(combined_tt) != 0.:
		# normalize combined tt
		combined_tt_normalized = (combined_tt - torch.min(combined_tt)) / torch.max(combined_tt)
	total_time_to_pred = torch.arange(0, max_pred_window, 1).to(device)
	total_time_to_pred_normalized = (total_time_to_pred - torch.min(total_time_to_pred)) / (torch.max(total_time_to_pred) - torch.min(total_time_to_pred))
	
	data_dict = {
		"sample_ids":sample_ids,
		"observed_data": combined_vals, 
		"data_to_predict": combined_vals_reconstr if feat_reconstr_idx is not None else combined_vals,
		"observed_tp": combined_tt_normalized,
		"observed_tp_unnorm": combined_tt, 
		"tp_to_predict": total_time_to_pred_normalized,
		"tp_to_predict_unnorm": total_time_to_pred,
		"pred_horizon_idx":pred_horizon_idx,
		"observed_mask": combined_mask,
		"mask_predicted_data":combined_mask_reconstr if feat_reconstr_idx is not None else combined_mask, # for reconstruction purpose; some fixed features are not reconstructed..
		"mask_surv": combined_mask_surv,
		"data_extra_info" : (combined_tt_ext, combined_vals_ext, combined_mask_ext) if check_extrapolation else (None, None, None),
		"labels": combined_labels,
		"event_time_idx": combined_time_to_event_idx, 
		"event_times": dur_total, 
		"end_of_obs_idx":end_of_obs_idx,
		"max_end_of_obs_idx":int(np.max(end_of_obs_idx)),
		"max_obs_time":max_obs_time,
		"remaining_time_to_event":np.asarray(remaining_time_to_event),
		"feat_names":np.asarray(feat_names)[feat_reconstr_idx] if feat_reconstr_idx is not None else feat_names}

	return data_dict

def get_data_min_max(records, dataset = None, device = None, check_extrapolation = False, data_min = None, data_max = None):
	inf = torch.Tensor([float("Inf")])[0]#.to(device)

	if check_extrapolation:
		for b, (record_id, tt, _, vals, _, mask, _, mask_surv, labels, dur) in enumerate(records):
			n_features = vals.size(-1)

			batch_min = []
			batch_max = []
			for i in range(n_features):
				non_missing_vals = vals[:,i][mask[:,i] == 1]
				if len(non_missing_vals) == 0:
					batch_min.append(inf)
					batch_max.append(-inf)
				else:
					batch_min.append(torch.min(non_missing_vals))
					batch_max.append(torch.max(non_missing_vals))

			batch_min = torch.stack(batch_min)
			batch_max = torch.stack(batch_max)

			if (data_min is None) and (data_max is None):
				data_min = batch_min
				data_max = batch_max
			else:
				data_min = torch.min(data_min, batch_min)
				data_max = torch.max(data_max, batch_max)		
	else:
		for b, (record_id, tt, vals, mask, mask_surv, labels, dur) in enumerate(records):
			n_features = vals.size(-1)

			batch_min = []
			batch_max = []
			for i in range(n_features):
				non_missing_vals = vals[:,i][mask[:,i] == 1]
				if len(non_missing_vals) == 0:
					batch_min.append(inf)
					batch_max.append(-inf)
				else:
					batch_min.append(torch.min(non_missing_vals))
					batch_max.append(torch.max(non_missing_vals))

			batch_min = torch.stack(batch_min)
			batch_max = torch.stack(batch_max)

			if (data_min is None) and (data_max is None):
				data_min = batch_min
				data_max = batch_max
			else:
				data_min = torch.min(data_min, batch_min)
				data_max = torch.max(data_max, batch_max)	

	return data_min, data_max

def train_surv_model(model, data_obj, params_dic, device = None, surv_est = None, max_pred_window = None, run_id = None, dataset = None, survival_loss_scale = 10, n_latent_traj = 1, early_stopping = False, survival_loss_exp = False, train_info = None, n_events = 1, wait_until_full_surv_loss = 15):
	
	ckpt_path = os.path.join('experiments/','experiment_' + str(run_id) + '.ckpt')

	file_name = os.path.basename(__file__)[:-3]
	log_path = "logs/" + file_name + "_" + str(run_id) + ".log"
	if not os.path.exists("logs/"):
		makedirs("logs/")
	logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	input_command = ''
	logger.info(input_command)

	# create local storage dir.
	parent_dir = "model_performance/"
	path = os.path.join(parent_dir, run_id)
	if not os.path.exists(path):
		os.mkdir(path)
	else:
		print('using currently existing directory : ', path)

	parent_dir = "surv_curves/"
	path = os.path.join(parent_dir, run_id)
	if not os.path.exists(path):
		os.mkdir(path)
		print("Directory '% s' created" % run_id)
	else:
		print('using currently existing directory : ', path)

	# parent_dir = path + '/reconstruction'
	# if not os.path.exists(parent_dir):
	# 	os.mkdir(parent_dir)
	# 	print("Directory '% s' created" % parent_dir)
	# else:
	# 	print('using currently existing directory : ', parent_dir)

	optimizer = optim.Adamax(model.parameters(), lr=params_dic['lr'])
	if train_info is not None: # prev latest model has been loaded
		optimizer.load_state_dict(train_info['optimizer_state_dict'])
		# If you wish to resuming training from the last time point, call model.train() to ensure these layers are in training mode.
		model.train()

	min_max_data_tuple = model.get_min_max_data()
	num_batches = data_obj["n_train_batches"]
	wait_until_kl_inc = 10;
	for itr in tqdm(range(1, num_batches * params_dic['niters'] + 1), desc = 'Training across ' + str(params_dic['niters']) + ' epochs'):
		optimizer.zero_grad()
		update_learning_rate(optimizer, decay_rate = 0.999, lowest = params_dic['lr'] / 10)

		if itr // num_batches < wait_until_kl_inc if train_info is None else (train_info['itr'] + itr) // num_batches < wait_until_kl_inc:
			kl_coef = 0
			survival_loss_scale_actual = 0
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
			survival_loss_scale_actual = survival_loss_scale

		if survival_loss_exp:
			if (itr // num_batches) == 0 if train_info is None else (train_info['itr'] + itr) // num_batches == 0:
				survival_loss_scale_actual = 0.01
			elif (itr // num_batches) < wait_until_full_surv_loss if train_info is None else (train_info['itr'] + itr) // num_batches < wait_until_full_surv_loss:
				# initially 0.01 and then exponentially increases up to survival_loss_scale across the prespecified number of epochs
				survival_loss_scale_actual = 0.01*math.exp(1/wait_until_full_surv_loss*math.log(survival_loss_scale/0.01)*(itr // num_batches)) 
			else:
				survival_loss_scale_actual = survival_loss_scale
		else:
			survival_loss_scale_actual = survival_loss_scale

		batch_dict = get_next_batch(data_obj["train_dataloader"])
		batch_dict = remove_timepoints_wo_obs(batch_dict)
		train_res = model.compute_all_losses(batch_dict, kl_coef = kl_coef, surv_est = surv_est, survival_loss_scale = survival_loss_scale_actual) # survival set to true for time-to-event esimtation
		train_res["loss"].backward()
		optimizer.step()

		if itr % num_batches == 0:
			if train_info is not None:
				print('Epoch : ', (train_info['itr'] + itr) // num_batches)
			else:
				print('Epoch : ', itr//num_batches)
			with torch.no_grad():
				# if survival_mode_num != 3:
				model_performance = compute_loss_all_batches(model, data_obj["train_dataloader_full_batch"] if "train_dataloader_full_batch" in data_obj.keys() else None, 
					data_obj["valid_dataloader"] if "valid_dataloader" in data_obj.keys() else None, params_dic,
					n_batches_train = num_batches,
					device = device,
					kl_coef = kl_coef, itr = itr if train_info is None else train_info['itr'] + itr, filename_suffix = run_id, 
					surv_est = surv_est, dataset = dataset, feat_names = data_obj['attr'], max_pred_window = max_pred_window, min_max_data_tuple = min_max_data_tuple, 
					survival_loss_scale = survival_loss_scale_actual, n_latent_traj = n_latent_traj, optimizer = optimizer, n_events = n_events)
			# check for early stopping condition by looking at the last 4 iterations
			if early_stopping and (itr // num_batches) >= max(10, wait_until_full_surv_loss):
				metric = 2 # which corresponds to AUC
				threshold_iters = 3 # this is saying we stop the training if there are 3 consecutive lower scores
				if itr//num_batches > threshold_iters:
					cond = [model_performance[metric][-i -2] > model_performance[metric][-i -1] for i in range(threshold_iters)]
					if all(cond):
						print('no performance improvement across ' + str(threshold_iters + 1) + ' epochs...')
						print('terminating the training...')
						break

		if train_info is not None:
			if train_info['itr'] + itr >= num_batches * (params_dic['niters'] + 1):
				break 
	return

def get_gaussian_likelihood(truth, pred_y, mask = None, obsrv_std = None):
	# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
	# truth shape  [n_traj, n_tp, n_dim]
	n_traj, n_tp, n_dim = truth.size()
	# Compute likelihood of the data under the predictions
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	
	if mask is not None:
		# breakpoint()
		mask = mask.repeat(pred_y.size(0), 1, 1, 1)
	log_density_data = masked_gaussian_log_density(pred_y, truth_repeated, 
		obsrv_std = obsrv_std, mask = mask)
	log_density_data = log_density_data.permute(1,0)
	log_density = torch.mean(log_density_data, 1)
	return log_density

def get_survival_likelihood(hazards_y, batch_dict, include_aug_loss = False, eps = 1e-5, surv_est = None, n_events = 1, event_to_event_weight_dict = None):
	"""
	hazards_y (shape) = [num_traj, samples in batch, time points, 1]
	"""
	if surv_est == 'Hazard': 
		uncensored_ll = []; uncensored_ll_v2 = []; censored_ll = []
		if n_events == 1:
			for hazards_oi, pred_sel in zip(hazards_y[0,:,:,:], batch_dict['mask_surv']):
				# censored
				hazards_oi_sel = torch.masked_select(hazards_oi.view(-1), pred_sel.bool())	
				censored_ll.append(hazards_oi_sel.mul(-1).add(1 + eps).log().sum()) # v1 Ren et al. 
				# uncensored
				hazard_at_event = hazards_oi_sel[-1]
				hazards_oi_sel_bf_event = hazards_oi_sel[:-1] # orig : hazards_oi_sel = hazards_oi_sel[:-1]
				# original uncensored loss
				uncensored_ll.append(hazards_oi_sel_bf_event.mul(-1).add(1 + eps).log().sum().add(hazard_at_event.add(eps).log()))
			censored_ll = torch.stack(censored_ll, 0).to(get_device(hazards_y))
			uncensored_ll = torch.stack(uncensored_ll, 0).to(get_device(hazards_y))
			events = batch_dict['labels'].view(-1)
			return (censored_ll.mul(1 - events) + (uncensored_ll).mul(events)).sum().mul(-1)
		else: # competing events
			for idx, (label, pred_sel) in enumerate(zip(batch_dict['labels'], batch_dict['mask_surv'])):
				if label == 0: # censored case
					# hazard per sample
					hazards_ps = hazards_y[0, :, :, n_events][idx]
					# select relevant hazard across different risks and sum them up
					hazards_ps_sel = torch.masked_select(hazards_ps.view(-1), pred_sel.bool())	

					censored_ll.append((hazards_ps_sel + eps).log().sum())
					int_loss = (hazards_ps_sel + eps).log().sum()
					if np.isinf(int_loss.cpu().detach().numpy()):
						raise KeyError('Detect NaN survival loss')
				else: 
					# get corresponding hazard based on the label
					label_np = label.cpu().numpy()[0]
					hazards_ps = hazards_y[0, :, :, int(label_np - 1)][idx]
					hazards_ps_sel = torch.masked_select(hazards_ps.view(-1), pred_sel.bool())

					if len(hazards_ps_sel) == 1:
						# if a sample survives up to only one time unit, exclude the sample 
						continue
					hazard_at_event = hazards_ps_sel[-1]
					hazards_ps_not = hazards_y[0, :, :, n_events][idx]
					hazards_ps_sel_not = torch.masked_select(hazards_ps_not.view(-1), pred_sel.bool())
					hazard_ps_sel_not_bf_event = hazards_ps_sel_not[:-1]

					ll_ind = (hazard_ps_sel_not_bf_event + eps).log().sum().add(hazard_at_event.add(eps).log())# individual likelihood
					uncensored_ll.append(event_to_event_weight_dict[label_np]*ll_ind) 
					int_loss = (hazard_ps_sel_not_bf_event +  eps).log().sum().add(hazard_at_event.add(eps).log())
					if np.isinf(int_loss.cpu().detach().numpy()):
						raise KeyError('Detect NaN survival loss')

			if len(censored_ll) > 0 and len(uncensored_ll) > 0:
				censored_ll = torch.stack(censored_ll, 0).to(get_device(hazards_y[0]))
				uncensored_ll = torch.stack(uncensored_ll, 0).to(get_device(hazards_y[0]))
				loss = (censored_ll.sum() + uncensored_ll.sum()).mul(-1)
				if np.isinf(loss.cpu().detach().numpy()):
					raise KeyError('Detect NaN survival loss')
				return (censored_ll.sum() + uncensored_ll.sum()).mul(-1)
			else:
				return 0
	else:
		raise NotImplementedError

def compute_survival_curves(results, batch_dict, batch_dict_train, last_observed_points, surv_est = None, tp_res = 1, max_time_window = None, filename_suffix = None, validation = False, dataset = None, events_info_train_tuple = None, n_events = 1):
	"""
	Non-parametrically estimate survival prob.
	"""
	if surv_est == 'Hazard':
		if n_events == 1:
			hazards_y_oi = results['hazards_y'][0]
			surv_prob_total = []
			for hazards_oi, pred_range in zip(hazards_y_oi, batch_dict['pred_horizon_idx']):
				hazards_oi_sel = hazards_oi[pred_range[0]:].view(-1) 
				surv_prob = torch.cumprod(hazards_oi_sel.mul(-1.0).add(1.0), dim = 0)
				surv_prob_total.append(surv_prob.cpu().detach().numpy())
			return surv_prob_total
		else: # multiple events
			ef_surv_prob = []; cs_cif_total = []
			for event_idx in range(n_events):
				hazards_y_oi = results['hazards_y'][0] 
				cs_cif_per_event = []
				for hazards_oi, pred_range in zip(hazards_y_oi, batch_dict['pred_horizon_idx']):
					hazards_oi_sel = hazards_oi[pred_range[0]:, n_events].view(-1) 
					if event_idx == 0: # event free survival
						surv_prob = torch.cumprod(hazards_oi_sel, dim = 0)
						ef_surv_prob.append(surv_prob.cpu().detach().numpy())
					# cause-specific CIF
					hazards_oi_sel_oi = hazards_oi[pred_range[0]:, event_idx].view(-1) 
					cs_cif = torch.cumsum(torch.cat((hazards_oi_sel_oi[0][None], hazards_oi_sel_oi[1:] * torch.cumprod(hazards_oi_sel[:-1], dim = 0)), dim = 0), 0)
					cs_cif_per_event.append(cs_cif.cpu().detach().numpy())
				# ipdb.set_trace()
				cs_cif_total.append(cs_cif_per_event) 
			return ef_surv_prob, cs_cif_total
	else:
		raise NotImplementedError

def scaled_dot_product(q, k, v, mask=None):
	d_k = q.size()[-1]
	attn_logits = torch.matmul(q, k.transpose(-2, -1))
	attn_logits = attn_logits / math.sqrt(d_k)
	if mask is not None:
		attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
	attention = F.softmax(attn_logits, dim=-1)
	values = torch.matmul(attention, v)
	return values, attention

def divide_list(l, n):
	list_split = []
	for i in range(0, len(l), n):
		list_split.append(l[i:i+n])
	return list_split

def eval_model(model_info, batch_dict, surv_prob, df_perf_result = None, cs_cif_total = None, run_id = None, min_event_time = 1, max_pred_window = 700, tp_res = 1, n_events = 1, filename_hyp_tuning = 'default', dataset = 'general', idx = None, missing_rate = 0.0):
	if n_events == 1:
		data_train_tuple = (model_info['events_info_train_tuple'][1], model_info['events_info_train_tuple'][2], model_info['events_info_train_tuple'][3])
		data_test_tuple = (batch_dict['labels'], batch_dict['remaining_time_to_event'], batch_dict['end_of_obs_idx'])

		idx_res = 0
		df_test_result_comp = pd.DataFrame([], index = [idx_res], columns = ['param_set', 'best_epoch', 'ibs', 'mean_auc', 'bs 25 percentile', 'bs 25 percentile se', 'bs 50 percentile', 'bs 50 percentile se', 'bs 75 percentile', 'bs 75 percentile se', 'auc 25 percentile', 'auc 25 percentile se', 'auc 50 percentile', 'auc 50 percentile se', 'auc 75 percentile', 'auc 75 percentile se'])
		df_test_result_comp.index.name = 'result'
		
		perf_dic = {}		
		test_stat_dic, sampled_ses_dic = get_performance_at_quantiles(surv_prob, data_train_tuple, data_test_tuple, bootstrap = True, event_time_horizon = (min_event_time, max_pred_window, tp_res), dataset = dataset)
		quantile_perf_dic = {}
		quantile_perf_dic['point_ests'] = test_stat_dic
		quantile_perf_dic['bootstrap_ses'] = sampled_ses_dic

		# ibs = compute_integrated_brier_score(surv_prob, data_train_tuple, data_test_tuple, event_time_horizon = (min_event_time, max_pred_window, tp_res), filename_suffix = run_id, curr_epoch = curr_epoch)		
		
		test_stat_dic_bs = test_stat_dic['bs']
		test_stat_dic_bs_censored = test_stat_dic['bs_censored']
		test_stat_dic_bs_uncensored = test_stat_dic['bs_uncensored']
		sampled_ses_dic_bs = sampled_ses_dic['bs']		
		test_stat_dic_auc = test_stat_dic['auc']
		sampled_ses_dic_auc = sampled_ses_dic['auc']

		if df_perf_result is not None:
			for colname, colname_res in zip(['brier score (25 %)', 'brier score (50 %)', 'brier score (75 %)'], [25, 50, 75]):
				df_perf_result.at[idx, colname] = str(np.round(test_stat_dic_bs[colname_res], 3)) + ' (' + str(np.round(sampled_ses_dic_bs[colname_res], 3)) + ')'
			for colname, colname_res in zip(['brier score (25 %) censored', 'brier score (50 %) censored', 'brier score (75 %) censored'], [25, 50, 75]):
				df_perf_result.at[idx, colname] = str(np.round(test_stat_dic_bs_censored[colname_res], 3))
			for colname, colname_res in zip(['brier score (25 %) uncensored', 'brier score (50 %) uncensored', 'brier score (75 %) uncensored'], [25, 50, 75]):
				df_perf_result.at[idx, colname] = str(np.round(test_stat_dic_bs_uncensored[colname_res], 3))
			for colname, colname_res in zip(['AUC (25 %)', 'AUC (50 %)', 'AUC (75 %)'], [25, 50, 75]):
				df_perf_result.at[idx, colname] = str(np.round(test_stat_dic_auc[colname_res], 3)) + ' (' + str(np.round(sampled_ses_dic_auc[colname_res], 3)) + ')'
			df_perf_result.at[idx, 'reconstr_loss'] = rec_loss.cpu().detach().numpy()[0]
			df_perf_result.at[idx, 'run_id'] = run_id
			df_perf_result.at[idx, 'best_epoch'] = model_info['best_epoch']
			# df_perf_result.at[idx, 'ibs_censored'] = ibs_censored
			# df_perf_result.at[idx, 'ibs_uncensored'] = ibs_uncensored
			try:
				df_perf_result.at[idx, 'mean_auc'] = test_stat_dic['mean_auc']
				df_perf_result.at[idx, 'mean_auc_se'] = sampled_ses_dic['mean_auc']
			except:
				pass
			df_perf_result.to_csv(filename_hyp_tuning)
		# get quantile info
		df_test_result_comp.at[idx_res, 'best_epoch'] = model_info['best_epoch']
		df_test_result_comp.at[idx_res, 'mean_auc'] = np.round(test_stat_dic['mean_auc'],4); df_test_result_comp.at[idx_res, 'ibs'] = np.round(test_stat_dic['ibs'],4)
		df_test_result_comp.at[idx_res, 'bs 25 percentile'], df_test_result_comp.at[idx_res, 'bs 25 percentile se'] = quantile_perf_dic['point_ests']['bs'][0.25], quantile_perf_dic['bootstrap_ses']['bs'][0.25]
		df_test_result_comp.at[idx_res, 'bs 50 percentile'], df_test_result_comp.at[idx_res, 'bs 50 percentile se'] = quantile_perf_dic['point_ests']['bs'][0.50], quantile_perf_dic['bootstrap_ses']['bs'][0.50]
		df_test_result_comp.at[idx_res, 'bs 75 percentile'], df_test_result_comp.at[idx_res, 'bs 75 percentile se'] = quantile_perf_dic['point_ests']['bs'][0.75], quantile_perf_dic['bootstrap_ses']['bs'][0.75]

		df_test_result_comp.at[idx_res, 'auc 25 percentile'], df_test_result_comp.at[idx_res, 'auc 25 percentile se'] = quantile_perf_dic['point_ests']['auc'][0.25], quantile_perf_dic['bootstrap_ses']['auc'][0.25]
		df_test_result_comp.at[idx_res, 'auc 50 percentile'], df_test_result_comp.at[idx_res, 'auc 50 percentile se'] = quantile_perf_dic['point_ests']['auc'][0.50], quantile_perf_dic['bootstrap_ses']['auc'][0.50]
		df_test_result_comp.at[idx_res, 'auc 75 percentile'], df_test_result_comp.at[idx_res, 'auc 75 percentile se'] = quantile_perf_dic['point_ests']['auc'][0.75], quantile_perf_dic['bootstrap_ses']['auc'][0.75]
	else:
		# competing events :
		horizons = [0.25, 0.5, 0.75]
		events = ['event_' + str(idx + 1) for idx in np.arange(n_events)]
		metrics = ['ibs', 'mean_auc', 'bs 25 percentile', 'bs 25 percentile se', 'bs 50 percentile', 'bs 50 percentile se', 'bs 75 percentile', 'bs 75 percentile se', 'auc 25 percentile', 'auc 25 percentile se', 'auc 50 percentile', 'auc 50 percentile se', 'auc 75 percentile', 'auc 75 percentile se']
		cols = ['_'.join(element) for element in itertools.product(metrics, events)]
		# multi_cols = pd.MultiIndex.from_product([events, metrics], names=['events', 'metrics'])
		df_test_result_comp = pd.DataFrame([], index = [0], columns = ['param_set', 'best_epoch'] + cols)
		df_test_result_comp.index.name = 'result'

		df_test_result_comp.at[0, 'param_set'] 
		df_test_result_comp.at[0, 'best_epoch'] = model_info['best_epoch']
		# breakpoint()
		ef_surv_prob = surv_prob

		# store cs_cif_total and ef_surv_prob
		# with open('model_performance/' + run_id + '/test_cs_cif_total.npy', 'wb') as f:
		# 	np.save(f, cs_cif_total)
		# with open('model_performance/' + run_id + '/ef_surv_prob.npy', 'wb') as f:
		# 	np.save(f, ef_surv_prob)
		# breakpoint()
		for idx_, cs_cif_ in enumerate(cs_cif_total):
			print('---------------------------------------------------')
			print('outcome : ', idx_ + 1)
			# for model evaluation
			data_train_tuple = (model_info['events_info_train_tuple'][1] == idx_ + 1, model_info['events_info_train_tuple'][2], model_info['events_info_train_tuple'][3])
			data_test_tuple = (batch_dict['labels'] == idx_ + 1, batch_dict['remaining_time_to_event'], batch_dict['end_of_obs_idx'])
			# get performances at quantiles :
			test_stat_dic, sampled_ses_dic = get_performance_at_quantiles(cs_cif_, data_train_tuple, data_test_tuple, event_idx = idx_, run_id = run_id, sample_ids = np.asarray(batch_dict['sample_ids']), bootstrap = True, event_time_horizon = (min_event_time, max_pred_window, tp_res), dataset = dataset, n_events = n_events, horizons = horizons)
			# breakpoint()
			for metric, sub_dict in test_stat_dic.items():
				for quant in horizons:
					quant_str = str(int(quant*100))
					if metric == 'auc' or metric == 'bs':
						# df_test_result_comp.at[idx_, (quant, metrics)] = sub_dict[quant]
						# if metrics != 'num_samples':
						if metric == 'bs':
							df_test_result_comp.at[0, 'bs ' + quant_str + ' percentile_event_' + str(idx_+1)] = sub_dict[quant]
							df_test_result_comp.at[0, 'bs ' + quant_str + ' percentile se_event_' + str(idx_+1)] = sampled_ses_dic[metric][quant]
						elif metric == 'auc':
							df_test_result_comp.at[0, 'auc ' + quant_str + ' percentile_event_' + str(idx_+1)] = sub_dict[quant]
							df_test_result_comp.at[0, 'auc ' + quant_str + ' percentile se_event_' + str(idx_+1)] = sampled_ses_dic[metric][quant]
						# df_test_result_comp.at[idx_, (quant, metrics + '_95')] = conf_ints_dic[metrics][quant]
			"""
			Store the score : 
			"""
			df_test_result_comp.at[0, 'mean_auc_event_' + str(idx_+1)] = np.round(test_stat_dic['mean_auc'],4)
			df_test_result_comp.at[0, 'ibs_event_' + str(idx_+1)] = np.round(test_stat_dic['ibs'],4)
	# breakpoint()
	return df_test_result_comp
			
def remove_timepoints_wo_obs(batch_dict):
	# before computing the loss, remove the time points where there are no observations in this batch
	
	non_missing_tp = batch_dict["non_missing_tp"]
	batch_dict["observed_data"] = batch_dict["observed_data"][:, non_missing_tp]
	batch_dict["observed_mask"] = batch_dict["observed_mask"][:, non_missing_tp]
	batch_dict["observed_tp"] = batch_dict["observed_tp"][non_missing_tp]
	batch_dict["observed_tp_unnorm_enc"] = batch_dict["observed_tp_unnorm"][non_missing_tp]
	# batch_dict['non_missing_tp'] = non_missing_tp
	non_missing_tp_pred = batch_dict["non_missing_tp_pred"]
	batch_dict["data_to_predict"] = batch_dict["data_to_predict"][:, non_missing_tp_pred]
	batch_dict["observed_tp_unnorm_dec"] = batch_dict["observed_tp_unnorm"][non_missing_tp_pred]#batch_dict["observed_tp_unnorm"][non_missing_tp_pred]
	batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"][:, non_missing_tp_pred]
	return batch_dict

def create_perf_quantile_dict(quant_to_event_oi_train_test_surv_dict):
	quantiles_dict = {}; 
	for quant, (_, _, _, _, _, _) in quant_to_event_oi_train_test_surv_dict.items():
		quantiles_dict[quant] = []
	return quantiles_dict

def func_plot_survival_curves(surv_prob, cs_cif_total, labels, remaining_time_to_event, last_observed_points, n_events = 1, curr_epoch = None, filename_suffix = None):
	for idx_ in range(n_events):
		print('Event : ', str(idx_ + 1), ', Plotting 10 random survival curves...')
		fig, ax = plt.subplots()

		# get samples who experience events
		event_idx = [idx for idx, val in enumerate(labels.cpu().numpy()) if val[0] == idx_ + 1]
		if len(event_idx) > 5:
			random_choice_event = list(np.random.choice(event_idx, 5, replace = False))
		elif len(event_idx) > 0:
			random_choice_event = list(np.random.choice(event_idx, len(event_idx), replace = False))
		else:
			random_choice_event = []

		censored_idx = [idx for idx, val in enumerate(labels.cpu().numpy()) if val[0] == 0]
		random_choice_censored = list(np.random.choice(censored_idx, 5, replace = False))
		# if idx_ == 0:
		# 	surv_total = [] # this is just for test
		for j in random_choice_event + random_choice_censored:
			label = int(labels[j].cpu().numpy()[0])
			remaining_tte = int(remaining_time_to_event[j])
			try:
				last_obsved_time = int(last_observed_points[j].cpu().numpy())
			except:
				last_obsved_time = int(last_observed_points[j])
			# observed_tp_for_j = batch_dict['observed_tp_unnorm'][batch_dict['observed_mask'][j].sum(axis = 1) > 0]
			if n_events == 1:
				surv = surv_prob[j]
			else: # for multiple events we plot CIF
				# if label == 0:
				# 	surv = cs_cif_total[idx_][j] # plot cause-specific CIF#1 - surv_prob[j] # for censored plot 1 - event free survival
				# else:
				surv = cs_cif_total[idx_][j] # plot cause-specific CIF
			# time is referenced to 0 (prediction start time)
			t = np.arange(last_obsved_time, last_obsved_time + len(surv), 1) - last_obsved_time
			ax.plot(t, surv, label = 'Remaining Event time : ' + str(remaining_tte) + ', Event : ' + str(label) + ', Starting time : ' + str(last_obsved_time))# + ', Observed tps : ' + str(observed_tp_for_j))
			ax.set_ylim([0, 1])
		
		ax.set(xlabel='time (hours)', ylabel='Surv Prob' if n_events == 1 else 'CIF')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095))
		fig.savefig("surv_curves/" + filename_suffix + "/surv_curves_" + str(curr_epoch) + "_" + filename_suffix + "_event_" + str(idx_ + 1) + ".pdf", bbox_inches='tight')
		plt.close()

		# breakpoint()
	return

def get_cs_rmft_metric(cs_cif_cum, pred_window = 180, max_time = 700, last_observed_points_oi = None):
	"""
	cause specific restricted mean failture time (RMST)
	precisely this returns expected number of years lost before the end of pred_windows
	https://bmcmedresmethodol.biomedcentral.com/track/pdf/10.1186/s12874-021-01213-0.pdf
	"""
	if last_observed_points_oi is not None:
		try:
			max_time = max_time - int(np.max(last_observed_points_oi).cpu().numpy())
		except:
			max_time = max_time - int(np.max(last_observed_points_oi))
		# print('max_pred_window : ', max_pred_window)
		# breakpoint()
		# last_observed_points_oi
	else:
		max_pred_window = pred_window
	pred_window_oi = min(max_time, pred_window)

	cs_rmft = []
	for cs_cif_cum_per_sample in cs_cif_cum:
		cs_rmft.append(sum(cs_cif_cum_per_sample[0:pred_window_oi]))
	# breakpoint()
	return np.asarray(cs_rmft)


def _performance_at_quantiles(test_stat_dic, event_oi_train, event_oi, surv_metric_oi, test_events_oi, test_reamain_tte_oi, last_observed_points_oi, quant_to_event_oi_train_test_surv_dict = None, surv_metric_oi_cum = None, quant = None, horizons = None, quant_time = None, max_pred_window = None, time_varying_metric = False, remain_test_quantile_times = None, ef_surv = False, n_events = 1):

	# Get performance for censored and uncensored separately
	censored_idx = []; censored_time_list = []; uncensored_idx = []; uncensored_time_list = []
	for idx, (event_ind, remaining_tte) in enumerate(zip(test_events_oi.reshape(-1).cpu().numpy(), test_reamain_tte_oi)):
		if event_ind == 0:
			# cesnored
			censored_idx.append(idx)
			censored_time_list.append(remaining_tte)
		else:
			# uncesnored
			uncensored_idx.append(idx)
			uncensored_time_list.append(remaining_tte)
	
	# configure arrays for computing the eval metric
	# censored :
	event_oi_censored = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=len(censored_time_list))
	event_oi_censored['event'] = np.zeros(len(censored_time_list))
	event_oi_censored['time'] = censored_time_list

	# uncensored :
	event_oi_uncensored = np.empty(dtype=[('event', bool), ('time', np.float64)], shape=len(uncensored_time_list))
	event_oi_uncensored['event'] = np.ones(len(uncensored_time_list))
	event_oi_uncensored['time'] = uncensored_time_list

	if n_events == 1:
		bs = brier_score(event_oi_train, event_oi, surv_metric_oi, quant_time)[1][0]
		# when dealing with a single event, flip the surv prob to get CIF
		# In a single event, we get all the evaluation metrics of interest
		surv_metric_oi = np.asarray([1 - surv_ind for surv_ind in surv_metric_oi])
		surv_metric_oi_cum = [1 - surv_ind_cum for surv_ind_cum in surv_metric_oi_cum]
		auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, surv_metric_oi, quant_time)
	else:
		# note for multiple events, CIF is passed down
		bs = brier_score(event_oi_train, event_oi, 1-surv_metric_oi, quant_time)[1][0]
		# In multiple events, we get {c, auc, mean_auc} and {bs}, separately 
		auc, mean_auc = cumulative_dynamic_auc(event_oi_train, event_oi, surv_metric_oi, quant_time)

	# this dictionary is used for bootstrap
	quant_to_event_oi_train_test_surv_dict[quant] = (event_oi_train, event_oi, surv_metric_oi, quant_time, surv_metric_oi_cum if not ef_surv else None, last_observed_points_oi)
	# populate the dict with results
	test_stat_dic['bs'][quant] = bs
	test_stat_dic['auc'][quant] = auc[0]

	return test_stat_dic, quant_to_event_oi_train_test_surv_dict


def display_performance_at_quantiles(test_stat_dic, ef_surv = False, n_events = 1):
	print('Performance at quantiles : ')
	# print('num_samples : ', test_stat_dic['num_samples'])
	# if n_events == 1:
	for key, value in test_stat_dic['bs'].items():
		test_stat_dic['bs'][key] = np.round(value, 4)
	print('BS(t) at the percentiles : ', test_stat_dic['bs'])
	# print('bs censored : ', test_stat_dic['bs_censored'])
	# print('bs uncensored : ', test_stat_dic['bs_uncensored'])
	for key, value in test_stat_dic['auc'].items():
		test_stat_dic['auc'][key] = np.round(value, 4)
	print('AUC(t) at the percentiles : ', test_stat_dic['auc'])
	print('mean AUC(t) over 25-75 percentile : ', np.round(test_stat_dic['mean_auc'], 4))
	print('Integrated BS(t) over 25-75 percentile : ', np.round(test_stat_dic['ibs'], 4))
	return


def pre_process_data(data, data_info_dic, max_pred_window = None, n_events = 1, dataset = 'general', exclude_samples_wo_pred_window = True, random_seed = 0): 
		
	dat_cat = data[data_info_dic['feat_cat']].copy()
	# data preprocessing before missing values are replaced by 0
	# categorical variables : 0 -> -1
	for col in dat_cat.columns:
		dat_cat.loc[dat_cat[col] == 0, col] = -1
	# continuous variables : 0 -> 1e-3
	dat_num = data[data_info_dic['feat_cont']].copy()
	for col in dat_num.columns:
		dat_num.loc[dat_num[col] == 0, col] = 1e-3

	x1 = dat_cat.values
	x2 = dat_num.values
	x = np.hstack([x1, x2])
	feat_names = list(dat_cat.columns) + list(dat_num.columns)

	# modeling competing risks (mutually exclusive events) :
	time = data[data_info_dic['time_col']].values
	if n_events > 1:
		times = np.vstack((time, time)).T 
		events_ = data[data_info_dic['event_col']].values
		if max_pred_window is not None:
			durations_ = data[data_info_dic['time_to_event_col']].clip(upper=max_pred_window).values - times  # Clip tte wth max_pred_window
		else:
			durations_ = data[data_info_dic['time_to_event_col']].values - times  # Clip tte wth max_pred_window
		event = []; durations = []
		for events_entry, dur_entry in zip(events_, durations_):
			if sum(events_entry) == 0: # event free
				event.append(0)
				durations.append(min(dur_entry))
			elif sum(events_entry) == 1: # only one event happened
				event_oi = np.argmax(events_entry == 1)
				event.append(event_oi + 1)
				durations.append(dur_entry[event_oi])
			else: # multiple events happened : choose the event which happened the first
				min_idx = np.argmin(dur_entry)
				event.append(min_idx + 1) # competing event idx
				durations.append(dur_entry[min_idx]) 

				"""
				TODO : create a dictionary and only includes measurements up until the event of interest
				"""
		event = np.asarray(event)
		durations = np.asarray(durations)
		# breakpoint()
	else:
		event = data[data_info_dic['event_col']].values
		if max_pred_window is not None:
			durations = (data[data_info_dic['time_to_event_col']].clip(upper=max_pred_window) - data[data_info_dic['time_col']]).values  # Clip tte wth max_pred_window
		else:
			durations = (data[data_info_dic['time_to_event_col']] - data[data_info_dic['time_col']]).values

	ids_ = data[data_info_dic['id_col']].values
	sample_id_to_range_dic = {}; got_first_idx = False; prev_id_ = ids_[0]; start_idx = None; end_idx = None
	if n_events > 1:
		times_ = data[data_info_dic['time_col']].values
		for idx_, (id_, time_, dur_) in enumerate(zip(ids_, times_, durations)):
			if prev_id_ != id_:
				sample_id_to_range_dic[prev_id_] = (start_idx, end_idx)
				got_first_idx = False; start_idx = None; end_idx = None
			if not got_first_idx:
				start_idx = idx_
				got_first_idx = True
			if dur_ > 0 and got_first_idx: # only take into account observations before the event of interest
				end_idx = idx_ + 1
			else:
				end_idx = None
			prev_id_ = id_
			# idx_ += 1
		# for the last sample
		sample_id_to_range_dic[prev_id_] = (start_idx, end_idx)
		# breakpoint()
	else:
		for idx_, (id_, dur_) in enumerate(zip(ids_, durations)):
			if prev_id_ != id_:
				sample_id_to_range_dic[prev_id_] = (start_idx, end_idx)
				got_first_idx = False; start_idx = None; end_idx = None
			if not got_first_idx:
				start_idx = idx_
				got_first_idx = True
			if dur_ > 0 and got_first_idx:  # only take into account observations before the event of interest
				end_idx = idx_ + 1
			else:
				end_idx = None
			prev_id_ = id_
		# for the last sample
		sample_id_to_range_dic[prev_id_] = (start_idx, end_idx)
	# set missing values to 0; all zeroed values will be masked
	x_ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0).fit_transform(x)
	sampled_ids = sorted(set(data[data_info_dic['id_col']]))
	skipped_samples = []; # those who experience event at time t which corresponds to the last observed time (i.e. observed remaining tte = 0))
	if dataset == 'general':
		x, m, t, t_end, e, dur, ms = [], [], [], [], [], [], [] # where m is a mask for observed values, ms is a mask for hazard function
		x_ext, m_ext, t_ext, ids = [], [], [], []
		# sample_ids = sorted(list(set(data[data_info_dic['id_col']]))); 
		for id_ in tqdm(sampled_ids, desc = 'Pre-processing data...'):
			# if id_ in sampled_ids: # make sure id_ belongs to the sampled cohort
			ids.append(id_)
			start, end = sample_id_to_range_dic[id_]
			if exclude_samples_wo_pred_window:
				if end is None:
					skipped_samples.append(id_)
					continue
			else:
				assert end is not None, f"tte value has to be greater than the lastest observation time for ID: {id_}"
			x.append(x_[start:end])
			m.append((x_[start:end] != 0)*1)
			t.append(time[start:end])
			t_end.append(time[end-1])
			if max_pred_window is not None: # perform administrative censoring
				last_obs_time = int(time[end-1]) 
				dur_from_last_obs = int(durations[end-1])
				dur_from_init_obs = int(durations[start])
				remain_dur = int(max_pred_window - dur_from_last_obs - last_obs_time) 
				if remain_dur <= 0:
					e.append(0)
					remain_dur = max(0, int(max_pred_window - dur_from_last_obs - last_obs_time))
					dur_from_last_obs = min(dur_from_last_obs, int(max_pred_window - last_obs_time))  # Make sure dur_from_last_obs >= 0
					dur.append(durations[start:end] - (dur_from_init_obs - max_pred_window)) # administratively censor samples at the max prediction time window
				else:
					e.append(event[start:end][-1]) 
					dur.append(durations[start:end])
				# mask for survival such that hazards from the latest measurement for each sample are incorporated into the loss
				ms_ = list(np.zeros(last_obs_time, dtype = bool)) + list(np.ones(dur_from_last_obs, dtype = bool)) + list(np.zeros(remain_dur, dtype = bool))
				ms.append(ms_)
			else:
				ms.append(0)  
				e.append(event[start:end][-1])
	else:
		raise NotImplementedError
	# print('Complete!')
	# print('\n')
	print('excluded samples due to event times overlapping last observation times (i.e. remaining t_i = 0) : ', skipped_samples)
	print('n = ', len(skipped_samples))
	return ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, max(t_end)

def get_data_obj_merged(data_train, data_valid, data_info_dic, n_samples = None, max_pred_window = None, n_events = 1, dataset = 'general', random_seed = 0, device = None, feat_reconstr = None, process_eval_set = False, check_extrapolation = False, param_dics = None, min_max_tuple = None, max_obs_time = None, exclude_samples_wo_pred_window = True):

	if process_eval_set:
		eval_data = []
		ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, _ = utils.pre_process_data(data_valid, data_info_dic, max_pred_window = max_pred_window, n_events = n_events, random_seed = random_seed, exclude_samples_wo_pred_window=exclude_samples_wo_pred_window)
		# breakpoint()
		if check_extrapolation:
			for id_, x_, x_ext_, m_, m_ext_, ms_, t_, t_ext_, e_, dur_ in zip(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur):
				eval_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(t_ext_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(x_ext_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(m_ext_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
		else:
			for id_, x_, m_, ms_, t_, e_, dur_ in zip(ids, x, m, ms, t, e, dur):
				eval_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
	else:
		train_data, valid_data = [], []
		if n_events > 1: train_events = []
		if check_extrapolation:
			# train
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, max_obs_time = utils.pre_process_data(data_train, data_info_dic, max_pred_window = max_pred_window, n_events = n_events, random_seed = random_seed, exclude_samples_wo_pred_window=exclude_samples_wo_pred_window)
			for id_, x_, x_ext_, m_, m_ext_, ms_, t_, t_ext_, e_, dur_ in zip(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur):
				train_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(t_ext_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(x_ext_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(m_ext_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
			# valid
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, _ = utils.pre_process_data(data_valid, data_info_dic, max_pred_window = max_pred_window, n_events = n_events, random_seed = random_seed, exclude_samples_wo_pred_window=exclude_samples_wo_pred_window)
			for id_, x_, x_ext_, m_, m_ext_, ms_, t_, t_ext_, e_, dur_ in zip(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur):
				valid_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(t_ext_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(x_ext_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(m_ext_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
		else:
			# train
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, max_obs_time = utils.pre_process_data(data_train, data_info_dic, max_pred_window = max_pred_window, n_events = n_events, random_seed = random_seed, exclude_samples_wo_pred_window=exclude_samples_wo_pred_window)
			for id_, x_, m_, ms_, t_, e_, dur_ in zip(ids, x, m, ms, t, e, dur):
				train_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
				if n_events > 1: train_events.append(e_)
			# valid
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, _ = utils.pre_process_data(data_valid, data_info_dic, max_pred_window = max_pred_window, n_events = n_events, random_seed = random_seed, exclude_samples_wo_pred_window=exclude_samples_wo_pred_window)
			for id_, x_, m_, ms_, t_, e_, dur_ in zip(ids, x, m, ms, t, e, dur):
				valid_data.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))

		# for comepting events get event ratios dict : weight for the event with most frequent occurrences gets one and other events are accordingly up-weighted
		# this may be used for weighting the survival loss accordingly
		if n_events > 1:
			train_events = np.asarray(train_events)
			event_to_event_weight_dict = {}
			total_events = (train_events != 0).sum()
			for idx_event in range(n_events):
				event_to_event_weight_dict[idx_event + 1] = total_events - (train_events == idx_event + 1).sum()
			min_event_weight = min(event_to_event_weight_dict.values())
			sum_weights = sum(event_to_event_weight_dict.values())
			for event, weight in event_to_event_weight_dict.items():
				event_to_event_weight_dict[event] = weight/min_event_weight
			# breakpoint()
	
	feat_reconstr_idx = [feat_names.index(feat) for feat in feat_reconstr] if feat_reconstr is not None else None
	if not process_eval_set: # this needs to be removed for a practical implementation
		# np.random.seed(random_seed)
		torch.manual_seed(random_seed)
		
		# train_data, valid_data = model_selection.train_test_split(total_dataset, train_size= 0.8125, random_state = 42, shuffle = True)

		if check_extrapolation:
			record_id, tt, _, vals, _, mask, _, mask_surv, labels, dur = train_data[0]
		else:
			record_id, tt, vals, mask, mask_surv, labels, dur = train_data[0]

		n_samples = len(train_data)
		input_dim = vals.size(-1)

		batch_size = min(min(len(train_data), param_dics['batch_size']), n_samples)
		data_min, data_max = utils.get_data_min_max(train_data, dataset = dataset, device = device, check_extrapolation = check_extrapolation)

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, device, data_type = "train",
			data_min = data_min, data_max = data_max, dataset = dataset, max_pred_window = max_pred_window, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = check_extrapolation, max_obs_time = max_obs_time))
		
		train_dataloader_full_batch = DataLoader(train_data, batch_size= len(train_data), shuffle=False, 
			collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, device, data_type = "train",
			data_min = data_min, data_max = data_max, dataset = dataset, max_pred_window = max_pred_window, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = check_extrapolation, max_obs_time = max_obs_time))

		valid_dataloader = DataLoader(valid_data, batch_size = len(valid_data), shuffle=False, 
			collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, device, data_type = "test",
			data_min = data_min, data_max = data_max, dataset = dataset, max_pred_window = max_pred_window, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = check_extrapolation, max_obs_time = max_obs_time))


		data_objects = {"train_dataloader": utils.inf_generator(train_dataloader), 
						"train_dataloader_full_batch": utils.inf_generator(train_dataloader_full_batch),
						"valid_dataloader": utils.inf_generator(valid_dataloader),
						"input_dim": input_dim,
						"n_train_batches": len(train_dataloader),
						"n_valid_batches": len(valid_dataloader),
						"attr": feat_names if feat_reconstr is None else feat_reconstr} #optional
		if n_events > 1:
			return data_objects, (data_min, data_max), max_obs_time, event_to_event_weight_dict
		else:
			return data_objects, (data_min, data_max), max_obs_time, None
	elif process_eval_set:
		# for test set 
		record_id, tt, vals, mask, mask_surv, labels, dur = eval_data[0]
		n_samples = len(eval_data)
		input_dim = vals.size(-1)

		# use min-max values from the training sets
		data_min, data_max = min_max_tuple

		test_dataloader = DataLoader(eval_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, device, data_type = "test",
			data_min = data_min, data_max = data_max, dataset = dataset, max_pred_window = max_pred_window, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = check_extrapolation, max_obs_time = max_obs_time))

		data_objects = {"test_dataloader": utils.inf_generator(test_dataloader),
						"input_dim": input_dim,
						"n_test_batches": len(test_dataloader),
						"attr": feat_names if feat_reconstr is None else feat_reconstr} #optional

		return data_objects

