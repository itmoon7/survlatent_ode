####################################################
# SurvLatent ODE
# Author : Intae Moon
####################################################

import os
import sys
import pickle
import numpy as np
import pandas as pd
import math

import lib.utils as utils
from sklearn.impute import SimpleImputer
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence, Independent

from lib.latent_ode import LatentODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver
from lib.ode_func import ODEFunc

from tqdm import tqdm
import pickle

class LatentODESub(LatentODE):  
	def __init__(self, latent_dim = 20, rec_dim = 40, rec_layers = 3, gen_layers = 3, units_ode = 50, units_gru = 50, input_dim = 20, reconstr_dim = 20, device = None, gru_aug = False, attention_aug = False, attn_num_heads = 4, n_events = 1, temporal_encoding = False, num_layer_hazard_dec = 2, mult_event_units = 5):
		
		# create components for Latent ODE
		ode_func_net = utils.create_net(latent_dim, latent_dim, 
			n_layers = gen_layers, n_units = units_ode, nonlinear = nn.Tanh)

		gen_ode_func = ODEFunc( # generative ode layer
			input_dim = input_dim, 
			latent_dim = latent_dim, 
			ode_func_net = ode_func_net,
			device = device).to(device)

		enc_input_dim = int(input_dim) * 2 # for mask concatenation
		ode_func_net = utils.create_net(rec_dim, rec_dim, 
			n_layers = rec_layers, n_units = units_ode, nonlinear = nn.Tanh)

		# recognition ode layer
		rec_ode_func = ODEFunc( 
			input_dim = enc_input_dim, 
			latent_dim = rec_dim,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", latent_dim, odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		encoder_z0 = Encoder_z0_ODE_RNN(rec_dim, enc_input_dim, z0_diffeq_solver, z0_dim = latent_dim, n_gru_units = units_gru, device = device).to(device)
		
		diffeq_solver = DiffeqSolver(input_dim, gen_ode_func, 'dopri5', latent_dim, odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

		decoder = Decoder(latent_dim, reconstr_dim).to(device)
		if n_events == 1:
			decoder_surv = Decoder(latent_dim, None, surv_est = self.surv_est).to(device)
		elif n_events == 2:
			decoder_surv = []
			decoder_surv_1 = Decoder(latent_dim, None, surv_est = self.surv_est, n_events = n_events, num_layer = num_layer_hazard_dec, mult_event_units = mult_event_units).to(device)
			decoder_surv_2 = Decoder(latent_dim, None, surv_est = self.surv_est, n_events = n_events, num_layer = num_layer_hazard_dec, mult_event_units = mult_event_units).to(device)
		else:
			raise NotImplementedError
		
		obsrv_std = 1.0 # 0.01 orig; maybe try to do 0.1, 0.5
		obsrv_std = torch.Tensor([obsrv_std]).to(device)
		z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

		self.n_events = n_events
		self.device = device
		self.latent_dim = latent_dim
		self.rec_dim = rec_dim
		self.rec_layers = rec_layers
		self.gen_layers = gen_layers
		self.units_ode = units_ode
		self.units_gru = units_gru
		self.test_batch_dict = None

		self.pred_y_avg = None
		self.pred_y_var = None
		self.attention_aug = attention_aug
		self.attn_num_heads = attn_num_heads
		self.temporal_encoding = temporal_encoding

		self.mult_event_units = mult_event_units

		super().__init__(input_dim = input_dim, 
						latent_dim = latent_dim, 
						encoder_z0 = encoder_z0, 
						decoder = decoder, 
						decoder_surv = decoder_surv if n_events ==1 else (decoder_surv_1, decoder_surv_2),
						diffeq_solver = diffeq_solver, 
						z0_prior = z0_prior, 
						device = device,
						obsrv_std = obsrv_std, 
						gru_aug = gru_aug,
						n_gru_units = units_gru)

	def get_min_max_data(self):
		return self.min_max_tuple

	def process_eval_data(self, data, data_info_dic, batch_size = 100, max_pred_window = None, n_samples = None, dataset = 'general', run_id = None, include_test_set = False, feat_reconstr = None, check_extrapolation = False, model_info = None, random_seed = 0):
		
		self.random_seed = random_seed
		self.check_extrapolation = check_extrapolation
		self.dataset = dataset
		# load training data stats
		self.min_max_tuple = model_info['min_max_data_tuple']
		self.max_obs_time = model_info['max_obs_time']

		# min_max_tuple_to_save = (self.min_max_tuple[0].numpy(), self.min_max_tuple[1].numpy())
		# np.save('min_max_tuple_at_eval.npy', min_max_tuple_to_save)
		# breakpoint()
		
		param_dics = {}
		param_dics['batch_size'] = batch_size

		# torch.manual_seed(random_seed)
		# np.random.seed(random_seed)
		data_obj = utils.get_data_obj_merged(None, data, data_info_dic, process_eval_set = True, device = self.device, feat_reconstr = feat_reconstr, max_pred_window = max_pred_window, n_events = self.n_events, random_seed = self.random_seed, param_dics = param_dics, min_max_tuple = self.min_max_tuple, max_obs_time = self.max_obs_time)

		return utils.remove_timepoints_wo_obs(utils.get_next_batch(data_obj["test_dataloader"]))

	def get_test_data_dic(self, filename_suffix = None):
		"""
		Returns test_batch_dict which includes info about test data.
		If self.test_batch_dict not available, load the locally stored version and return it
		"""
		if self.test_batch_dict is not None:
			return self.test_batch_dict
		else:
			with open('model_performance/' + filename_suffix + '/test_batch_dict.pkl', 'rb') as handle:
				self.test_batch_dict = pickle.load(handle)
			return self.test_batch_dict
		

	def fit(self, data_train, data_valid, data_info_dic, device = None, n_epochs = 30, batch_size = 50, lr = 1e-2, max_pred_window = None, n_samples = None, run_id = None, surv_loss_scale = 10, n_latent_traj = 1, early_stopping = False, survival_loss_exp = True, train_info = None, feat_reconstr = None, check_extrapolation = False, wait_until_full_surv_loss = 15, dataset = 'general', random_seed = 0):
		# check extrapolation
		self.check_extrapolation = check_extrapolation
		self.dataset = dataset
		self.random_seed = random_seed

		# set random seed and experiment ID
		experimentID = None
		save_path = 'experiments/'

		ckpt_path = os.path.join(save_path, "run_" + str(run_id) + '.ckpt')

		# set params
		param_dics = {}
		param_dics['n_samples'] = n_samples
		param_dics['batch_size'] = batch_size
		param_dics['lr'] = lr
		param_dics['niters'] = n_epochs
		param_dics['latent_dim'] = self.latent_dim
		param_dics['rec_dim'] = self.rec_dim
		param_dics['rec_layers'] = self.rec_layers
		param_dics['gen_layers'] = self.gen_layers
		param_dics['units_ode'] = self.units_ode
		param_dics['units_gru'] = self.units_gru
		param_dics['attn_num_heads'] = self.attn_num_heads

		data_obj, self.min_max_tuple, self.max_obs_time = utils.get_data_obj_merged(data_train, data_valid, data_info_dic, device = self.device, feat_reconstr = feat_reconstr, max_pred_window = max_pred_window, n_events = self.n_events, random_seed = self.random_seed, param_dics = param_dics)
		
		batch_dict_valid = utils.remove_timepoints_wo_obs(utils.get_next_batch(data_obj["valid_dataloader"]))
		# f = open('valid_dataloader_in_training.pkl', "wb") # prev : ckp_sig_feats_dic_Jan_21th_2021_binary_wo_duplicates_thresh_0_10, ckp_sig_feats_dic_Nov_13th_binary_mut_burden, ckp_sig_feats_dic_Nov_13th_binary, ckp_sig_feats_dic_Sep_25th_binary
		# pickle.dump(batch_dict_valid,f)
		# f.close()
		# min_max_tuple_to_save = (self.min_max_tuple[0].numpy(), self.min_max_tuple[1].numpy())
		# np.save('min_max_tuple_at_train.npy', min_max_tuple_to_save)
		# breakpoint()
		param_dics['input_dim'] = data_obj["input_dim"]
		utils.train_surv_model(self, data_obj, param_dics, 
							   device = self.device, surv_est = self.surv_est, 
							   max_pred_window = max_pred_window, run_id = run_id, dataset = dataset, survival_loss_scale = surv_loss_scale, n_latent_traj = n_latent_traj, early_stopping = early_stopping, survival_loss_exp = survival_loss_exp, train_info = train_info, n_events = self.n_events, wait_until_full_surv_loss = wait_until_full_surv_loss)
		return

	def get_surv_prob(self, batch_dict, model_info = None, max_pred_window = None, filename_suffix = None, device = None, test_batch_size = 200, reconstr_loss = False, n_events = 1):
		
		last_observed_points = batch_dict['end_of_obs_idx'] # auxiliary info for plotting
		num_samples = len(batch_dict['sample_ids'])
	
		batch_total_observed_data = utils.divide_list(batch_dict["observed_data"], test_batch_size)
		batch_total_observed_mask = utils.divide_list(batch_dict["observed_mask"], test_batch_size)
		batch_total_end_obs_idx = utils.divide_list(batch_dict["end_of_obs_idx"], test_batch_size)
		
		init_flag = True; count = 0;
		for i in range(len(batch_total_observed_data)):
			pred_y_, hazards_y, info = self.get_reconstruction_survival(batch_dict["tp_to_predict"], 
																batch_total_observed_data[i], batch_dict["observed_tp"], batch_total_end_obs_idx[i], 
																mask = batch_total_observed_mask[i], get_latent_hazard = True, temporal_encoding = self.temporal_encoding)
			latent_states = info['latent_hazard']
			if init_flag:
				init_flag = False
			else:
				hazards_y = torch.cat((prev_hazards_y, hazards_y), 1)
				pred_y_ = torch.cat((prev_pred_y, pred_y_), 1)
				latent_states = prev_latent_states + latent_states
			prev_pred_y = pred_y_.detach()
			prev_latent_states = latent_states
			prev_hazards_y = hazards_y.detach()	
		
		pred_y = pred_y_[:, :, :batch_dict["max_end_of_obs_idx"] + 1, :] #, pred_y[:, :, batch_dict["max_end_of_obs_idx"]:, :]
		pred_y = torch.index_select(pred_y, 2, batch_dict['observed_tp_unnorm_dec'].int())#[:, :, batch_dict['non_missing_tp_pred'], :] 
		
		if n_events == 1:
			surv_prob = self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_pred_window = max_pred_window, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'], n_events = n_events)
			return surv_prob
		else:	
			ef_surv_prob, cs_cif_total = self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_pred_window = max_pred_window, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'], n_events = n_events)
			return ef_surv_prob, cs_cif_total

	def get_reconst_traj(self, batch_dict = None, n_latent_traj = 1, credible_interval = False):
		"""

		"""
		if credible_interval and n_latent_traj == 1:
			raise ValueError('n_latent_traj must be greater than 1 to obtain credible interval')
		elif not credible_interval and n_latent_traj > 1:
			raise ValueError('Must set credible_interval to True when evaluating more than one trajectory')

		if self.pred_y_avg is not None and credible_interval:
			return self.pred_y_avg, self.pred_y_var

		if batch_dict is None:
			# convert data into dataloader format 
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = self._pre_process_data(data, data_info_dic, max_pred_window = max_pred_window, dataset = dataset)

			data_obj, _ = self._get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, None, 
											 dataset = dataset, device = self.device, max_pred_window = max_pred_window, min_max_tuple = model_info['min_max_data_tuple'], process_test_set = True)

			# get hazards and prediction
			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
			self.test_batch_dict = batch_dict

		# compute surv prob
		last_observed_points = [] # auxiliary info for plotting
		num_samples = len(batch_dict['sample_ids'])
		for j in range(num_samples):
			last_observed_point = batch_dict['observed_tp_unnorm'][batch_dict['observed_mask'][j].sum(axis = 1) > 0][-1]
			last_observed_points.append(last_observed_point)

		if n_latent_traj > 1:
			"""
			Use Welford online algorithm to compute variance
			"""
			for i in tqdm(range(n_latent_traj), desc = 'getting ' + str(n_latent_traj) + ' trajectroies...'):
				pred_y, _, info = self.get_reconstruction_survival(batch_dict["tp_to_predict"], 
																batch_dict["observed_data"], batch_dict["observed_tp"], 
																mask = batch_dict["observed_mask"])
				
				pred_y_curr = pred_y[0].detach().numpy()
				if i == 0:
					m2n_pred_y = np.zeros(np.shape(pred_y_curr))
					pred_y_var = np.zeros(np.shape(pred_y_curr))
					pred_y_avg = pred_y_curr
				else:
					pred_y_avg_prev = pred_y_avg.copy()
					pred_y_avg = pred_y_avg + (pred_y_curr - pred_y_avg)/(i + 1)
		
					m2n_pred_y = m2n_pred_y + (pred_y_curr - pred_y_avg_prev) * (pred_y_curr - pred_y_avg)
					pred_y_var = m2n_pred_y/i # unbiased estimate
			# update reconst traj
			self.pred_y_avg = pred_y_avg
			self.pred_y_var = pred_y_var

			return pred_y_avg, pred_y_var
		else:
			pred_y, _, info = self.get_reconstruction_survival(batch_dict["tp_to_predict"], 
																batch_dict["observed_data"], batch_dict["observed_tp"], 
																mask = batch_dict["observed_mask"])
			breakpoint()
			# get pred up until last observ idx and afterwards
			# if batch_dict["end_of_obs_idx"] is None:
			# 	pred_y_extr = None
			# else:
			# 	pred_y, pred_y_extr = pred_y[:, :, :batch_dict["end_of_obs_idx"], :], pred_y[:, :, batch_dict["end_of_obs_idx"]:, :]
			return pred_y #self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_pred_window = max_pred_window, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'])
		# return
	# def get_feat_names():
	# 	if self.feat_names is not None:
	# 		return self.feat_names

class SurvLatentODE_Cox(LatentODESub):
	def __init__(self):
		raise NotImplementedError

class SurvLatentODE(LatentODESub):
	def __init__(self, dec_latent_dim = 20, enc_latent_dim = 40, enc_f_nn_layers = 3, dec_g_nn_layers = 3, num_units_ode = 50, num_units_gru = 50, input_dim = 20, reconstr_dim = None, device = None, gru_aug = False, attention_aug = False, attn_num_heads = 4, n_events = 1, temporal_encoding = False, mult_event_units = 5, haz_dec_layers = 2):
		self.surv_est = 'Hazard'
		super().__init__(latent_dim = dec_latent_dim,
						 rec_dim = enc_latent_dim,
						 rec_layers = enc_f_nn_layers,
						 gen_layers = dec_g_nn_layers,
						 units_ode = num_units_ode,
						 units_gru = num_units_gru,
						 device = device,
						 input_dim = input_dim, 
						 gru_aug = gru_aug,
						 attention_aug = attention_aug, 
						 attn_num_heads = attn_num_heads,
						 temporal_encoding = temporal_encoding,
						 n_events = n_events, 
						 mult_event_units = mult_event_units, 
						 num_layer_hazard_dec = haz_dec_layers,
						 reconstr_dim = reconstr_dim if reconstr_dim is not None else input_dim)

	def _get_surv_prob(self, hazards_y, batch_dict, last_observed_points, cred_interval = False, max_pred_window = None, filename_suffix = None, events_info_train_tuple = None, n_events = 1):
		results = {'hazards_y':hazards_y}
		return utils.compute_survival_curves(results, batch_dict, None, last_observed_points, surv_est = self.surv_est, n_events = n_events)

	def _get_reconst_traj(self, data):
		pass
