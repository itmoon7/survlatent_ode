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

	def _pre_process_data(self, data, data_info_dic, n_samples = None, max_time = None, pred_from_sequence = True, include_test_set = False): # we would not want to have mimic dataset indicator. just excpetion for speed
		
		dat_cat = data[data_info_dic['feat_cat']]
		# for gender (Men : 'M', Women : 'F') -> (Men : 1, Women : -1)
		# everything else (1, 0) -> (1, -1)
		for col in dat_cat.columns:
			df_oi = dat_cat[col]
			df_oi.loc[df_oi.values == 0, ] = -1

		dat_num = data[data_info_dic['feat_cont']]
		# change 0 to very small num so they don't get confused with missing values
		for col in dat_num.columns:
			df_oi = dat_num[col]
			df_oi.loc[df_oi.values == 0, ] = 1e-3

		x1 = dat_cat.values
		x2 = dat_num.values
		x = np.hstack([x1, x2])
		feat_names = list(dat_cat.columns) + list(dat_num.columns)

		# in the case of competing risks (mutually exclusive events) :
		time = data[data_info_dic['time_col']].values
		if self.n_events > 1:
			# self.n_events = len(data_info_dic['event_col'])
			times = np.vstack((time, time)).T # create two identical cop
			events_ = data[data_info_dic['event_col']].values
			durations_ = data[data_info_dic['time_to_event_col']].values - times
			event = []; durations = []
			# breakpoint()
			for events_entry, dur_entry in zip(events_, durations_):
				# no event happened
				if sum(events_entry) == 0:
					event.append(0)
					durations.append(min(dur_entry))
				# only one event happened
				elif sum(events_entry) == 1: 
					event_oi = np.argmax(events_entry == 1)
					event.append(event_oi + 1)
					durations.append(dur_entry[event_oi])
				# multiple events happened : choose the event which happened the first
				else:
					min_idx = np.argmin(dur_entry)
					event.append(min_idx + 1) # competing event idx.
					durations.append(dur_entry[min_idx]) 
			event = np.asarray(event)
			durations = np.asarray(durations)
		else:
			# self.n_events = 1
			event = data[data_info_dic['event_col']].values
			durations = (data[data_info_dic['time_to_event_col']] - data[data_info_dic['time_col']]).values

		ids_ = data[data_info_dic['id_col']].values
		sample_id_to_range_dic = {}; got_first_idx = False; prev_id_ = ids_[0]; start_idx = None; end_idx = None
		for idx_, id_ in enumerate(ids_):
			if prev_id_ != id_:
				sample_id_to_range_dic[prev_id_] = (start_idx, idx_)
				# if start_idx == idx_:
				# 	breakpoint()
				got_first_idx = False; start_idx = None; end_idx = None
			if not got_first_idx:
				start_idx = idx_
				got_first_idx = True
			prev_id_ = id_
		# for the last sample
		sample_id_to_range_dic[prev_id_] = (start_idx, idx_ + 1)
		# set missing values to 0; all zeroed values will be masked
		x_ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0.0).fit_transform(x)

		unique_ids = list(set(data[data_info_dic['id_col']].values))
		if n_samples is not None:
			sampled_ids = np.random.choice(list(unique_ids), replace = False, size = n_samples if n_samples < len(list(unique_ids)) else len(list(unique_ids)))
		else:
			sampled_ids = unique_ids

		# breakpoint()
		if self.dataset == 'general':
			# pass
			if pred_from_sequence:
				with open('dvt_data/mrn_to_sequence_time_dict.pkl', 'rb') as pkl_file:
					mrn_to_sequence_time_dict = pickle.load(pkl_file)
			x, m, t, t_end, e, dur, ms = [], [], [], [], [], [], [] # where m is a mask for observed values, ms is a mask for survival from the last observation
			x_ext, m_ext, t_ext = [], [], []
			sample_ids = sorted(list(set(data[data_info_dic['id_col']]))); sample_ids_inc = []
			for id_ in tqdm(sample_ids, desc = 'Iterating through each sample...'):
				if id_ in sampled_ids: # make sure id_ belongs to the sampled cohort
					sample_ids_inc.append(id_)
					start, end = sample_id_to_range_dic[id_]
					
					x.append(x_[start:end])
					m.append((x_[start:end] != 0)*1)
					t.append(time[start:end])
					t_end.append(time[end-1])
					# breakpoint()
					if max_time is not None: # perform administrative censoring
						if pred_from_sequence:
							last_obs_time = int(mrn_to_sequence_time_dict[id_])
						else:
							last_obs_time = int(time[end-1]) # get sequence date as the last observation time
						dur_from_last_obs = int(durations[end-1])
						dur_from_init_obs = int(durations[start])
						remain_dur = int(max_time - dur_from_last_obs - last_obs_time) # this makes sense since we're getting prediction from the lastest observation
						if remain_dur <= 0:
							e.append(0)
							remain_dur = max(0, int(max_time - dur_from_last_obs - last_obs_time))
							dur_from_last_obs = min(dur_from_last_obs, int(max_time - last_obs_time))
							dur.append(durations[start:end] - (dur_from_init_obs - max_time)) # administratively censor samples at the max time from last observation
						else:
							e.append(event[start:end][-1]) # since event is specified across all entries, just grab the last one
							dur.append(durations[start:end])
						# mask from the initial observation
						# breakpoint()
						ms_ = list(np.zeros(last_obs_time, dtype = bool)) + list(np.ones(dur_from_last_obs, dtype = bool)) + list(np.zeros(remain_dur, dtype = bool))
						# if sum(ms_) == 0:
						# 	breakpoint()
						ms.append(ms_)
						# if id_ == 780216:
						# 	breakpoint()
					else:
						ms.append(0)  
						e.append(event[start:end][-1])

		elif self.dataset == 'mimic':
			if include_test_set:
				with open('../neural_ode_surv/mimic_data/sample_id_to_range_dic_full_meas_more_labs_36_hours.pkl', 'rb') as handle: # sample_id_to_range_dic_full_meas, sample_id_to_range_dic_full_meas_compet_risks_v4
					sample_id_to_range_dic = pickle.load(handle)
			# with open('./mimic_data/sample_id_to_range_dic_full_meas.pkl', 'rb') as handle: # sample_id_to_range_dic_full_meas, sample_id_to_range_dic_full_meas_compet_risks_v4
			# 	sample_id_to_range_dic = pickle.load(handle)
			x, m, t, t_end, e, dur, ms = [], [], [], [], [], [], [] # where m is a mask for observed values, ms is a mask for survival from the last observation
			x_ext, m_ext, t_ext = [], [], []
			sample_ids = sorted(list(set(data[data_info_dic['id_col']]))); sample_ids_inc = []
			for id_ in tqdm(sample_ids, desc = 'Iterating through each sample...'):
				# if n_samples is not None:
				if id_ in sampled_ids:
					sample_ids_inc.append(id_)
					if self.check_extrapolation:
						start, end, end_ext = sample_id_to_range_dic[id_]
					else:
						start, end = sample_id_to_range_dic[id_]
					x.append(x_[start:end])
					m.append((x_[start:end] != 0)*1)
					t.append(time[start:end])
					t_end.append(time[end - 1])
					# breakpoint()
					# get extra information beyond observation window
					if self.check_extrapolation:
						x_ext.append(x_[end:end_ext])
						m_ext.append((x_[end:end_ext] != 0)*1)
						t_ext.append(time[end:end_ext])
					# breakpoint()
					if max_time is not None: # perform administrative censoring
						last_obs_time = int(time[end-1])
						dur_from_last_obs = int(durations[end-1])
						dur_from_init_obs = int(durations[start])
						remain_dur = int(max_time - dur_from_last_obs - last_obs_time) # this makes sense since we're getting prediction from the lastest observation
						if remain_dur <= 0:
							e.append(0)
							remain_dur = max(0, int(max_time - dur_from_last_obs - last_obs_time))
							dur_from_last_obs = min(dur_from_last_obs, int(max_time - last_obs_time))
							# dur.append(durations[start:end])
							# breakpoint()
							dur.append(durations[start:end] - (dur_from_init_obs - max_time)) # administratively censor samples at the max time from last observation
							# print(durations[start:end] - (dur_from_init_obs - max_time))
							# breakpoint()
						else:
							# breakpoint()
							e.append(event[start:end][-1]) # since event is specified across all entries, just grab the last one
							dur.append(durations[start:end])
						# mask from the initial observation
						ms_ = list(np.zeros(last_obs_time, dtype = bool)) + list(np.ones(dur_from_last_obs, dtype = bool)) + list(np.zeros(remain_dur, dtype = bool))
						# ms_ = list(np.ones(dur_from_last_obs, dtype = bool)) + list(np.zeros(remain_dur, dtype = bool))
						# breakpoint()
						ms.append(ms_)
					else:
						ms.append(0)  
						e.append(event[start:end][-1])
		print('Complete!')
		print('\n')
		self.max_obs_time = max(t_end)
		# breakpoint()
		return sample_ids_inc, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names

	def _get_data_obj(self, ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, param_dics, process_test_set = False, device = None, max_time = None, min_max_tuple = None, include_test_set = False, feat_reconstr = None, test_restrict = None):

		# splitting the data into train, test, and validation sets
		n = len(x)
		# get indices for feats to reconstruct
		feat_reconstr_idx = [feat_names.index(feat) for feat in feat_reconstr] if feat_reconstr is not None else None
		# breakpoint()
		# total of 18 features
		total_dataset = []
		# breakpoint()
		if self.check_extrapolation:
			for id_, x_, x_ext_, m_, m_ext_, ms_, t_, t_ext_, e_, dur_ in zip(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur):
				total_dataset.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(t_ext_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(x_ext_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(m_ext_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))
		else:
			for id_, x_, m_, ms_, t_, e_, dur_ in zip(ids, x, m, ms, t, e, dur):
				total_dataset.append((str(id_), torch.tensor(t_, dtype = torch.float), torch.tensor(x_, dtype = torch.float), torch.tensor(m_, dtype = torch.float), torch.tensor(ms_, dtype = torch.float), torch.tensor(e_, dtype = torch.float), torch.tensor(dur_, dtype = torch.float)))

		# breakpoint()
		# total_time_points = 
		# i) create the dataloader compatible with the ode-rnn framework
		# Shuffle and split
		# train 55, valid 15, test 30
		if include_test_set: # this needs to be removed for a practical implementation
			if self.dataset == 'mimic':
				train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.7, random_state = 42, shuffle = True)
				train_data, valid_data = model_selection.train_test_split(train_data, train_size= 0.7857, random_state = 33, shuffle = True) #0.7857
				# print('general dataset')
			elif test_restrict == 'non_khorana': # change as you want it 
				with open('dvt_data/khorana_mrns.npy', 'rb') as f:
					khorana_mrns = np.load(f)
				# breakpoint()
				print('non_khorana')
				test_data = []; other_data = []
				for id_, x_, m_, ms_, t_, e_, dur_ in total_dataset:
					if int(id_) in khorana_mrns:
						test_data.append((id_, x_, m_, ms_, t_, e_, dur_))
					else:
						other_data.append((id_, x_, m_, ms_, t_, e_, dur_))
				# train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.7, random_state = 42, shuffle = True)
				train_data, valid_data = model_selection.train_test_split(other_data, train_size= 0.7857, random_state = 33, shuffle = True) #0.7857
				# breakpoint()
			else:
				train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.7, random_state = 42, shuffle = True)
				train_data, valid_data = model_selection.train_test_split(train_data, train_size= 0.7857, random_state = 33, shuffle = True) #0.7857

			if self.check_extrapolation:
				record_id, tt, _, vals, _, mask, _, mask_surv, labels, dur = train_data[0]
			else:
				record_id, tt, vals, mask, mask_surv, labels, dur = train_data[0]

			n_samples = len(total_dataset)
			input_dim = vals.size(-1)

			batch_size = min(min(len(train_data), param_dics['batch_size']), n_samples)
			# get min and max using train and valid sets only :
			# breakpoint()
			# get normalization params
			data_min, data_max = utils.get_data_min_max(train_data, dataset = self.dataset, device = self.device, check_extrapolation = self.check_extrapolation)
			# breakpoint()

			train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "train",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))
			
			train_dataloader_baseline = DataLoader(train_data, batch_size= len(train_data), shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "train",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))

			# breakpoint(), len(valid_data), batch_size*2
			valid_dataloader = DataLoader(valid_data, batch_size = batch_size*2, shuffle=False, # prev : batch_size*2
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "test",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))

			test_dataloader = DataLoader(test_data, batch_size = len(test_data), shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "test",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))


			data_objects = {"train_dataloader": utils.inf_generator(train_dataloader), 
							"train_dataloader_cox": utils.inf_generator(train_dataloader_baseline),
							"valid_dataloader": utils.inf_generator(valid_dataloader),
							"test_dataloader": utils.inf_generator(test_dataloader),
							"input_dim": input_dim,
							"n_train_batches": len(train_dataloader),
							"n_valid_batches": len(valid_dataloader),
							"attr": feat_names if feat_reconstr is None else feat_reconstr} #optional
			return data_objects, (data_min, data_max)
		elif not process_test_set:
			train_data, valid_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
			# train_data, valid_data = model_selection.train_test_split(train_data, train_size= 0.875, random_state = 33, shuffle = True)
			# train 60, valid 10, test 30
			# train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.7, random_state = 42, shuffle = True)
			# train_data, valid_data = model_selection.train_test_split(train_data, train_size= 0.857, random_state = 33, shuffle = True)

			if self.check_extrapolation:
				record_id, tt, _, vals, _, mask, _, mask_surv, labels, dur = train_data[0]
			else:
				record_id, tt, vals, mask, mask_surv, labels, dur = train_data[0]

			n_samples = len(train_data)
			input_dim = vals.size(-1)

			batch_size = min(min(len(train_data), param_dics['batch_size']), n_samples)
			data_min, data_max = utils.get_data_min_max(total_dataset, dataset = self.dataset, device = self.device, check_extrapolation = self.check_extrapolation)
			# breakpoint()

			train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "train",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))
			
			train_dataloader_baseline = DataLoader(train_data, batch_size= len(train_data), shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "train",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))

			valid_dataloader = DataLoader(valid_data, batch_size = batch_size*2, shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "test",
					data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))

			# test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			# 	collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, device, data_type = "test",
			# 		data_min = data_min, data_max = data_max, dataset = dataset, max_time = max_time))

			data_objects = {"train_dataloader": utils.inf_generator(train_dataloader), 
							"train_dataloader_cox": utils.inf_generator(train_dataloader_baseline),
							"valid_dataloader": utils.inf_generator(valid_dataloader),
							"input_dim": input_dim,
							"n_train_batches": len(train_dataloader),
							"n_valid_batches": len(valid_dataloader),
							"attr": feat_names if feat_reconstr is None else feat_reconstr} #optional
			# breakpoint()
			return data_objects, (data_min, data_max)
		elif process_test_set:
			# for test set 
			record_id, tt, vals, mask, mask_surv, labels, dur = total_dataset[0]
			n_samples = len(total_dataset)
			input_dim = vals.size(-1)

			data_min, data_max = self.min_max_tuple

			test_dataloader = DataLoader(total_dataset, batch_size = n_samples, shuffle=False, 
				collate_fn= lambda batch: utils.variable_time_collate_fn_survival(batch, self.device, data_type = "test",
				data_min = data_min, data_max = data_max, dataset = self.dataset, max_time = max_time, feat_reconstr_idx = feat_reconstr_idx, feat_names = feat_names, check_extrapolation = self.check_extrapolation, max_obs_time = self.max_obs_time))

			data_objects = {"test_dataloader": utils.inf_generator(test_dataloader),
							"input_dim": input_dim,
							"n_test_batches": len(test_dataloader),
							"attr": feat_names if feat_reconstr is None else feat_reconstr} #optional

			return data_objects, None

	def get_min_max_data(self):
		return self.min_max_tuple

	def process_test_data(self, data, data_info_dic, batch_size = 100, max_time = None, n_samples = None, dataset = 'general', run_id = None, include_test_set = False, feat_reconstr = None, check_extrapolation = False, test_restrict = None, model_info = None):
		# set random seed and experiment ID
		random_seed = 1991
		experimentID = None
		# save_path = 'experiments/'
		self.check_extrapolation = check_extrapolation
		self.dataset = dataset
		self.min_max_tuple = model_info['min_max_data_tuple']
		# breakpoint()
		# self.min_max_tuple = np.load('model_performance/0_surv_latent_ode_v0_high_d_norm/min_max_tuple_at_train.npy')#, min_max_tuple_to_save)

		# breakpoint()
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)

		try:
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = self._pre_process_data(data, data_info_dic, n_samples = n_samples, max_time = max_time, include_test_set = include_test_set)
		except:
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = utils.pre_process_data(data, data_info_dic, n_events = self.n_events, check_extrapolation = check_extrapolation, dataset = dataset, n_samples = n_samples, max_time = max_time, include_test_set = include_test_set, impute = 'mean')
		param_dics = {}
		param_dics['batch_size'] = batch_size

		try:
			data_obj, _ = self._get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, param_dics, device = self.device, feat_reconstr = feat_reconstr, max_time = max_time, include_test_set = include_test_set, test_restrict = test_restrict, process_test_set = not include_test_set)
		except:
			data_obj, self.min_max_tuple, self.feat_reconstr_idx = utils.get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, param_dics, max_obs_time = max_obs_time, device = self.device, dataset = dataset, feat_reconstr = feat_reconstr, max_time = max_time, include_test_set = include_test_set, process_test_set = not include_test_set, test_restrict = test_restrict, check_extrapolation = check_extrapolation)
		
		return utils.remove_timepoints_wo_obs(utils.get_next_batch(data_obj["test_dataloader"])), None
		# return utils.get_next_batch(data_obj["test_dataloader"]), min_max_tuple

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
		

	def fit(self, data, data_info_dic, device = None, niters = 30, batch_size = 50, lr = 1e-2, max_time = None, n_samples = None, dataset = 'general', run_id = None, include_test_set = False, survival_loss_scale = 10, n_latent_traj = 1, early_stopping = False, survival_loss_exp = True, train_info = None, feat_reconstr = None, check_extrapolation = False, wait_until_full_surv_loss = 15, test_restrict = None):
		"""
		include_test_set : indicator for including test set as a part of main data pre-processing. 
						   if set to True, user can load the processed test_batch_dict by invoking get_test_data_dic() method
		"""

		# check extrapolation
		self.check_extrapolation = check_extrapolation
		self.dataset = dataset

		# set random seed and experiment ID
		random_seed = 1991 # orig random seed : 1991
		experimentID = None
		save_path = 'experiments/'

		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		ckpt_path = os.path.join(save_path, "run_" + str(run_id) + '.ckpt')

		# apply max time (post data processing)
		ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = self._pre_process_data(data, data_info_dic, n_samples = n_samples, max_time = max_time, include_test_set = include_test_set)
		# self.feat_names = feat_names
		# breakpoint()

		# set params
		param_dics = {}
		param_dics['n_samples'] = n_samples
		param_dics['batch_size'] = batch_size
		param_dics['lr'] = lr
		param_dics['niters'] = niters
		param_dics['latent_dim'] = self.latent_dim
		param_dics['rec_dim'] = self.rec_dim
		param_dics['rec_layers'] = self.rec_layers
		param_dics['gen_layers'] = self.gen_layers
		param_dics['units_ode'] = self.units_ode
		param_dics['units_gru'] = self.units_gru
		param_dics['attn_num_heads'] = self.attn_num_heads
		# param_dics['units_gru']

		# create data objects
		data_obj, self.min_max_tuple = self._get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, param_dics, device = self.device, feat_reconstr = feat_reconstr, max_time = max_time, include_test_set = include_test_set, test_restrict = test_restrict)
		param_dics['input_dim'] = data_obj["input_dim"]
		# breakpoint()
		if include_test_set: 
			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
			self.test_batch_dict = batch_dict

			parent_dir = "model_performance/"
			path = os.path.join(parent_dir, run_id)
			if not os.path.exists(path):
				os.mkdir(path)
			else:
				print('using currently existing directory : ', path)

			min_max_tuple_to_save = (self.min_max_tuple[0].numpy(), self.min_max_tuple[1].numpy())
			np.save('model_performance/' + run_id + '/min_max_tuple_at_train.npy', min_max_tuple_to_save)
			

		utils.train_surv_model(self, data_obj, param_dics, 
							   device = self.device, surv_est = self.surv_est, 
							   max_time = max_time, run_id = run_id, dataset = dataset, survival_loss_scale = survival_loss_scale, n_latent_traj = n_latent_traj, early_stopping = early_stopping, survival_loss_exp = survival_loss_exp, train_info = train_info, n_events = self.n_events, wait_until_full_surv_loss = wait_until_full_surv_loss)
		return

	def get_surv_prob(self, data, data_info_dic, batch_dict = None, model_info = None, max_time = None, dataset = 'mimic', filename_suffix = None, device = None, n_latent_traj = 1, credible_interval = False, test_batch_size = 200, reconstr_loss = False, export_latent_states = False, n_samples_surv_curv = 500, n_events = 1):
		"""
		obtain survival probability
		"""

		if credible_interval and n_latent_traj == 1:
			raise ValueError('n_latent_traj must be greater than 1 to obtain credible interval')
		elif not credible_interval and n_latent_traj > 1:
			raise ValueError('Must set credible_interval to True when evaluating more than one trajectory')

		if batch_dict is None:
			# convert data into dataloader format 
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = self._pre_process_data(data, data_info_dic, max_time = max_time, dataset = dataset)

			data_obj, _ = self._get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, None, 
											 dataset = dataset, device = self.device, max_time = max_time, min_max_tuple = model_info['min_max_data_tuple'], process_test_set = True)

			# get hazards and prediction
			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
			self.test_batch_dict = batch_dict
		
		# compute surv prob
		last_observed_points = batch_dict['end_of_obs_idx'] # auxiliary info for plotting
		num_samples = len(batch_dict['sample_ids'])
		# for j in range(num_samples):
		# 	last_observed_point = batch_dict['observed_tp_unnorm'][batch_dict['observed_mask'][j].sum(axis = 1) > 0][-1]
		# 	last_observed_points.append(last_observed_point)

	
		batch_total_observed_data = utils.divide_list(batch_dict["observed_data"], test_batch_size)
		batch_total_observed_mask = utils.divide_list(batch_dict["observed_mask"], test_batch_size)
		batch_total_end_obs_idx = utils.divide_list(batch_dict["end_of_obs_idx"], test_batch_size)
		# batch_total_sample_ids = utils.divide_list(batch_dict["sample_ids"], test_batch_size)

		# breakpoint()
		# =======================================================================================
		print('Getting latent states of test data...')			
		init_flag = True; count = 0;
		for i in range(len(batch_total_observed_data)):
			pred_y_, hazards_y, info = self.get_reconstruction_survival(batch_dict["tp_to_predict"], 
																batch_total_observed_data[i], batch_dict["observed_tp"], batch_total_end_obs_idx[i], 
																mask = batch_total_observed_mask[i], get_latent_hazard = True, temporal_encoding = self.temporal_encoding)
			# breakpoint()
			latent_states = info['latent_hazard']
			if init_flag:
				init_flag = False
			else:
				hazards_y = torch.cat((prev_hazards_y, hazards_y), 1)
				pred_y_ = torch.cat((prev_pred_y, pred_y_), 1)
				latent_states = prev_latent_states + latent_states#torch.cat((prev_latent_states, latent_states), 0)
				# sample_ids = sample_ids + 
			prev_pred_y = pred_y_.detach()
			prev_latent_states = latent_states
			# if n_events == 1:
			prev_hazards_y = hazards_y.detach()	
			# prev_sample_ids = batch_total_sample_ids[i]
		print('Complete!')
		if export_latent_states:
			f = open('model_performance/' + filename_suffix + '/latent_states_test_set.pkl', "wb") # prev : ckp_sig_feats_dic_Jan_21th_2021_binary_wo_duplicates_thresh_0_10, ckp_sig_feats_dic_Nov_13th_binary_mut_burden, ckp_sig_feats_dic_Nov_13th_binary, ckp_sig_feats_dic_Sep_25th_binary
			pickle.dump(latent_states,f)
			f.close()

			with open('model_performance/' + filename_suffix + '/test_sample_ids.npy', 'wb') as f:
				np.save(f, batch_dict["sample_ids"])

			# export data prediction as well 				
			np.save('model_performance/' + filename_suffix + '/test_data_pred.npy', pred_y_[0].cpu().detach().numpy())

		# =======================================================================================
		
		# breakpoint()
		if self.check_extrapolation:
			pred_y, pred_y_extr = pred_y_[:, :, :batch_dict["max_end_of_obs_idx"] + 1, :], pred_y_[:, :, batch_dict["max_end_of_obs_idx"] + 1:, :]
			pred_y = torch.index_select(pred_y, 2, batch_dict['observed_tp_unnorm_dec'].int())#[:, :, batch_dict['non_missing_tp_pred'], :] # choose the relevent time points for reconstruction loss
		else:
			pred_y = pred_y_[:, :, :batch_dict["max_end_of_obs_idx"] + 1, :] #, pred_y[:, :, batch_dict["max_end_of_obs_idx"]:, :]
			pred_y = torch.index_select(pred_y, 2, batch_dict['observed_tp_unnorm_dec'].int())#[:, :, batch_dict['non_missing_tp_pred'], :] 
		
		if reconstr_loss:
			fp_mu, fp_std, fp_enc = info["first_point"]
			fp_std = fp_std.abs()
			fp_distr = Normal(fp_mu, fp_std)

			assert(torch.sum(fp_std < 0) == 0.)

			kldiv_z0 = kl_divergence(fp_distr, self.z0_prior) # ~ N(0,1)

			if torch.isnan(kldiv_z0).any():
				print(fp_mu)
				print(fp_std)
				raise Exception("kldiv_z0 is Nan!")

			# Mean over number of latent dimensions
			kldiv_z0 = torch.mean(kldiv_z0,(1,2))

			# Compute likelihood of all the points
			rec_likelihood = utils.get_gaussian_likelihood(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], obsrv_std = self.obsrv_std)			
			if n_events == 1:
				surv_prob = self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_time_window = max_time, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'], n_events = n_events)
				return surv_prob, rec_likelihood
			else:	
				ef_surv_prob, cs_cif_total = self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_time_window = max_time, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'], n_events = n_events)
				# breakpoint()
				return ef_surv_prob, cs_cif_total, rec_likelihood
		else:
			return self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_time_window = max_time, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'], n_events = n_events)

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
			ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names = self._pre_process_data(data, data_info_dic, max_time = max_time, dataset = dataset)

			data_obj, _ = self._get_data_obj(ids, x, x_ext, m, m_ext, ms, t, t_ext, e, dur, feat_names, None, 
											 dataset = dataset, device = self.device, max_time = max_time, min_max_tuple = model_info['min_max_data_tuple'], process_test_set = True)

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
			return pred_y #self._get_surv_prob(hazards_y, batch_dict, last_observed_points, max_time_window = max_time, filename_suffix = filename_suffix, events_info_train_tuple = model_info['events_info_train_tuple'])
		# return
	# def get_feat_names():
	# 	if self.feat_names is not None:
	# 		return self.feat_names

class SurvLatentODE_Cox(LatentODESub):
	def __init__(self):
		raise NotImplementedError

class SurvLatentODE(LatentODESub):
	def __init__(self, latent_dim = 20, rec_dim = 40, rec_layers = 3, gen_layers = 3, units_ode = 50, units_gru = 50, input_dim = 20, reconstr_dim = None, device = None, gru_aug = False, attention_aug = False, attn_num_heads = 4, n_events = 1, temporal_encoding = False, mult_event_units = 5, num_layer_hazard_dec = 2):
		self.surv_est = 'Hazard'
		super().__init__(latent_dim = latent_dim,
						 rec_dim = rec_dim,
						 rec_layers = rec_layers,
						 gen_layers = gen_layers,
						 units_ode = units_ode,
						 units_gru = units_gru,
						 device = device,
						 input_dim = input_dim, 
						 gru_aug = gru_aug,
						 attention_aug = attention_aug, 
						 attn_num_heads = attn_num_heads,
						 temporal_encoding = temporal_encoding,
						 n_events = n_events, 
						 mult_event_units = mult_event_units, 
						 num_layer_hazard_dec = num_layer_hazard_dec,
						 reconstr_dim = reconstr_dim if reconstr_dim is not None else input_dim)

	def _get_surv_prob(self, hazards_y, batch_dict, last_observed_points, cred_interval = False, max_time_window = None, filename_suffix = None, events_info_train_tuple = None, n_events = 1):
		results = {'hazards_y':hazards_y}
		return utils.compute_survival_curves(results, batch_dict, None, last_observed_points, surv_est = self.surv_est, n_events = n_events)

	def _get_reconst_traj(self, data):
		pass
