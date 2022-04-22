####################################################
# SurvLatent ODE
# Author : Intae Moon
#
# Partially adpated from Latent ODEs for Irregularly-Sampled Time Series (Rubanova et al. 2019)
####################################################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline

class ODE_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
		z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
		concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
		classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

		Baseline.__init__(self, input_dim, latent_dim, device = device, 
			obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		ode_rnn_encoder_dim = latent_dim
	
		self.ode_gru = Encoder_z0_ODE_RNN( 
			latent_dim = ode_rnn_encoder_dim, 
			input_dim = (input_dim) * 2, # input and the mask
			z0_diffeq_solver = z0_diffeq_solver, 
			n_gru_units = n_gru_units, 
			device = device).to(device)

		self.z0_diffeq_solver = z0_diffeq_solver

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)

		self.decoder_cox = nn.Sequential(
			nn.Linear(latent_dim, 1)
			)

		self.decoder_weibull_lambda = nn.Sequential(
			nn.Linear(latent_dim, 1, bias = True)
			) 

		self.decoder_weibull_sigma = nn.Sequential(
			nn.Linear(latent_dim, 1, bias = False),
			nn.Tanh(),
			nn.Linear(1, 1, bias = True)
			) 

		# initialize sigma weights
		# net.modules()
		# self.decoder_cox = nn.Sequential(
		# 	nn.Linear(latent_dim, n_units),
		# 	nn.Tanh(),
		# 	nn.Linear(n_units, 1),)

		# scale the time resolution!
		utils.init_network_weights(self.decoder)
		utils.init_network_weights(self.decoder_cox)
		utils.init_network_weights(self.decoder_weibull_lambda, mode = 'weibull_lambda')
		utils.init_network_weights(self.decoder_weibull_sigma, mode = 'weibull_sigma')

	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = None, mode = None):

		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for ODE-RNN")

		# time_steps_to_predict and truth_time_steps should be the same 
		assert(len(truth_time_steps) == len(time_steps_to_predict))
		assert(mask is not None)
		
		data_and_mask = data
		if mask is not None:
			data_and_mask = torch.cat([data, mask],-1)

		_, _, latent_ys, _ = self.ode_gru.run_odernn(
			data_and_mask, truth_time_steps, run_backwards = False)
		
		latent_ys = latent_ys.permute(0,2,1,3)
		last_hidden = latent_ys[:,:,-1,:]
		# breakpoint()
			#assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		outputs = self.decoder(latent_ys)

		# cox_outputs = self.decoder_cox(latent_ys)

		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (latent_ys[:,:,-1,:], 0.0, latent_ys[:,:,-1,:])}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(latent_ys)
			else:
				extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

		extra_info['last_hidden_layer'] = last_hidden
		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		# breakpoint()
		return outputs, extra_info

	def get_reconstruction_survival(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = None, mode = None, survival_mode_num = None):

		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for ODE-RNN")

		# time_steps_to_predict and truth_time_steps should be the same 
		assert(len(truth_time_steps) == len(time_steps_to_predict))
		assert(mask is not None)
		
		data_and_mask = data
		if mask is not None:
			data_and_mask = torch.cat([data, mask],-1)

		_, _, latent_ys, _ = self.ode_gru.run_odernn(
			data_and_mask, truth_time_steps, run_backwards = False)
		
		latent_ys = latent_ys.permute(0,2,1,3)
		last_hidden = latent_ys[:,:,-1,:]
		# breakpoint()
			#assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		outputs = self.decoder(latent_ys)
		if survival_mode_num == 1: # cox
			cox_outputs = self.decoder_cox(latent_ys)
		elif survival_mode_num == 2: # weibull
			weibull_output = self.decoder_weibull_lambda(latent_ys)
			weibull_sigma = self.decoder_weibull_sigma(latent_ys)
			
		# breakpoint()
		# print('\n')
		# print('Layer weight for sigma : ', self.decoder_weibull_sigma[2].weight)
		# print('Layer bias for sigma : ', self.decoder_weibull_sigma[2].bias)
		# print('\n')

		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (latent_ys[:,:,-1,:], 0.0, latent_ys[:,:,-1,:])}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(latent_ys)
			else:
				extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

		extra_info['last_hidden_layer'] = last_hidden
		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		# breakpoint()
		if survival_mode_num == 1: # cox
			return outputs, cox_outputs, None, extra_info
		elif survival_mode_num == 2: # weibull
			return outputs, weibull_output, weibull_sigma, extra_info

