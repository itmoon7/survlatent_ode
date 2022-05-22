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
from torch.distributions import Categorical, Normal
import lib.utils as utils
from torch.nn.modules.rnn import LSTM, GRU
# from lib.utils import get_device

# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit_for_vae(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100, gru_aug = False,  # add GRU augmentation 
		device = torch.device("cpu")):
		super().__init__()

		self.gru_aug = gru_aug
		self.hidden_size = latent_dim
		if update_gate is None:
			self.update_gate = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units), # for GRU augmentation, you'd need to change dimension here 
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate
		# breakpoint()
		if reset_gate is None:
			self.reset_gate = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			self.new_state_net = nn.Sequential(
			   nn.Linear(latent_dim + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim))
			utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, x, hidden):
		# if not self.gru_aug:
		if hidden is None:
			hidden = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
		y_concat = torch.cat([hidden, x], -1) 

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([hidden * reset_gate, x], -1)
		new_state = self.new_state_net(concat)
		# new_state_std = new_state_std.abs()

		new_hidden = (1-update_gate) * new_state + update_gate * hidden
		return new_hidden

# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100, gru_aug = False,  # add GRU augmentation 
		device = torch.device("cpu")):
		super(GRU_unit, self).__init__()

		self.gru_aug = gru_aug
		if update_gate is None:
			self.update_gate = nn.Sequential(
			   nn.Linear(input_dim*2 if self.gru_aug else latent_dim * 2 + input_dim, n_units), # for GRU augmentation, you'd need to change dimension here 
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate
		# breakpoint()
		if reset_gate is None:
			self.reset_gate = nn.Sequential(
			   nn.Linear(input_dim*2 if self.gru_aug else latent_dim * 2 + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			self.new_state_net = nn.Sequential(
			   nn.Linear(input_dim*2 if self.gru_aug else latent_dim * 2 + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim if self.gru_aug else latent_dim * 2))
			utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, y_mean, y_std, x, masked_update = True):
		# if not self.gru_aug:
		y_concat = torch.cat([y_mean, y_std, x], -1) 

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
		
		new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		new_state_std = new_state_std.abs()

		new_y = (1-update_gate) * new_state + update_gate * y_mean
		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std
		# else:
		# 	# masked_update = False # no need to update mask for GRU aug mode
		# 	y_concat = torch.cat([y_mean, x], -1) 

		# 	# breakpoint()
		# 	update_gate = self.update_gate(y_concat)
		# 	reset_gate = self.reset_gate(y_concat)
		# 	concat = torch.cat([y_mean * reset_gate, x], -1)
			
		# 	new_state = self.new_state_net(concat)
		# 	# new_state_std = new_state_std.abs()

		# 	new_y = (1-update_gate) * new_state + update_gate * y_mean
		# 	# new_y_std = (1-update_gate) * new_state_std + update_gate * y_std
		# 	return new_y

		assert(not torch.isnan(new_y).any())

		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			utils.check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_y = mask * new_y + (1-mask) * y_mean
			new_y_std = mask * new_y_std + (1-mask) * y_std

			if torch.isnan(new_y).any():
				print("new_y is nan!")
				print(mask)
				print(y_mean)
				print(prev_new_y)
				exit()

		new_y_std = new_y_std.abs()
		# breakpoint()
		return new_y, new_y_std

class Encoder_z0_ODE_RNN(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
		z0_dim = None, z0_dim_haz = None, GRU_update = None, z0_diffeq_solver_hazard = None,
		n_gru_units = 100, attention = False, 
		device = torch.device("cpu")):
		
		super(Encoder_z0_ODE_RNN, self).__init__()

		self.attention = attention
		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		# breakpoint()
		if GRU_update is None:
			self.GRU_update = GRU_unit(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device).to(device)
		else:
			self.GRU_update = GRU_update

		# breakpoint()
		self.z0_diffeq_solver = z0_diffeq_solver
		if z0_diffeq_solver_hazard is not None:
			self.z0_dim_haz = z0_dim_haz
			self.model_hazard = True
			self.z0_diffeq_solver_hazard = z0_diffeq_solver_hazard
			# instantiate hazard decoder
			self.decoder_hazard = nn.Sequential(
				nn.Linear(latent_dim, 40),
				nn.Tanh(),
				nn.Linear(40, self.z0_dim_haz))
			utils.init_network_weights(self.decoder_hazard)
		else:
			self.model_hazard = False

		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.extra_info = None

		self.transform_z0 = nn.Sequential(
							nn.Linear(latent_dim * 2, 100),
							nn.Tanh(),
							nn.Linear(100, self.z0_dim * 2),)
		utils.init_network_weights(self.transform_z0)

		if self.attention:
			self.attention_layer = nn.Sequential(
									nn.Linear(latent_dim * 2, self.z0_dim),
									nn.Tanh(),
									nn.Linear(self.z0_dim, self.z0_dim),
									nn.Tanh(),
									nn.Linear(self.z0_dim, 1))

			# self.attention_layer = nn.Sequential(
			# 						nn.Linear(latent_dim, self.z0_dim),
			# 						nn.Tanh(),
			# 						nn.Linear(self.z0_dim, self.z0_dim),
			# 						nn.Tanh(),
			# 						nn.Linear(self.z0_dim, 1))

			utils.init_network_weights(self.transform_z0)
		# if self.model_hazard == True:
		# 	self.transform_z0_hazard = nn.Sequential(
		# 	   nn.Linear(self.z0_dim_haz * 2, 40),
		# 	   nn.Tanh(),
		# 	   nn.Linear(40, 2),)
		# 	utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = True, save_info = False):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
			xi = data[:,0,:].unsqueeze(0)
			last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
			extra_info = None
		else:
			last_yi, last_yi_std, latent_ys, latent_ys_std, extra_info = self.run_odernn(
																			data, time_steps, run_backwards = run_backwards,
																			save_info = save_info)
		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim) # [1,50,40]
		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim) # [1,50,40]
		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1))) # transform z0
		std_z0 = std_z0.abs() 
		if save_info:
			self.extra_info = extra_info
		return mean_z0, std_z0


	def run_odernn(self, data, time_steps, 
		run_backwards = True, save_info = False):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		n_traj, n_tp, n_dims = data.size()
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

		device = utils.get_device(data)

		prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device) 
		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = interval_length / 50 # prev it was 50

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []; latent_ys_std = []
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:
			if (prev_t - t_i) < minimum_step: # small time gap between observations
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())
				ode_sol = prev_y + inc 
				ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

				assert(not torch.isnan(ode_sol).any())
			else: # large time gap between observations
				n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y, time_points) # z0_diffeq_solver is a class of DiffeqSolver

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
				exit()

			yi_ode = ode_sol[:, :, -1, :]
			xi = data[:,i,:].unsqueeze(0)
			
			yi, yi_std = self.GRU_update(yi_ode, prev_std, xi) 
			prev_y, prev_std = yi, yi_std		
			prev_t, t_i = time_steps[i],  time_steps[i-1]

			latent_ys.append(yi)
			latent_ys_std.append(yi_std)

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
					 "yi": yi.detach(), "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)
		latent_ys_std = torch.stack(latent_ys_std, 1)

		assert(not torch.isnan(yi).any())
		assert(not torch.isnan(yi_std).any())

		return yi, yi_std, latent_ys, latent_ys_std, extra_info



class Decoder(nn.Module):
	def __init__(self, latent_dim, input_dim, surv_est = None, n_events = 1, num_layer = 2, mult_event_units = 5):
		super(Decoder, self).__init__()
		# decode data from latent space where we are solving an ODE back to the data space
		if surv_est is None:
			# orig 
			# decoder = nn.Sequential(
			#    nn.Linear(latent_dim, input_dim)) # try tuning this for more complex data reconstruction

			# data decoder :
			# decoder = nn.Sequential(
			# 	nn.Linear(latent_dim, latent_dim),
			# 	nn.Tanh(),
			# 	nn.Linear(latent_dim, input_dim),
			# 	nn.Sigmoid())

			decoder = nn.Sequential(
				nn.Linear(latent_dim, latent_dim),
				nn.Sigmoid(),
				nn.Linear(latent_dim, input_dim))

			# before it was linear -> sigmoid -> linear

		elif surv_est == 'Cox':
			if n_events == 1:
				decoder = nn.Sequential(
					nn.Linear(latent_dim, 10),
					nn.Tanh(),
					nn.Linear(10, 1, bias = False))
			else:
				decoder = nn.Sequential(
					nn.Linear(latent_dim, 10),
					nn.Tanh(),
					nn.Linear(10, 5, bias = False))
		elif surv_est == 'Softmax':
			decoder = nn.Sequential(
				nn.Linear(latent_dim, 10),
				nn.Tanh(),
				nn.Linear(10, 1))
		elif surv_est == 'Hazard':
			if n_events == 1:
				if num_layer == 2:
					decoder = nn.Sequential(
						nn.Linear(latent_dim, int(latent_dim/2)),
						nn.Tanh(),
						nn.Linear(int(latent_dim/2), 1),
						nn.Sigmoid()
						)
				elif num_layer == 3:
					decoder = nn.Sequential(
						nn.Linear(latent_dim, int(latent_dim/2)),
						nn.Tanh(),
						nn.Linear(int(latent_dim/2), int(latent_dim/2)),
						nn.Tanh(), # just arbitrailiy choose 5
						nn.Linear(int(latent_dim/2), 1),
						nn.Sigmoid()
						)
			else:
				if num_layer == 2:
					# before
					# decoder = nn.Sequential(
					# 	nn.Linear(latent_dim, 10),
					# 	nn.Tanh(),
					# 	nn.Linear(10, mult_event_units) 
					# 	)
					decoder = nn.Sequential(
						nn.Linear(latent_dim, int(latent_dim/2)),
						nn.Tanh(),
						nn.Linear(int(latent_dim/2), mult_event_units) 
						)
				elif num_layer == 3:
					decoder = nn.Sequential(
						nn.Linear(latent_dim, int(latent_dim/2)),
						nn.Tanh(),
						nn.Linear(int(latent_dim/2), int(latent_dim/2)),
						nn.Tanh(), # just arbitrailiy choose 5
						nn.Linear(int(latent_dim/2), mult_event_units)
						)

		else:
			raise KeyError('Only support Cox and Softmax atm')
		utils.init_network_weights(decoder, mode = surv_est)	
		self.decoder = decoder

	def forward(self, data):
		return self.decoder(data)


class LearnedSoftPlus(torch.nn.Module):
	def __init__(self, init_beta=1.0, threshold=20):
		super().__init__()
		# keep beta > 0
		self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log())
		self.threshold = 20
	def forward(self, x):
		beta = self.log_beta.exp()
		beta_x = beta * x
		return torch.where(beta_x < 20, torch.log1p(beta_x.exp()) / beta, x)
