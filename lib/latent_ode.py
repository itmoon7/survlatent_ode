####################################################
# SurvLatent ODE
# Author : Intae Moon
#
# Partially adpated from Latent ODEs for Irregularly-Sampled Time Series (Rubanova et al. 2019)
####################################################

import numpy as np
import sklearn as sk

#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
# from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *
# from lib.multi_head_attention import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

import matplotlib.pyplot as plt
from tqdm import tqdm

import time

class LatentODE(nn.Module):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, decoder_surv, diffeq_solver, 
				 z0_prior, device, obsrv_std=None, gru_aug=False, n_gru_units=50): 

		super().__init__()

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		if self.n_events == 1:
			self.decoder_surv = decoder_surv
		else:
			self.decoder_surv_1 = decoder_surv[0]
			self.decoder_surv_2 = decoder_surv[1]
		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
		self.z0_prior = z0_prior
		self.gru_aug = gru_aug
		self.n_gru_units = n_gru_units
		
		# GRU unit
		if self.gru_aug:
			print('Augmenting GRU...')
			self.GRU_update = GRU_unit(self.latent_dim, self.latent_dim,
									   n_units = self.n_gru_units, gru_aug = True, 
									   device=device).to(device)
		
		if self.attention_aug:
			print('\n')
			print('Augmenting the model with multi-head self attention...')
			print('\n')

			if self.n_events == 1:
				self.attention_decoder = DecoderBlock(self.latent_dim,
													  self.attn_num_heads,
													  self.latent_dim).to(device)
			elif self.n_events == 2:
				self.attention_decoder_1 = DecoderBlock(self.latent_dim,
														self.attn_num_heads,
														self.latent_dim).to(device)
				self.attention_decoder_2 = DecoderBlock(self.latent_dim,
														self.attn_num_heads,
														self.latent_dim).to(device)
				self.merging_decoder = nn.Sequential(nn.Linear(self.mult_event_units*self.n_events,
												 	 		   self.n_events + 1)).to(device) 
				utils.init_network_weights(self.merging_decoder, mode = self.surv_est)	
			else:
				raise KeyError('more than 2 events will be implemented : TBD')

			self.attention_decoder_data = DecoderBlock(self.latent_dim, self.attn_num_heads,
													   self.latent_dim).to(device)
		else:
			if self.n_events == 2:
				self.merging_decoder = nn.Sequential(nn.Linear(self.mult_event_units*self.n_events,
															   self.n_events + 1)).to(device)

	def _plot_recons_traj(self, observed_traj, pred_traj, pred_traj_extr, data_extra_info,
						  mask=None, n_feats_per_plot=9, feat_names=None, filename_suffix=None,
						  curr_epoch=None, min_event_time=None, pred_horizon_hours=100):
		"""
		observed_traj : (n_samples, n_timepoints, n_features), where unobserved datapoints are set to 0 
		pred_traj : (n_samples, n_timepoints, n_features)
		pred_traj_extr : (n_samples, n_timepoints_to_predic from 48 hour mark, n_features)

		sample-wise reconstruction traj : randomly selected 10 samples
		average reconstruction traj : average 
		"""
		times_oi = torch.arange(0, np.shape(observed_traj)[1])

		np.random.seed(0) # set seed 0
		sampled_idx = np.random.choice(np.arange(len(observed_traj)), replace=False, size=10)
		print('sample idx : ', sampled_idx)
		if self.check_extrapolation:
			times_oi_extr_pred = torch.arange(np.shape(observed_traj)[1],
											  np.shape(observed_traj)[1] + np.shape(pred_traj_extr)[1])
			times_extr, obs_traj_extr, masks_extra = data_extra_info
		
		colors = ['k', 'g', 'r', 'b', 'c', 'y', 'm', 'lime', 'darkred', 'rosybrown']
		for obs_traj_sample, pred_traj_sample, idx in zip(observed_traj[sampled_idx, :, :],
														  pred_traj[sampled_idx, :, :],
														  sampled_idx):
			fig, ax = plt.subplots()
			# plot observed trajectory for each feat
			for feat_num in range(n_feats_per_plot):
				if feat_names[feat_num] in ['age', 'gender']:
					continue
				obs_traj_oi = obs_traj_sample[:, feat_num]
				obs_traj_to_plot = obs_traj_oi[obs_traj_oi != 0]
				times_to_plot = times_oi[obs_traj_oi != 0]
				ax.scatter(times_to_plot, obs_traj_to_plot.cpu(), color=colors[feat_num],
						   alpha=0.5, label=feat_names[feat_num] + ' observed')

				# plot pred trajectory for each feat
				pred_traj_sample_oi = pred_traj_sample[:, feat_num]
				ax.plot(times_oi, pred_traj_sample_oi.cpu(), color=colors[feat_num],
						linestyle='dotted', label=feat_names[feat_num] + ' pred')

			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol = 4)
			fig.savefig(("surv_curves/" + filename_suffix + "/reconstruction/" +
					     str(curr_epoch) + '_reconst_feat_traj_for_sample_' + str(idx) + ".pdf"),
						bbox_inches='tight')
			plt.close()
		
		if self.check_extrapolation:
			for (time_per_sample, mask_per_sample,
				obs_traj_extra_sample,
				pred_traj_extr_sample, idx) in zip(times_extr, masks_extra, obs_traj_extr,
												   pred_traj_extr[sampled_idx, :, :],
												   sampled_idx):
				fig, ax = plt.subplots()
				# plot observed trajectory for each feat
				for feat_num in range(n_feats_per_plot):
					if feat_names[feat_num] in ['age', 'gender']:
						continue
					obs_traj_oi = obs_traj_extra_sample[:, feat_num]
					mask_per_sample_oi = mask_per_sample[:, feat_num]
					obs_traj_to_plot = obs_traj_oi[mask_per_sample_oi == 1].cpu()
					times_to_plot = time_per_sample[mask_per_sample_oi == 1] # at least one observed
					ax.scatter(times_to_plot, obs_traj_to_plot, color=colors[feat_num],
							   alpha=0.5, label=feat_names[feat_num] + ' observed')

					# plot pred trajectory for each feat
					# pred_traj_sample_oi = pred_traj_sample[:, feat_num]
					# ax.plot(times_oi, pred_traj_sample_oi, color = colors[feat_num], linestyle = 'dotted', alpha = 0.5, label = feat_names[feat_num] + ' pred')

					pred_traj_extr_sample_oi = pred_traj_extr_sample[:, feat_num].cpu()
					ax.plot(times_oi_extr_pred[0:pred_horizon_hours],
							pred_traj_extr_sample_oi[0:pred_horizon_hours],
							color=colors[feat_num], linestyle='dashed',
							label=feat_names[feat_num] + ' extrapolation')
					# breakpoint()
					# break
				# break
				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol=4)
				fig.savefig(("surv_curves/" + filename_suffix + "/reconstruction/" +
							 str(curr_epoch) + '_extrapol_feat_traj_for_sample_' +
							 str(idx) + ".pdf"), bbox_inches='tight')
				plt.close()
		# in average
		observed_traj_mean = observed_traj.mean(dim = 0).cpu()
		pred_traj_mean = pred_traj.mean(dim = 0).cpu()

		fig, ax = plt.subplots()
		for feat_num in range(n_feats_per_plot):
			if feat_names[feat_num] in ['age', 'gender']:
				continue
			ax.scatter(times_oi, observed_traj_mean[:, feat_num], color=colors[feat_num],
					   alpha=0.5, label=feat_names[feat_num] + ' observed')
			ax.plot(times_oi, pred_traj_mean[:, feat_num], color=colors[feat_num],
					linestyle='dotted', alpha=0.5, label=feat_names[feat_num] + ' pred')

		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol = 4)
		fig.savefig(("surv_curves/" + filename_suffix + "/reconstruction/" + str(curr_epoch) +
					 '_reconst_feat_traj_avg.pdf'), bbox_inches='tight')
		plt.close()
		return

	def get_reconstruction_traj(self, results, batch_dict, filename_suffix=None, feat_names=None,
								curr_epoch=None, min_event_time=None, high_prop_feats_idx=None):
		"""
		Get reconstrunction trajectory

		batch_dict['observed_data'] : (n_samples, n_timepoints, n_features)
		batch_dict['observed_mask'] : (n_samples, n_timepoints, n_features)
		results['pred_y'] : (n_traj, n_samples, n_timepoints, n_features)
		results['pred_y_extr'] : (n_traj, n_samples, n_timepoints_to_predic from 48 hour mark, n_features)
		"""

		if high_prop_feats_idx is not None:
			observed_mask = batch_dict['observed_mask'][:, :, high_prop_feats_idx]
			observed_data = batch_dict['observed_data'][:, :, high_prop_feats_idx]# * observed_mask
			pred_data = results['pred_y'][0][:, :, high_prop_feats_idx]
			feat_names = np.asarray(feat_names)[high_prop_feats_idx]
		else:
			observed_mask = batch_dict['mask_predicted_data']
			observed_data = batch_dict['data_to_predict']# * observed_mask
			pred_data = results['pred_y'][0]# * observed_mask
		# breakpoint()

		if self.check_extrapolation:
			pred_data_extr = results['pred_y_extr'][0]
			# extrapolation_check = True
		else:
			pred_data_extr = None
			if self.check_extrapolation:
				raise KeyError('No ground truth data to evaluate extrapolation')
			# extrapolation_check = False
		# breakpoint()
		n_feats_per_plot = np.shape(observed_data)[2] if np.shape(observed_data)[2] < 10 else 10
		self._plot_recons_traj(observed_data, pred_data, pred_data_extr, batch_dict['data_extra_info'],
							   mask=observed_mask, filename_suffix=filename_suffix,
							   feat_names=feat_names, curr_epoch=curr_epoch, min_event_time=min_event_time,
							   n_feats_per_plot=n_feats_per_plot)
		return

	def compute_all_losses(self, batch_dict, n_latent_traj=1, kl_coef=1., surv_est=None,
						   survival_loss_scale=10, reconstr_info=None):	
	
		if reconstr_info is None:
			(pred_y_mult_traj,
				hazards_y_mult_traj,
				info) = self.get_reconstruction_survival(batch_dict["tp_to_predict"], 
														 batch_dict["observed_data"],
														 batch_dict["observed_tp"],
														 batch_dict['end_of_obs_idx'],
														 mask=batch_dict["observed_mask"],
														 n_latent_traj=n_latent_traj,
														 temporal_encoding=self.temporal_encoding)
		else:
			pred_y_mult_traj, hazards_y_mult_traj, info = reconstr_info[0], reconstr_info[1], reconstr_info[2]
		if n_latent_traj > 1:
			pred_y_ = pred_y_mult_traj[0][None, :, :, :]
			hazards_y = hazards_y_mult_traj[0][None, :, :, :]
		else:
			pred_y_ = pred_y_mult_traj
			hazards_y = hazards_y_mult_traj
	
		if self.check_extrapolation:
			pred_y, pred_y_extr = (pred_y_[:, :, :batch_dict["max_end_of_obs_idx"] + 1, :],
				pred_y_[:, :, batch_dict["max_end_of_obs_idx"] + 1:, :])
			pred_y = torch.index_select(pred_y, 2, batch_dict['observed_tp_unnorm_dec'].int())
		else:
			pred_y = pred_y_[:, :, :batch_dict["max_end_of_obs_idx"] + 1, :]
			pred_y = torch.index_select(pred_y, 2, batch_dict['observed_tp_unnorm_dec'].int())
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
		rec_likelihood = utils.get_gaussian_likelihood(batch_dict["data_to_predict"],
													   pred_y,
													   mask=batch_dict["mask_predicted_data"],
													   obsrv_std=self.obsrv_std) 
		surv_likelihood = utils.get_survival_likelihood(hazards_y,
													    batch_dict,
													    surv_est=surv_est,
													    n_events=self.n_events,
													    event_to_event_weight_dict=self.event_to_event_weight_dict)

		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)
		
		loss = loss + surv_likelihood * survival_loss_scale 
		results = {}
		results["loss"] = torch.mean(loss)
		results["survival_loss"] = -1*torch.mean(surv_likelihood).detach() if type(surv_likelihood) == torch.Tensor else 0
		results["likelihood"] = -1*torch.mean(rec_likelihood).detach()
		if surv_est == 'Softmax' or surv_est == 'Hazard': # latent ode nonparam
			if n_latent_traj > 1:
				results["hazards_y"] = hazards_y_mult_traj
			else:
				results["hazards_y"] = hazards_y
		elif surv_est == 'Cox': # latent ode cox
			if n_latent_traj > 1:
				results["f_out_cox"] = hazards_y_mult_traj
			else:
				results["f_out_cox"] = hazards_y

		# decoded outputs. So just provide emprical means
		if n_latent_traj > 1:
			pred_y_mean = pred_y_mult_traj.mean(axis = 0)[None, :, :, :].detach()
			pred_y_mean, pred_y_mean_extr = (pred_y_mean[:, :, :batch_dict["max_end_of_obs_idx"], :],
				pred_y_mean[:, :, batch_dict["max_end_of_obs_idx"]:, :])
			results["pred_y"] = pred_y_mean
			if self.check_extrapolation:
				results["pred_y_extr"] = pred_y_mean_extr # extrapolate points 
		else:
			results["pred_y"] = pred_y
			if self.check_extrapolation:
				results["pred_y_extr"] = pred_y_extr # extrapolate points 
		# breakpoint()		
		return results

	def get_reconstruction_survival(self, time_steps_to_predict, truth, truth_time_steps,
									end_of_obs_idx, mask=None, n_latent_traj=1, run_backwards=True,
									eps=1e-7, get_multiple_traj=False, test_batch_size=100,
									get_latent_hazard=False, temporal_encoding=False,
									latent_harzard_time_window=365):

		if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
			isinstance(self.encoder_z0, Encoder_z0_RNN):

			truth_w_mask = truth
			if mask is not None:
				truth_w_mask = torch.cat((truth, mask), -1)
			first_point_mu, first_point_std = self.encoder_z0(truth_w_mask,
														 	  truth_time_steps,
														 	  run_backwards=run_backwards)
			means_z0 = first_point_mu#.repeat(1, 1, 1)
			sigma_z0 = first_point_std#.repeat(1, 1, 1)
			first_point_enc = utils.sample_standard_gaussian(first_point_mu,
															 first_point_std,
															 n_latent_traj=n_latent_traj,
															 random_seed=self.random_seed)				
		else:
			raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
		
		first_point_std = first_point_std.abs() + eps # to prevent zero std.
		assert(torch.sum(first_point_std < 0) == 0.)

		first_point_enc_aug = first_point_enc

		means_z0_aug = means_z0
			
		assert(not torch.isnan(time_steps_to_predict).any())
		assert(not torch.isnan(first_point_enc).any())
		assert(not torch.isnan(first_point_enc_aug).any())

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
		
		pred_y = self.decoder(sol_y)
		if self.n_events > 1:
			hazards_y_1 = self.decoder_surv_1(sol_y)
			hazards_y_2 = self.decoder_surv_2(sol_y)
			hazards_y = torch.cat((hazards_y_1, hazards_y_2), dim = 3).to(self.device)
			# merge decoder networks to 3 neurons using a linear function and softmax across last neurons
			latent_hazard = hazards_y.cpu().detach().clone()
			hazards_y = torch.softmax(self.merging_decoder(hazards_y), dim = 3)	
		else:
			latent_hazard = sol_y[0].cpu().detach().clone()
			hazards_y = self.decoder_surv(sol_y)
		
		if get_latent_hazard: 
			latent_hazard_main_event = []
			if self.n_events == 1:
				for hazard_per_sample, last_obs_idx in zip(latent_hazard, end_of_obs_idx):
					try:
						last_obs_idx_oi = int(last_obs_idx.cpu().numpy())
					except:
						last_obs_idx_oi = int(last_obs_idx)#.cpu().numpy())
					latent_hazard_main_event.append(hazard_per_sample[last_obs_idx_oi:last_obs_idx_oi+
					                                latent_harzard_time_window, :].numpy()) 
			else:	
				for hazard_per_sample, last_obs_idx in zip(latent_hazard[0], end_of_obs_idx):
					try:
						last_obs_idx_oi = int(last_obs_idx.cpu().numpy())
					except:
						last_obs_idx_oi = int(last_obs_idx)#.cpu().numpy())
					latent_hazard_main_event.append(hazard_per_sample[last_obs_idx_oi:last_obs_idx_oi+
					                                latent_harzard_time_window, :].numpy()) 
		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.cpu().detach(),
			"latent_hazard": latent_hazard_main_event if get_latent_hazard else None # hardcoded for now : 47 is end of the observation period and 340 is 90% quantile surv time
		}
		return pred_y, hazards_y, all_extra_info

	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		# input_dim = starting_point.size()[-1]
		# starting_point = starting_point.view(1,1,input_dim)

		# Sample z0 from prior
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc
		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = starting_point_enc.size()
			# append a vector of zeros to compute the integral of lambda
			zeros = torch.zeros(n_traj_samples, n_traj, self.input_dim).to(self.device)
			starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
			n_traj_samples = 3)

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
		
		return self.decoder(sol_y)


