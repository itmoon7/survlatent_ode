import io
import pkgutil
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid

from lib.neural_ode_surv import *

from lib.utils import *

import torch
import gc
from datetime import date

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
	data = pd.read_csv('../neural_ode_surv/mimic_data/df_mimic_icu_lab_vals_treats_more_labs_36_hours_no_outliers.csv')  # df_mimic_icu_lab_vals_treats_full_meas.csv, df_mimic_icu_lab_vals_treats_first_48_hours
	
	feat_cat = ['gender']
	feat_cont = ['age', 'heart rate', 'respiratory rate', 'systolic blood pressure', 'diastolic blood pressure', 'mean blood pressure', 'oxygen saturation', 'temperature', 'glucose', 'central venous pressure', 'hematocrit', 'potassium', 'sodium', 'pulmonary artery pressure systolic', 'ph', 'hemoglobin', 'chloride', 'co2 (etco2, pco2, etc.)', 'partial pressure of carbon dioxide', 'creatinine', 'blood urea nitrogen', 'bicarbonate', 'platelets', 'anion gap', 'white blood cell count', 'magnesium', 'positive end-expiratory pressure set', 'calcium', 'tidal volume observed', 'partial thromboplastin time', 'red blood cell count', 'mean corpuscular volume', 'prothrombin time inr', 'prothrombin time pt', 'fraction inspired oxygen set', 'peak inspiratory pressure', 'calcium ionized', 'phosphate', 'respiratory rate set', 'phosphorous', 'tidal volume set']
	feat_reconstr = ['heart rate', 'respiratory rate', 'systolic blood pressure', 'diastolic blood pressure', 'mean blood pressure', 'oxygen saturation', 'temperature', 'glucose', 'central venous pressure', 'hematocrit', 'potassium', 'sodium', 'pulmonary artery pressure systolic', 'ph', 'hemoglobin', 'chloride', 'co2 (etco2, pco2, etc.)', 'partial pressure of carbon dioxide', 'creatinine', 'blood urea nitrogen', 'bicarbonate', 'platelets', 'anion gap', 'white blood cell count', 'magnesium', 'positive end-expiratory pressure set', 'calcium', 'tidal volume observed', 'partial thromboplastin time', 'red blood cell count', 'mean corpuscular volume', 'prothrombin time inr', 'prothrombin time pt', 'fraction inspired oxygen set', 'peak inspiratory pressure', 'calcium ionized', 'phosphate', 'respiratory rate set', 'phosphorous', 'tidal volume set']
	data_info_dic = {'id_col' : 'RANDID', 'event_col' : 'mort_hosp', 'time_col' : 'times', 'time_to_event_col' : 'tt_hos_mort_hours', 'feat_cat' : feat_cat, 'feat_cont' : feat_cont}
	
	if type(data_info_dic['event_col']) == list:
		n_events = len(data_info_dic['event_col'])
	else:
		n_events = 1

	feats_dim = len(feat_cat) + len(feat_cont)
	print('feat dimension : ', feats_dim)
	reconstr_dim = len(feat_reconstr)

	batch_size=100; gen_layers=7; latent_dim=40; lr=0.01; rec_dim=50; rec_layers=5; mult_event_units=5; num_layer_hazard_dec=3; survival_loss_scale=100; units_gru=80; units_ode=70; wait_until_full_surv_loss=3;

	n_latent_traj = 1 # number of sampled trajectories
	max_time = 600; n_samples = 30000; niters = 30; 
	include_test_set = True # include test set as a part of the main data pre-processing
	early_stopping = True; check_extrapolation = True; dataset = 'mimic'

	model = SurvLatentODE(input_dim = feats_dim, reconstr_dim = reconstr_dim, latent_dim = latent_dim, rec_dim = rec_dim, rec_layers = rec_layers, 
				 		  gen_layers = gen_layers, units_ode = units_ode, units_gru = units_gru, device = DEVICE, n_events = n_events)
	
	"""
	Select run_id
	"""
	run_id = 'mimic_data_train_test_valid_365' 
	
	# if not evaluate_only:
	model.fit(data, data_info_dic, n_samples = n_samples, max_time = max_time, run_id = run_id, niters = niters, batch_size = batch_size, include_test_set = include_test_set, survival_loss_scale = survival_loss_scale, early_stopping = early_stopping, feat_reconstr = feat_reconstr, dataset= dataset, wait_until_full_surv_loss = wait_until_full_surv_loss, check_extrapolation = check_extrapolation)

	# load model
	print('\n')
	print('Loading the best performance model...')
	print('run_id : ', run_id)
	path = 'model_performance/' + run_id + '/best_model.pt'
	model_info, _ = get_ckpt_model(path, model, DEVICE)
	
	print('processing test data...')
	batch_dict, min_max_tuple = model.process_test_data(data, data_info_dic, n_samples = n_samples, max_time = max_time, run_id = run_id, include_test_set = include_test_set, feat_reconstr = feat_reconstr, check_extrapolation = check_extrapolation, dataset = dataset, model_info = model_info)
	
	if n_latent_traj > 1:
		with torch.no_grad():
			surv_prob, surv_prob_var, rec_loss = model.get_surv_prob(None, None, batch_dict = batch_dict, model_info = model_info, max_time = max_time, filename_suffix = run_id, device = DEVICE, n_latent_traj = n_latent_traj, reconstr_loss = True, credible_interval = n_latent_traj > 1, n_events = n_events)
		np.save('model_performance/' + run_id + '/test_surv_prob.npy', surv_prob)
	else:
		with torch.no_grad():
			if n_events == 1:
				surv_prob, rec_loss = model.get_surv_prob(None, None, batch_dict = batch_dict, model_info = model_info, max_time = max_time, filename_suffix = run_id, device = DEVICE, n_latent_traj = n_latent_traj, reconstr_loss = True, credible_interval = n_latent_traj > 1, n_events = n_events, export_latent_states = True)
				np.save('model_performance/' + run_id + '/test_surv_prob.npy', surv_prob)
			else:
				ef_surv_prob, cs_cif_total, rec_loss = model.get_surv_prob(None, None, batch_dict = batch_dict, model_info = model_info, max_time = max_time, filename_suffix = run_id, device = DEVICE, n_latent_traj = n_latent_traj, reconstr_loss = True, credible_interval = n_latent_traj > 1, n_events = n_events)
	
	print('Evaluating the model...')
	# this is for hyper-param tuning only : 
	idx = 0
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		if n_events == 1:
			df_perf_result = pd.DataFrame([], index = np.arange(100), columns = ['run_id', 'params', 'best_epoch', 'reconstr_loss', 'mean_auc', 'mean_c_idx', 'mean_bs'])
			df_perf_result = evaluate_test_set(df_perf_result, model_info, batch_dict, surv_prob, rec_loss, run_id = run_id, max_event_time = max_time, dataset = dataset, idx = idx)
		else:
			df_perf_result = pd.DataFrame([], index = np.arange(100), columns = ['run_id', 'params', 'best_epoch', 'reconstr_loss', 'mean_auc_1', 'mean_c_idx_1', 'mean_bs_ef', 'mean_auc_2', 'mean_c_idx_2'])
			df_perf_result = evaluate_test_set(df_perf_result, model_info, batch_dict, ef_surv_prob, rec_loss, cs_cif_total = cs_cif_total, run_id = run_id, n_events = n_events, max_event_time = max_time, dataset = dataset, idx = idx)

	"""
	=====================================================================================
	"""
	del model
	gc.collect()
	return

if __name__ == '__main__':
	main()