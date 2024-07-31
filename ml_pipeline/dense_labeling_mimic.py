import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from ml_pipeline.helper_files.config_vars_labeling import get_circ_fail_vars, get_path_vars, get_respi_fail_vars, get_sepsis_vars
from ml_pipeline.time_series_relabeling import get_input_array, get_target_array, save_data_slices

from csv import writer

# TODO: artifact removal --> read data ranges from yaml file

def get_respi_fail_hirid_method(patient_id):

    main_folder_path, peep_interpolate_limit, ffill_p_f_hours, window_length, p_f_ratio_threshold, window_ratio, pao2_normal, fio2_normal, _, _ = get_respi_fail_vars('../labeling_config.yaml')

    labs_file = pd.read_csv(main_folder_path + str(patient_id) + '_all_vals.csv', index_col=False)
    labs_file['charttime'] = pd.to_datetime(labs_file['charttime'])
    labs_file = labs_file.sort_values(by='charttime').set_index('charttime')

    index_ = labs_file.index
    n_steps = len(labs_file.index)

    # annotate respiratory failure
    respi_status = pd.Series(0, index=index_)

    # P/F ratio
    if not (labs_file['po2'].isna().all() or n_steps < 24):
        exclusion_flag = 0
        p_f_raw = labs_file[['po2', 'fio2']].reindex(index_, method='ffill', limit=ffill_p_f_hours) 
        p_f_raw = p_f_raw.interpolate(limit=ffill_p_f_hours)
        p_f_raw['po2'] = p_f_raw['po2'].fillna(pao2_normal) 
        p_f_raw['fio2'] = p_f_raw['fio2'].fillna(fio2_normal)
        ratio = 100 * p_f_raw['po2'] / p_f_raw['fio2'] 

        p_f_ratio =  pd.Series(ratio, index=index_)

        # get ventilation status
        vent_df = labs_file['vent_presence']

        # peep 
        peep = labs_file['peep'].reindex(index_, method='ffill', limit=peep_interpolate_limit).squeeze()

        # take element-wise AND of peep and ventilation
        vent_peep_status_dense = vent_df.squeeze() & peep.squeeze() > 4

        peep[peep.isnull()] = 1
        peep[~peep.isnull()] = 0
        vent_peep_status_not_dense = (vent_df.squeeze()) & peep
        
        # annotate respiratory failure
        respi_status = pd.DataFrame(0, index=index_, columns=['respi_failure']).squeeze()

        n_steps = len(vent_peep_status_not_dense)

        for i in range(n_steps):
            start_window = i
            end_window = min(n_steps, i + window_length)

            vent_win = np.array(vent_peep_status_dense[start_window:end_window])
            vent_win_not_dense = np.array(vent_peep_status_not_dense[start_window:end_window])
            p_f_win = np.array(p_f_ratio[start_window:end_window])
            
            no_vent_cond = np.sum((p_f_win < p_f_ratio_threshold) & (vent_win == False)) >= window_ratio * len(vent_win)
            vent_cond_not_dense = np.sum((p_f_win < p_f_ratio_threshold)& (vent_win_not_dense)) >= window_ratio * len(vent_win)
            vent_cond_dense = np.sum((p_f_win) < p_f_ratio_threshold & (vent_win)) >= window_ratio * len(vent_win)

            final = no_vent_cond + vent_cond_not_dense + vent_cond_dense
            if final: 
                respi_status.iloc[i] = 1
    else:
        exclusion_flag = 1

    return exclusion_flag, respi_status

def get_respi_fail_NEJM (patient_id):

    main_folder_path, _, ffill_p_f_hours, window_length, p_f_ratio_threshold, window_ratio, pao2_normal, fio2_normal, paco2_normal, ph_normal = get_respi_fail_vars('../labeling_config.yaml')

    labs_file = pd.read_csv(main_folder_path + str(patient_id) + '_all_vals.csv', index_col=False)
    labs_file['charttime'] = pd.to_datetime(labs_file['charttime'])
    labs_file = labs_file.set_index('charttime')

    index_ = labs_file.index
    n_steps = len(labs_file.index)
    respi_status = pd.Series(0, index=index_)

    if not (labs_file['po2'].isna().all() or n_steps < 24):

        exclusion_flag = 0
        p_f_raw = labs_file[['po2', 'spo2', 'fio2']]
        p_f_raw = p_f_raw.ffill(limit=ffill_p_f_hours)
        p_f_raw = p_f_raw.interpolate(limit=ffill_p_f_hours)
        p_f_raw['po2'] = p_f_raw['po2'].fillna(pao2_normal) 
        p_f_raw['spo2'] = p_f_raw['spo2'].fillna(95)
        p_f_raw['fio2'] = p_f_raw['fio2'].fillna(fio2_normal)

        po2 = p_f_raw['po2'].squeeze()

        paCO2_pH = labs_file[['pco2', 'ph']]
        paCO2_pH = paCO2_pH.ffill(limit=ffill_p_f_hours)
        paCO2_pH = paCO2_pH.interpolate(limit=ffill_p_f_hours)
        paCO2_pH['pco2'] = paCO2_pH['pco2'].fillna(paco2_normal) 
        paCO2_pH['ph'] = paCO2_pH['ph'].fillna(ph_normal)
        paco2 = paCO2_pH['pco2'].squeeze()
        pH = paCO2_pH['ph'].squeeze()
            
        for i in range(n_steps):
            start_window = i
            end_window = min(n_steps, i + window_length)

            paco2_win = np.array(paco2[start_window:end_window])
            pH_win = np.array(pH[start_window:end_window])
            po2_win = np.array(po2[start_window:end_window])

            hypoxia = np.sum(((po2_win < 55))) >= window_ratio * len(paco2_win)
            hypercapnia = np.sum((paco2_win > 45) & (pH_win < 7.35)) >= window_ratio * len(paco2_win)

            final = hypoxia + hypercapnia

            if final: 
                respi_status.iloc[i] = 1
        
    else:
        exclusion_flag = 1

    return exclusion_flag, respi_status

def get_circ_fail(patient_id):

    main_folder_path, ffill_lactate_hours, ffill_drugs_hours, ffill_map_hours, lactate_normal_value, map_normal_value, lactate_threshold, map_threshold, window_ratio, window_length, hr_normal_value, hr_threshold = get_circ_fail_vars('../labeling_config.yaml')

    labs_file = pd.read_csv(main_folder_path + str(patient_id) + '_all_vals.csv', index_col=False)
    labs_file['charttime'] = pd.to_datetime(labs_file['charttime'])
    labs_file = labs_file.set_index('charttime')
    index_ = labs_file.index
    n_steps = len(labs_file.index)
    circ_status = pd.Series(0, index=index_)

    if not (n_steps < 24 or labs_file['mbp'].isna().all() or labs_file['lactate'].isna().all()):

        exclusion_flag = 0
        lactate_final = labs_file['lactate']
        lactate_final = lactate_final.ffill(limit=ffill_lactate_hours)
        lactate_final.loc[:lactate_final.first_valid_index()] = lactate_normal_value
        lactate_final = lactate_final.interpolate(limit=ffill_lactate_hours)
        lactate_final = lactate_final.fillna(lactate_normal_value)

        drug_df = labs_file['drug_presence'] # to series

        map_values = labs_file['mbp']
        map_values = map_values.ffill(limit=ffill_map_hours)
        map_values.loc[:map_values.first_valid_index()] = map_normal_value
        map_values = map_values.interpolate(limit=ffill_map_hours)
        map_values = map_values.fillna(map_normal_value)

        # Now calculate circ failure on imputed dataframes
        # final circ_status column filled with 0s
        event_window_length = window_length
        half_length = event_window_length//2

        # EXTRA STUFF: HR --> impute in the same manner as all values above (MAP, lactate)
        hr = labs_file['heart_rate']
        hr = hr.ffill(limit=ffill_map_hours)
        hr.loc[:hr.first_valid_index()] = hr_normal_value
        hr = hr.interpolate(limit=ffill_map_hours)
        hr = hr.fillna(hr_normal_value)

        # urine output
        uo = labs_file['uo_rt_6hr']
        uo = uo.ffill()
        uo = uo.fillna(1.5)

        for idx in range(n_steps):

            start_idx = max(0, idx - half_length)
            end_idx = min(n_steps, idx + half_length)

            map_wind = np.array(map_values[start_idx:end_idx+1])
            lactate_wind = np.array(lactate_final[start_idx:end_idx+1])
            drugs_wind = np.array(drug_df[start_idx:end_idx+1])
            hr_wind = np.array(hr[start_idx:end_idx+1])
            uo_wind = np.array(uo[start_idx:end_idx+1])

            pharma_cond = drugs_wind > 0
            map_cond = map_wind < 65
            hr_cond = hr_wind > hr_threshold
            uo_cond = uo_wind < 0.5

            map_full_cond = (map_cond | pharma_cond)
            #map_full_cond = map_cond
            lact_crit_arr = (lactate_wind > 2)
            map_state = np.sum(map_full_cond) >= window_ratio * len(map_full_cond)
            lac_state = np.sum(lact_crit_arr) >= window_ratio * len(map_full_cond)
            uo_state = np.sum(uo_cond) >= window_ratio * len(map_full_cond)
            hr_state = np.sum(hr_cond) >= 1/2 * len(map_full_cond)
            secondary = lac_state or uo_state
            if map_state and lac_state:
                circ_status.iloc[idx] = 1.0

    else:
        exclusion_flag = 1
    
    return exclusion_flag, circ_status

def get_kidney_fail(patient_id):

    config, main_folder_path = get_path_vars('../labeling_config.yaml')

    labs_file = pd.read_csv(main_folder_path + str(patient_id) + '_all_vals.csv', index_col=False)
    labs_file['charttime'] = pd.to_datetime(labs_file['charttime'])
    labs_file = labs_file.set_index('charttime')

    n_steps = len(labs_file.index)
    if (n_steps < 24 or labs_file['aki_stage'].isna().all()):
        exclusion_flag = 1
    else:
        exclusion_flag = 0

    kidney_status = pd.Series(0, index=labs_file.index)

    target_value = 2.0
    earliest_index = labs_file['aki_stage'].squeeze().ge(target_value)
    if earliest_index.any() == True:
        earliest_index = labs_file.index.get_loc(earliest_index.idxmax())
        kidney_status[earliest_index:] = 1

    return exclusion_flag, kidney_status

def get_sepsis(patient_id):

    config, main_folder_path = get_path_vars('../labeling_config.yaml')
    ids_file = pd.read_csv( main_folder_path + '0labels.txt')
    detected_sepsis = ids_file[ids_file['id'] == patient_id]['sepsis'].values[0]
    
    labs_file = pd.read_csv(main_folder_path + str(patient_id) + '_all_vals.csv', index_col=False)
    labs_file['charttime'] = pd.to_datetime(labs_file['charttime'])
    labs_file = labs_file.set_index('charttime')

    n_steps = len(labs_file.index)
    exclusion_flag = 0
    sepsis_index = None
    sepsis_time = None

    sepsis_status = pd.Series(0, index=labs_file.index, name="sepsis")
    if (n_steps <24):
        exclusion_flag = 1
    if detected_sepsis == True:
        sepsis_time = pd.to_datetime(ids_file[ids_file['id'] == patient_id]['sepsis_time'].values[0])
        first_vital = min(labs_file.index)
        if (sepsis_time - first_vital) < pd.Timedelta(hours=4):
            exclusion_flag = 1
        sepsis_index = labs_file.index.get_loc(sepsis_time)
        sepsis_status[sepsis_index:] = 1
    
    return exclusion_flag, sepsis_status


def parallel_labeling(out_folder, str_disease, pat_id, failure_labeling_function, vitals_folder_path, whole_sequence_training):

    if whole_sequence_training:
        whole_sequences_path = out_folder + str_disease + '/whole_sequences/'
        exclusion_flag, respi_status = failure_labeling_function(pat_id)

    else:

        try:

            data_slices_path = out_folder + str_disease + '/' + str(pat_id)

            exclusion_flag, respi_status = failure_labeling_function(pat_id)
            exclusion_flag, target_dataset_negative, target_dataset_positive, result_index, len_neg_tensors = get_target_array(exclusion_flag, respi_status)
            input_dataset_negative, input_dataset_positive, exclusion_flag = get_input_array(pat_id, vitals_folder_path, exclusion_flag, result_index, len_neg_tensors)
            save_data_slices(input_dataset_negative, target_dataset_negative, input_dataset_positive, target_dataset_positive, exclusion_flag, data_slices_path)

            path_onset_index = out_folder + str_disease + '/'
            if not(os.path.exists(path_onset_index)):
                os.makedirs(path_onset_index)

            path_onset_index = out_folder + str_disease + '/'
            row = [pat_id, result_index]
            with open(path_onset_index + 'onset_index.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(row)
                f_object.close()
        except Exception as e:
            print(type(e).__name__)
            print('Skipped patient ' + str(pat_id) + ' due to error')



def disease_labeling(str_disease, pt_ids, vitals_folder_path, out_folder, whole_sequence_training):
    disease_function_map = {
        'respiratory_HiRID': get_respi_fail_hirid_method,
        'respiratory_NEJM': get_respi_fail_NEJM,
        'circulatory': get_circ_fail,
        'sepsis': get_sepsis,
        'kidney': get_kidney_fail,
    }
    print(str_disease, out_folder)

    failure_labeling_function = disease_function_map.get(str_disease)

    if failure_labeling_function:
        Parallel(n_jobs=-1)(delayed(parallel_labeling)(out_folder, str_disease, pat_id, failure_labeling_function, vitals_folder_path) for (idx, pat_id) in tqdm(pt_ids.items()))



