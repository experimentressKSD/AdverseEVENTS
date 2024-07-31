import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
from csv import writer

from ml_pipeline.dense_labeling_mimic import get_respi_fail_hirid_method, get_respi_fail_NEJM, get_circ_fail, get_sepsis, get_kidney_fail

def create_inputs_outputs(vitals_path, pat_id, status, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, labeled_disease, max_len_stay_days):

    exclusion_flag = 0

    # create output/target
    threshold = 3
    result_index = status[status.groupby((status != status.shift()).cumsum()[status.eq(1)]).transform('count') > threshold]
    if result_index.empty == False:
        result_index = status.index.get_loc(result_index.idxmax())
    else:
        result_index = None

    n_steps = len(status.index)

    start_event_labeling = before_onset_positive * (60 //30)
    end_event_labelling = after_onset_positive * (60 //30)

    # get vitals as inputs
    vitals_file = pd.read_csv(vitals_path + str(pat_id) + '_vitals.csv', index_col=False)
    vitals_file['time'] = pd.to_datetime(vitals_file['time'])
    vitals_file = vitals_file.set_index('time')
    vitals_file = vitals_file.drop(['Unnamed: 0', 'id'], axis=1)

    ranges = {'heartrate':(0.0, 300.0), 'sbp': (10.0, 300.0), 'dbp': (10.0, 175.0), 'mbp': (10.0, 200.0), 'respiration': (0.0, 45.0), 'temperature': (25.0, 45.0), 'spo2': (10.0, 100.0)}

    # Replace values with NaNs if they are outside the specified range for each column
    vitals_file = vitals_file.apply(lambda col: np.where((col < ranges[col.name][0]) | (col > ranges[col.name][1]), np.nan, col))
    vitals = vitals_file.resample('30T').mean() 
    vitals = vitals.interpolate().ffill().bfill()  

    # drop MBP if disease is circulatory failure to avoid feature circularity
    if labeled_disease == 'circulatory':
        vitals = vitals.drop('mbp', axis=1)
    

    target = np.zeros(vitals.shape[0])

    if result_index != None:

        if result_index < 8:
            exclusion_flag = 1

        start_event = max(0, result_index - start_event_labeling)
        end_event = min(n_steps, result_index + end_event_labelling)
        target[start_event:end_event] = 1 

        if end_pos_seq_after_onset:
            vitals = vitals[:end_event]
            target = target[:end_event]

    else: # negative patient stay
        target = np.zeros(vitals.shape[0])


    if vitals.isna().any().any():
        exclusion_flag = 1
    vitals = vitals.to_numpy()



    return vitals, target, exclusion_flag, result_index
            




def parallel_labeling_whole_sequences(whole_seqs_out_path, labeled_disease, pat_id, failure_labeling_function, vitals_path,  before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, max_len_stay_days):

    max_seq_length = 336
    padding_value = 100000.0

    

    try:

        exclusion_flag, status = failure_labeling_function(pat_id)

        if exclusion_flag == 0:
            vitals, target, exclusion_flag_out, result_index = create_inputs_outputs(vitals_path, pat_id, status, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, labeled_disease, max_len_stay_days)

            if exclusion_flag_out == 0:
                
                row = [pat_id, result_index]
                with open(whole_seqs_out_path + 'onset_index.csv', 'a') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(row)
                    f_object.close()


                np.save(whole_seqs_out_path + str(pat_id) + '_raw_vitals.npy', vitals)
                np.save(whole_seqs_out_path + str(pat_id) + '_raw_target.npy', target)

    except Exception as e:
        print("Error in patient ", pat_id, " with error: ", e)
        pass


        




def disease_labeling(labeled_disease, all_pt_ids, vitals_path, whole_seqs_out_path, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, max_len_stay_days):
    disease_function_map = {
        'respiratory_HiRID': get_respi_fail_hirid_method,
        'respiratory_NEJM': get_respi_fail_NEJM,
        'circulatory': get_circ_fail,
        'sepsis': get_sepsis,
        'kidney': get_kidney_fail,
    }
    print(labeled_disease, whole_seqs_out_path)

    failure_labeling_function = disease_function_map.get(labeled_disease)

    if failure_labeling_function:
        Parallel(n_jobs=-1)(delayed(parallel_labeling_whole_sequences)(whole_seqs_out_path, labeled_disease, pat_id, failure_labeling_function, vitals_path,  before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, max_len_stay_days) for (idx, pat_id) in tqdm(all_pt_ids.items()))

def get_input_output_eicu(vitals_path, vitals_type, pat_id, labels_file, before_onset_positive, after_onset_positive, whole_seqs_out_path, max_len_stay_days):

    exclusion_flag = 0
    indices_to_onset = labels_file[labels_file['patient_ID'] == (pat_id)]['indices_to_onset'].values[0]
    if np.isnan(indices_to_onset) == False:
        if indices_to_onset < 2 * (60//5) or indices_to_onset > (max_len_stay_days - 1) * 24 * (60//5) : # onset less than 2 hours from start or after max_days - 1 days
            exclusion_flag = 1
    
    stay_length = labels_file[labels_file['patient_ID'] == (pat_id)]['stay_length'].values[0]
    if stay_length < 12 * (60//5) or stay_length > 7 * 24 * (60//5) or np.isnan(stay_length): # if stay length less than 12H or more than 7 days
        exclusion_flag = 1
    
    if exclusion_flag == 0:

        if vitals_type == 'central':
            vitals_of_interest = ['spo2', 'heartrate', 'respiration', 'time']
        elif vitals_type == 'no_bp':
            vitals_of_interest = ['temperature', 'spo2', 'heartrate', 'respiration', 'time']
        elif vitals_type == 'all':
            vitals_of_interest = ['temperature', 'spo2', 'heartrate', 'respiration', 'sbp', 'dbp', 'mbp', 'time']

        vitals_csv = pd.read_csv(vitals_path + str(pat_id) + '.csv', index_col=False)
        vitals_subset = vitals_csv[vitals_of_interest]
        #vitals_subset['time'] = pd.to_datetime(vitals_subset['time'])
        vitals_subset = vitals_subset.set_index('time')
        
        ranges = {'heartrate':(0.0, 300.0), 'sbp': (10.0, 300.0), 'dbp': (10.0, 175.0), 'mbp': (10.0, 200.0), 'respiration': (0.0, 45.0), 'temperature': (25.0, 45.0), 'spo2': (10.0, 100.0)}

        # Replace values with NaNs if they are outside the specified range for each column
        vitals_subset = vitals_subset.apply(lambda col: np.where((col < ranges[col.name][0]) | (col > ranges[col.name][1]), np.nan, col))

        vitals_subset = vitals_subset.interpolate().bfill().ffill()

        # if any NaNs in the vitals, exclude the patient
        if vitals_subset.isna().any().any():
            exclusion_flag = 1

        if exclusion_flag == 0:
            
            vitals = vitals_subset.to_numpy()

            target = np.zeros(vitals.shape[0])

            if np.isnan(indices_to_onset) == False: # positive patient stay

                start_event = int(max(0, indices_to_onset - int(before_onset_positive * (60//5))))
                end_event = int(min(vitals.shape[0], int(indices_to_onset + after_onset_positive * (60//5))))

                target[start_event:end_event] = 1

                vitals = vitals[:end_event, :]
                target = target[:end_event]

            np.save(whole_seqs_out_path + str(pat_id) + '_raw_vitals.npy', vitals)
            np.save(whole_seqs_out_path + str(pat_id) + '_raw_target.npy', target)



def raw_data_saving(labeled_disease, all_pt_ids, vitals_path, whole_seqs_out_path, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, vitals_type, labels_file, max_len_stay_days):

    Parallel(n_jobs=-1)(delayed(get_input_output_eicu)(vitals_path, vitals_type, pat_id, labels_file, before_onset_positive, after_onset_positive, whole_seqs_out_path, max_len_stay_days) for (idx, pat_id) in tqdm(all_pt_ids.items()))

