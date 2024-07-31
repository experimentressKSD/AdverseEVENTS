import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from tcn import TCN
import keras
from scipy.interpolate import interp1d

def keep_unique(ids):
    unique_ids = []
    last_seen_id = None
    for id_ in ids:
        if id_ != last_seen_id:
            unique_ids.append(id_)
            last_seen_id = id_
    return unique_ids

def iterate_patients(ls_inputs, ls_targets, disease, out_path, model):

    predictions_path = out_path + disease + '/' + str(model) + '_predictions.npy'
    predictions = np.load(predictions_path)

    ls_patient_ids = []
    ls_sample_types = []
    ls_seq_numbers = []
    ls_predictions = []

    ls_predictions = [] # each element in the list is the set of predictions for a patient
    ls_targets_final = []

    for idx, path in enumerate(ls_inputs): # reconstruct patients and events from path names

        path_elements = path.split("/")

        patient_id = path_elements[5]
        ls_patient_ids.append(int(patient_id))

        data_type =  path_elements[6]
        if data_type == 'negative_samples':
            ls_sample_types.append(0)
        elif data_type == 'positive_samples':
            ls_sample_types.append(1)

        pattern = r'\d+'
        sequence_number = int(re.search(pattern, path_elements[7]).group()) # extract the number of the sample
        ls_seq_numbers.append(sequence_number)

        ls_predictions.append(predictions[idx])

        # target array
        #print(ls_inputs[idx], ls_targets[idx])
        target_arr = np.load(ls_targets[idx])
        ls_targets_final.append(target_arr)

    # Combine the lists into a structured array
    structured_data = np.array(list(zip(ls_patient_ids, ls_sample_types, ls_seq_numbers, ls_predictions, ls_targets_final)),
                    dtype=[('patient_id', int), ('label', int), ('sequence_number', int), ('predictions', '8float'), ('targets', '8int')])

    # Sort the structured array based on patient ID, label, and sequence number
    sorted_data = np.sort(structured_data, order=['patient_id', 'label', 'sequence_number'])

    reordered_np_array = sorted_data['predictions']
    reordered_targets = sorted_data['targets']
    ordered_pt_ids = sorted_data['patient_id']

    # remove duplicate patient IDs but keep the same order as before
    ordered_pt_ids = keep_unique(ordered_pt_ids)

    # split the np array of predictions into a list of predictions for each patient ID
    split_indices = [i for i, pid in enumerate(ordered_pt_ids) if i == 0 or pid != ordered_pt_ids[i-1]]

    # Split the 2D NumPy array into a list of 2D arrays
    split_arrays = [reordered_np_array[start:end] for start, end in zip(split_indices, split_indices[1:] + [None])]
    split_targets = [reordered_targets[start:end] for start, end in zip(split_indices, split_indices[1:] + [None])]

    # iterate through each patient ID, unrolling the predictions and summing them if there are overlapping predictions

    for (preds_pt, targets_pt, pt_id) in zip(split_arrays, split_targets, ordered_pt_ids):
        stride = 4
        overlap = 4
        nb_timesteps_per_window = 8
        original_array_size = (preds_pt.shape[0] - 1) * stride + nb_timesteps_per_window
        unrolled_array = np.zeros((original_array_size, ))
        unrolled_targets = np.zeros((original_array_size, ))

        # Unroll the rolling window data into the result array, maxing all the predictions
        for i, window_data in enumerate(preds_pt):
            start_index = i * stride
            end_index = start_index + 8
            unrolled_array[start_index:end_index] = np.maximum(window_data[:8], unrolled_array[start_index:end_index])
        # Unroll the rolling window targets into the result array, maxing all the targets
        for i, window_target in enumerate(targets_pt):
            start_index = i * stride
            end_index = start_index + 8
            unrolled_targets[start_index:end_index] = np.maximum(window_target[:8], unrolled_targets[start_index:end_index])

        directory = out_path + disease + '/' + str(model) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        final_concatenated_array = np.stack([unrolled_array, unrolled_targets], axis=0)
        np.save(directory + str(pt_id) + '_final_predictions.npy', final_concatenated_array)


def compute_results(disease, out_path, model):

    disease_dir = out_path + disease + '/'

    directory = disease_dir + str(model) + '/'

    total_pts = 0

    total_events = 0
    captured_events = 0
    precision_total = []

    # true_alarms_final = 0
    # total_alarms_final = 0

    onset_indices_csv = pd.read_csv(disease_dir + 'onset_index.csv', header=None, index_col=False)

    max_time_to_onset = 0
    min_time_to_onset = 0
    
    
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):

            total_pts += 1

            split_filename = filename.split("_")
            pt_id = int(split_filename[0])
            onset_index = (onset_indices_csv[onset_indices_csv[0] == pt_id][1].values[0])
            file_path = os.path.join(directory, filename)
            loaded_array = np.load(file_path)
            predictions = (loaded_array[0] >= 0.5).astype(int)

            # construct a target array that is positive 8H (8 * 2 = 16 timesteps) before the onset index, and 0 immediately after the onset time, and 0 the rest of the time.
            targets = (np.zeros((predictions.shape))).astype(int)
            if np.isnan(onset_index) == False:
                #print(onset_index)
                onset_index = int(onset_index)
                start_event = max(0, onset_index - 8 * 2)
                targets[start_event:onset_index]  = 1

                targets = targets[:onset_index]
                predictions = predictions[:onset_index]

            # targets = loaded_array[1].astype(int)
            # #predictions = np.random.randint(2, size=targets.shape[0])

            total_alarms = np.sum(predictions)
            bitwise_and = targets & predictions
            true_alarms = np.sum(bitwise_and)

            # true_alarms_final += true_alarms
            # total_alarms_final += total_alarms
                
            precision = true_alarms/total_alarms
            precision_total.append(precision)

            positive_event = np.sum(targets) > 0
            if positive_event:
                total_events += 1
                if true_alarms > 0:
                   captured_events += 1

    final_precision = np.nanmean(precision_total)
    #final_precision = true_alarms_final/total_alarms_final
    final_recall = captured_events/total_events
    print("Precision: "+ str(final_precision) + " Recall: " + str(final_recall))
    print("Total patients: " + str(total_pts) + ", total positive patients: " + str(total_events))

    #
        
def parallelize_pt_eval(pt_id, vitals_folder_path, feature_means, feature_stds, stride, model, save_npy_path, run_output_path):
    try: 
        pt_id = int(pt_id)

        # get the vitals as inputs
        vitals_file = pd.read_csv(vitals_folder_path + str(pt_id) + '_vitals.csv', index_col=False)
        vitals_file['time'] = pd.to_datetime(vitals_file['time'])
        vitals_file = vitals_file.set_index('time')
        vitals = vitals_file.drop(['Unnamed: 0', 'id'], axis=1).resample('30T').mean() 
        vitals = vitals.ffill().bfill()

        
        #vitals = (np.array(vitals) - feature_means) / feature_stds

        # to tf.dataset
        input_dataset = tf.keras.utils.timeseries_dataset_from_array(np.array(vitals), None, sequence_length = 4*2, sequence_stride = stride, batch_size=None)

        # interpolate vitals
        input_npy_list = []
        decoder_input_list = []
        for idx, tensor in enumerate(input_dataset):
            input_npy = tensor.numpy()
            # input_npy = np.delete(input_npy, 3, axis=1)
            original_nb_timesteps = input_npy.shape[0] # 4 hr window with 30 min timesteps --> 4 *2 = 8
            new_nb_timesteps = 48 # 4 hr window with 5 min timesteps --> 4 * 12 = 48

            old_time = np.linspace(0, 1, original_nb_timesteps)
            new_time = np.linspace(0, 1, new_nb_timesteps)
            new_arr = np.zeros((new_nb_timesteps, input_npy.shape[1]))

            # Interpolate each feature
            for i in range(input_npy.shape[1]):
                interpolator = interp1d(old_time, input_npy[:, i], kind='linear')  # Change to 'cubic' for spline
                new_arr[:, i] = interpolator(new_time)

            input_npy = new_arr
            input_npy = (input_npy - feature_means) / feature_stds
            decoder_inputs = np.zeros((input_npy.shape[0], 1))
            decoder_input_list.append(decoder_inputs)
            input_npy_list.append(input_npy)

        input_dataset = tf.data.Dataset.from_tensor_slices(input_npy_list)
        decoder_input_dataset = tf.data.Dataset.from_tensor_slices(decoder_input_list)
        zipped_dataset = tf.data.Dataset.zip(((input_dataset, decoder_input_dataset), )).batch(128)

        # inference with model
        saved_model = keras.models.load_model(run_output_path + str(model) + '_weights.keras', custom_objects={'TCN': TCN})
        predictions_probability = saved_model.predict(zipped_dataset)
        np.save(save_npy_path + str(pt_id) + '_preds.npy', predictions_probability)

    except Exception as e:
        print("skipped patient due to Error: ", e)

def evaluate_indiv(test_pt_ids, vitals_folder_path, out_path, disease, model, run_output_path, target_shape):
        
    feature_means = np.load(run_output_path + 'training_means.npy')
    feature_stds = np.load(run_output_path + 'training_stds.npy')

    # read positive ICD patients
    pos_ICD_pts = []
    path = '/datasets/amelatur/data_slices/ICD_pt_ids/' + disease + '/pos_pts.txt'
    with open (path, 'r') as f:
        for line in f:
            pos_ICD_pts.append(line.strip())

    # read negative ICD patients
    neg_ICD_pts = []
    path = '/datasets/amelatur/data_slices/ICD_pt_ids/' + disease + '/neg_pts.txt'
    with open (path, 'r') as f:
        for line in f:
            neg_ICD_pts.append(line.strip())

    all_ICD_pts = pos_ICD_pts + neg_ICD_pts

    all_test_ICD_pts = list(set(all_ICD_pts) & set(test_pt_ids))

    stride = 1

    save_npy_path = run_output_path + 'raw_predictions/'
    if os.path.exists(save_npy_path):
        # If it exists, remove it
        shutil.rmtree(save_npy_path)
    
    # Create the directory
    os.makedirs(save_npy_path)

    Parallel(n_jobs=-1)(delayed(parallelize_pt_eval)(pt_id, vitals_folder_path, feature_means, feature_stds, stride, model, save_npy_path, run_output_path) for (pt_id) in tqdm(all_test_ICD_pts))