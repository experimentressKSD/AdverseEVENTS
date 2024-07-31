import pandas as pd
import numpy as np
import random
import os
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

from ml_pipeline.models.models_whole_seqs import create_lookahead_mask, create_full_mask


def pt_split_seqs(whole_seqs_out_path, patient_ids):

    train_size = 0.8
    train_patients = patient_ids.sample(frac=train_size, random_state=0)
    test_patients = patient_ids.drop(train_patients.index)
    print('Test patients:', len(test_patients))

    # validation
    validation_size = 0.15
    validation_patients = train_patients.sample(frac=validation_size, random_state=0)
    train_patients = train_patients.drop(validation_patients.index)
    print('Validation patients:', len(validation_patients))
    print('Train patients:', len(train_patients))

    with open(whole_seqs_out_path + 'test_pt_ids.txt', 'w') as f:
        for pt in test_patients:
            f.write("%s\n" % pt)
    with open(whole_seqs_out_path + 'validation_pt_ids.txt', 'w') as f:
        for pt in validation_patients:
            f.write("%s\n" % pt)
    with open(whole_seqs_out_path + 'train_pt_ids.txt', 'w') as f:
        for pt in train_patients:
            f.write("%s\n" % pt)

    return train_patients, test_patients, validation_patients

def pt_split_seqs_k_fold(whole_seqs_out_path, patient_ids):

    num_folds = 5
    patient_ids = patient_ids.sample(frac=1).reset_index(drop=True)

    test_sets = np.array_split(patient_ids, 5)
    test_sets = [pd.Series(test_set) for test_set in test_sets]

    for idx, test_set in enumerate(test_sets):
        train_set = patient_ids[~patient_ids.isin(test_set)]

        fold_path = whole_seqs_out_path + 'fold_' + str(idx) + '/'
        
        if os.path.isdir(fold_path) == False: #if the fold path does not exist, create it
            os.mkdir(fold_path)

            with open(fold_path + 'test_pt_ids.txt', 'w') as f:
                for pt in test_set:
                    f.write("%s\n" % pt)

            with open(fold_path + 'train_pt_ids.txt', 'w') as f:
                for pt in train_set:
                    f.write("%s\n" % pt)

def data_generator_whole_seqs(patient_ids_selected, whole_seqs_out_path, mode, ones_weight, zeros_weight, model_str):

    for pat_id in patient_ids_selected:
        
        vital_path = whole_seqs_out_path + str(pat_id) + '_vitals.npy'
        target_path = whole_seqs_out_path + str(pat_id) + '_target.npy'

        target = np.load(target_path)
        vitals = np.load(vital_path)

        if mode == 'train':
            attention_mask = create_lookahead_mask(seq_len=target.shape[0])
        elif mode == 'test':
            attention_mask = create_lookahead_mask(seq_len=target.shape[0])

        sample_weights = np.ones((target.shape))
        sample_weights[target == 0] = zeros_weight # make sure the weights sum to 1
        sample_weights[target == 1] = ones_weight

        #decoder dummy input
        decoder_input = np.zeros((vitals.shape[0], 1))

        # fill with padding value
        first_padded_index = np.where(target == 100000.0)[0]
        if len(first_padded_index) > 0:
            first_padded_index = first_padded_index[0]
            decoder_input[first_padded_index:, :] = 100000.0    

        if model_str == 'transformer':
            yield (vitals, attention_mask), target, sample_weights
        elif model_str == 'LSTM_enc_dec':
            yield (vitals, decoder_input), target
        elif model_str == 'LSTM' or model_str == 'TCN':
            yield vitals, target, sample_weights
        else: # exit if model_str is not recognized
            print("Model string not recognized")
            exit(1)

def output_signature(model_str, input_shape):
    if model_str == 'transformer':
        output_signature=(
                (tf.TensorSpec(shape=input_shape, dtype=tf.float32), tf.TensorSpec(shape=(input_shape[0], input_shape[0]), dtype=tf.float32)), # vitals and attention mask
            tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32),tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32) # target and sample weights
        )
    elif model_str == 'LSTM_enc_dec':
        output_signature=(
                (tf.TensorSpec(shape=input_shape, dtype=tf.float32), tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32)), # vitals and decoder dummy input
            tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32), # target
        )
    elif model_str == 'LSTM' or model_str == 'TCN':
        output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32), # vitals
            tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32), # target
            tf.TensorSpec(shape=(input_shape[0], 1), dtype=tf.float32) # sample weights
        )
    else: # exit if model_str is not recognized
        print("Model string not recognized")
        exit(1)       

    return output_signature


def whole_seqs_data_loader(batch_size, patient_ids_selected, whole_seqs_out_path, mode, input_shape, ones_weight, zeros_weight, model_str):

    # Shuffle the patient ids
    if mode == 'train':
        random.shuffle(patient_ids_selected)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_whole_seqs(patient_ids_selected, whole_seqs_out_path, mode, ones_weight, zeros_weight, model_str), 
        output_signature=output_signature(model_str, input_shape))

    # Shuffle the data
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)# prefetch for increased performance

    return dataset


def pt_split_random(disease, out_path, run_output_path):

    training_split_size = 0.8 # take 20% of all patients as test set patients

    data_slices = out_path + disease + '/'
    pt_ids = [file for file in os.listdir(data_slices)]
    n_total_pts = len(pt_ids)
    print("Data slices path: ", data_slices)
    mask_arr_training = np.random.binomial(n=1, p=training_split_size, size=[n_total_pts])
    mask_arr_test = 1 - mask_arr_training
    training_pts = [pt for pt, keep in zip(pt_ids, mask_arr_training) if keep]
    test_pts = [pt for pt, keep in zip(pt_ids, mask_arr_test) if keep]

    with open(run_output_path + 'test_pt_ids.txt', 'w') as f:
        for pt in test_pts:
            f.write("%s\n" % pt)

    print('There are ' + str(len(pt_ids)) + ' included patients in the dataset, ' + str(len(training_pts)) + ' of which are used for training and ' + str(len(test_pts)) + ' of which are used for testing.')

    return training_pts, test_pts


    

def get_training_window_paths(disease, pt_id_strings, out_path):

    # get the list of paths dof all positive samples (inputs & targets)
    ls_inputs_pos = []
    ls_targets_pos = []

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

    for pt_id in pt_id_strings:

        if pt_id in pos_ICD_pts:

            pt_inputs = []
            pt_targets = []
            path_to_files = out_path + str(disease) + '/' + str(pt_id) + '/'
            pos_samples = path_to_files + 'positive_samples/'
            #print(pos_samples)

            if os.path.isdir(pos_samples): #if there are positive samples for the given pt_id
                files_positive = os.listdir(pos_samples)
                for file in files_positive:
                    if file.startswith('input'):
                        pt_inputs.append(pos_samples + file)
                    if file.startswith('target'):
                        pt_targets.append(pos_samples + file)

            # guarantee that ls inputs and ls targets are in the same order by sorting each alphanumerically
            pt_inputs.sort()
            pt_targets.sort()

            ls_inputs_pos += pt_inputs
            ls_targets_pos += pt_targets

    ls_inputs_neg = []
    ls_targets_neg = []

    # first get the list of paths of all negative samples
    for pt_id in pt_id_strings:

        if pt_id in neg_ICD_pts:    
            pt_inputs = []
            pt_targets = []
            path_to_files = out_path + str(disease) + '/' + str(pt_id) + '/'
            
            pos_samples = path_to_files + 'positive_samples/'
            neg_samples = path_to_files + 'negative_samples/'
            if os.path.isdir(pos_samples): #if there are positive samples for the given pt_id --> SKIP THIS PATIENT!
                continue
            elif os.path.isdir(neg_samples): #if there are negative samples for the given pt_id
            
                files_negative = os.listdir(neg_samples)
                for file in files_negative:
                    if file.startswith('input'):
                        pt_inputs.append(neg_samples + file)
                    if file.startswith('target'):
                        pt_targets.append(neg_samples + file)
            pt_inputs.sort()
            pt_targets.sort()

            ls_inputs_neg += pt_inputs
            ls_targets_neg += pt_targets
    # now randomly sample from the list of paths of all negative samples, such that we get the same nb of negative windows as positive ones
    probability_negative_to_keep = (len(ls_inputs_pos) * 1)/len(ls_inputs_neg)
    
    mask_arr_neg = np.random.binomial(n=1, p=probability_negative_to_keep, size=[len(ls_inputs_neg)])
    ls_inputs_neg = [window for window, keep in zip(ls_inputs_neg, mask_arr_neg) if keep]
    ls_targets_neg = [window for window, keep in zip(ls_targets_neg, mask_arr_neg) if keep]

    ls_inputs_pos += ls_inputs_neg
    ls_targets_pos += ls_targets_neg # concatenate positive and negative window paths

    return ls_inputs_pos, ls_targets_pos # return the lists of paths of the data windows

def get_test_window_paths(disease, out_path, test_pt_ids):

    ls_inputs = []
    ls_targets = []

    for pt_id in test_pt_ids:

        path_to_files = out_path + str(disease) + '/' + str(pt_id) + '/'
        pos_samples = path_to_files + 'positive_samples/'

        if os.path.isdir(pos_samples): #if there are positive samples for the given pt_id

            pt_inputs = []
            pt_targets = []
            files_positive = os.listdir(pos_samples)
            for file in files_positive:
                if file.startswith('input'):
                    pt_inputs.append(pos_samples + file)
                if file.startswith('target'):
                    pt_targets.append(pos_samples + file)

            pt_inputs.sort()
            pt_targets.sort()

            ls_inputs += pt_inputs
            ls_targets += pt_targets

        neg_samples = path_to_files + 'negative_samples/'

        if os.path.isdir(neg_samples): #if there are negative samples for the given pt_id

            pt_inputs = []
            pt_targets = []
            files_negative = os.listdir(neg_samples)
            for file in files_negative:
                if file.startswith('input'):
                    pt_inputs.append(neg_samples + file)
                if file.startswith('target'):
                    pt_targets.append(neg_samples + file)
            pt_inputs.sort()
            pt_targets.sort()

            ls_inputs += pt_inputs
            ls_targets += pt_targets

    return ls_inputs, ls_targets

        
class DatasetFromNpySlices(torch.utils.data.Dataset):
    def __init__(self, list_of_input_paths, list_of_target_paths, out_path, disease, run_output_path):
        self.list_of_input_paths = list_of_input_paths
        self.list_of_target_paths = list_of_target_paths
        self.feature_means = np.load(run_output_path + 'training_means.npy')
        self.feature_stds = np.load(run_output_path + 'training_stds.npy')
        self.disease = disease
    
    def __len__(self):
        return len(self.list_of_input_paths) # since list_of_input_paths and list_of_target_paths have the same length
    
    def __getitem__(self, idx):
        input_path = self.list_of_input_paths[idx]
        target_path = self.list_of_target_paths[idx]
        # print(input_path, target_path)

        input_npy = np.load(input_path)

        # CIRC ONLY - REMOVE MBP
        if "circ" in self.disease:
            input_npy = np.delete(input_npy, 3, axis=1)

        # increase resolution of input
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
        
        # TODO: check if this is the right way to normalize the input
        input_npy = (input_npy - self.feature_means) / self.feature_stds
        # print("MEAN: ", np.mean(input_npy, axis=0))
        # print("VAR: ", np.var(input_npy, axis=0))
        # print(input_npy)

        target_npy = np.load(target_path)
        target_npy = np.repeat(target_npy, new_nb_timesteps // original_nb_timesteps, axis=0)
        #target_npy = (np.average(target_npy, keepdims=True) >= 0.5).astype(int)
        target_npy = target_npy.reshape(-1, 1)
        return input_npy, target_npy

def generator_fcn(pytorch_dataloader):

    for batch in pytorch_dataloader:
        decoder_input = np.zeros((batch[0].shape[0], 48, 1))
        yield ((batch[0].numpy().astype("float32"), decoder_input), batch[1].numpy().astype("float32"))
        # yield (batch[0].numpy().astype("float32"), batch[1].numpy().astype("float32"))


def convert_pytorch_dataset_to_tensorflow(ls_inputs, ls_targets, input_shape, target_shape, out_path, disease, shuffle_choice, run_output_path):
    dataset = DatasetFromNpySlices(ls_inputs, ls_targets, out_path, disease, run_output_path)
    pytorch_dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=shuffle_choice, num_workers=16, persistent_workers=True)

    tf_dataset = tf.data.Dataset.from_generator(
        lambda: generator_fcn(pytorch_dataloader),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, input_shape[0], input_shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, input_shape[0], 1), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, target_shape[0], 1), dtype=tf.float32),
        )
        # output_signature=(
        #         tf.TensorSpec(shape=(None, input_shape[0], input_shape[1]), dtype=tf.float32),
        #     tf.TensorSpec(shape=(None, target_shape[0], 1), dtype=tf.float32),
        # )
    )
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset