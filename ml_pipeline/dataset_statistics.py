import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from tqdm import tqdm
import tensorflow as tf

def collect_normalization_stats_seq(training_pts, whole_seqs_out_path, input_shape):

    means = []
    vars = []

    positive_patients = []
    negative_patients = []

    positive_seq_lengths = []
    negative_seq_lengths = []

    for pat_id in tqdm(training_pts):
        
        vitals_path = whole_seqs_out_path + str(pat_id) + '_raw_vitals.npy'
        target_path = whole_seqs_out_path + str(pat_id) + '_raw_target.npy'

        if os.path.exists(vitals_path): # if the patient wasn't excluded

            vitals = np.load(vitals_path)
            target = np.load(target_path)

            mean = np.mean(vitals, axis=0)
            var = np.var(vitals, axis=0)
            means.append(mean)
            vars.append(var)

            


            if np.any(target == 1.0): # positive patient
                positive_patients.append(pat_id)
                positive_seq_lengths.append(vitals.shape[0])

            elif np.any(target == 1.0) == False: # negative patient
                negative_patients.append(pat_id)
                negative_seq_lengths.append(vitals.shape[0])

    overall_mean = np.mean(np.array(means), axis=0)
    overall_std = np.sqrt(np.mean(np.array(vars), axis=0))   

    np.save(whole_seqs_out_path + 'training_means.npy', overall_mean)
    np.save(whole_seqs_out_path + 'training_stds.npy', overall_std)

    with open(whole_seqs_out_path + 'len_positive_seqs.txt', 'w') as f:
        for item in positive_seq_lengths:
            # Write each item on a new line
            f.write("%s\n" % item)

    with open(whole_seqs_out_path + 'len_negative_seqs.txt', 'w') as f:
        for item in negative_seq_lengths:
            # Write each item on a new line
            f.write("%s\n" % item)

    with open(whole_seqs_out_path + 'positive_patient_ids.txt', 'w') as f:
        for item in positive_patients:
            # Write each item on a new line
            f.write("%s\n" % item)

    with open(whole_seqs_out_path + 'negative_patient_ids.txt', 'w') as f:
        for item in negative_patients:
            # Write each item on a new line
            f.write("%s\n" % item)

    assert len(positive_patients) == len(positive_seq_lengths)
    assert len(negative_patients) == len(negative_seq_lengths)

    print("There are ", len(positive_patients), " positive patients in the training set")
    print("There are ", len(negative_patients), " negative patients in the training set")
    print("Before case-control matching, the mean sequence length for positive patients is: ", np.mean(positive_seq_lengths))
    print("Before case-control matching, the mean sequence length for negative patients is: ", np.mean(negative_seq_lengths))
    print("Before case-control matching, the median sequence length for positive patients is: ", np.median(positive_seq_lengths))
    print("Before case-control matching, the median sequence length for negative patients is: ", np.median(negative_seq_lengths))

    print("Before case-control matching, max neg seq length: ", np.max(positive_seq_lengths))
    print("Before case-control matching, max neg seq length: ", np.max(negative_seq_lengths))

    print("Finished training set normalization statistics")


def case_control_matching(whole_seqs_out_path, input_shape):

    # Load patient ids
    positive_patients = []
    negative_patients = []

    with open(whole_seqs_out_path + 'positive_patient_ids.txt', 'r') as f:
        for line in f:
            positive_patients.append(int(line.strip()))

    with open(whole_seqs_out_path + 'negative_patient_ids.txt', 'r') as f:
        for line in f:
            negative_patients.append(int(line.strip()))

    # Load sequence lengths
    positive_seq_lengths = []
    negative_seq_lengths = []

    with open(whole_seqs_out_path + 'len_positive_seqs.txt', 'r') as f:
        for line in f:
            positive_seq_lengths.append(int(line.strip()))

    with open(whole_seqs_out_path + 'len_negative_seqs.txt', 'r') as f:
        for line in f:
            negative_seq_lengths.append(int(line.strip()))

    # compute how many positive patients need to matched to negative patients - aka class imbalance in training set
    nb_pos = len(positive_patients)
    nb_neg = len(negative_patients)
    ratio = int(nb_neg / nb_pos)
    if ratio == 0:
        ratio = 1

    neg_patient_counter = 0 # iterate through negative patients, adjusting their sequence length to match positive patients'

    final_patients = []
    pos_pts_final  = []
    neg_pts_final = []

    pos_seq_final_lengths = []
    neg_seq_final_lengths = []

    max_seq_length = input_shape[0]
    padding_value = 100000.0

    # get normalization stats
    overall_mean = np.load(whole_seqs_out_path + 'training_means.npy')
    overall_std = np.load(whole_seqs_out_path + 'training_stds.npy')

    total_length = 0
    total_ones = 0


    for pos_seq_len, pos_pt in zip(positive_seq_lengths, positive_patients):
        if neg_patient_counter >= len(negative_patients):
            break
        for i in range(ratio):
            neg_pt = negative_patients[neg_patient_counter]
            pt_target = np.load(whole_seqs_out_path + str(neg_pt) + '_raw_target.npy')
            pt_vitals = np.load(whole_seqs_out_path + str(neg_pt) + '_raw_vitals.npy')

            len_seq_neg = pt_vitals.shape[0]

            if pos_seq_len < len_seq_neg:
            
                pt_target = pt_target[:pos_seq_len]
                pt_vitals = pt_vitals[:pos_seq_len]

            pt_vitals = (pt_vitals - overall_mean) / overall_std

            total_length += pt_target.shape[0]

            pt_target = tf.keras.preprocessing.sequence.pad_sequences([pt_target],maxlen=max_seq_length, padding='post', truncating='post', value=padding_value, dtype='float32')
            
            pt_vitals = tf.keras.preprocessing.sequence.pad_sequences([pt_vitals],maxlen=max_seq_length, padding='post', truncating='post', value=padding_value, dtype='float32')

            pt_target = np.squeeze(pt_target, axis=0)
            pt_vitals = np.squeeze(pt_vitals, axis=0)

            pt_target = np.expand_dims(pt_target, axis=-1)

            np.save(whole_seqs_out_path + str(neg_pt) + '_target.npy', pt_target)
            np.save(whole_seqs_out_path + str(neg_pt) + '_vitals.npy', pt_vitals)

            final_patients.append(neg_pt)
            neg_pts_final.append(neg_pt)
            neg_seq_final_lengths.append(pos_seq_len)

            neg_patient_counter += 1

        pt_target = np.load(whole_seqs_out_path + str(pos_pt) + '_raw_target.npy')
        pt_vitals = np.load(whole_seqs_out_path + str(pos_pt) + '_raw_vitals.npy')

        # pt_target = pt_target[:pos_seq_len]
        # pt_vitals = pt_vitals[:pos_seq_len]

        nb_ones = np.sum(pt_target)
        total_ones += nb_ones
        total_length += pt_target.shape[0]

        pt_vitals = (pt_vitals - overall_mean) / overall_std

        pt_target = tf.keras.preprocessing.sequence.pad_sequences([pt_target],maxlen=max_seq_length, padding='post', truncating='post', value=padding_value, dtype='float32')
        
        pt_vitals = tf.keras.preprocessing.sequence.pad_sequences([pt_vitals],maxlen=max_seq_length, padding='post', truncating='post', value=padding_value, dtype='float32')

        pt_target = np.squeeze(pt_target, axis=0)
        pt_vitals = np.squeeze(pt_vitals, axis=0)

        pt_target = np.expand_dims(pt_target, axis=-1)

        np.save(whole_seqs_out_path + str(pos_pt) + '_target.npy', pt_target)
        np.save(whole_seqs_out_path + str(pos_pt) + '_vitals.npy', pt_vitals)

        final_patients.append(pos_pt)
        pos_pts_final.append(pos_pt)
        pos_seq_final_lengths.append(pos_seq_len)



    with open(whole_seqs_out_path + 'final_patient_ids.txt', 'w') as f:
        for item in final_patients:
            # Write each item on a new line
            f.write("%s\n" % item)

    print('Mean positive patients sequence length: ', np.mean(pos_seq_final_lengths))
    print('Mean negative patients sequence length: ', np.mean(neg_seq_final_lengths))
    print("Median positive patients sequence length: ", np.median(pos_seq_final_lengths))
    print("Median negative patients sequence length: ", np.median(neg_seq_final_lengths))

    print("Max neg seq length: ", np.max(pos_seq_final_lengths))
    print("Max neg seq length: ", np.max(neg_seq_final_lengths))

    print('Finished case-control matching. There are ', len(pos_pts_final), ' positive patients in the training set and ', len(neg_pts_final), ' negative patients in the training set')


    zeros_weight = round(total_ones/total_length, 2)

    ones_weight = 1 - zeros_weight

    print("There are ", total_ones, " ones in the training set for a total of ", total_length, " targets in the training set")
    print("Zeros weight: ", zeros_weight)
    print("Ones weight: ", ones_weight)

    return final_patients, ones_weight, zeros_weight


def collect_normalization_stats(paths_to_each_window, out_path, disease, run_output_path):

    means = []
    vars = [] # collect variance instead of stds directly 

    for path in paths_to_each_window:

        input_npy = np.load(path)

        # CIRC ONLY - REMOVE MBP
        if "circ" in disease:
            input_npy = np.delete(input_npy, 3, axis=1)

        # increase resolution of input
        original_nb_timesteps = input_npy.shape[0] # 4 hr window with 30 min timesteps --> 4 *2 = 8
        new_nb_timesteps = 48 # 4 hr window with 5 min timesteps --> 4 * 12 = 48

        old_time = np.linspace(0, 1, original_nb_timesteps)
        new_time = np.linspace(0, 1, new_nb_timesteps)
        data = np.zeros((new_nb_timesteps, input_npy.shape[1]))

        # Interpolate each feature
        for i in range(input_npy.shape[1]):
            interpolator = interp1d(old_time, input_npy[:, i], kind='linear')  # Change to 'cubic' for spline
            data[:, i] = interpolator(new_time)

        means.append(np.mean(data, axis=0))
        vars.append(np.var(data, axis=0))    

    print(np.array(vars).shape)

    overall_mean = np.mean(np.array(means), axis=0)
    overall_std = np.sqrt(np.mean(np.array(vars), axis=0))
    np.save(run_output_path + 'training_means.npy', overall_mean)
    np.save(run_output_path + 'training_stds.npy', overall_std)

    print("Means: ", overall_mean)
    print("Standard deviations: ", overall_std)



 

        


