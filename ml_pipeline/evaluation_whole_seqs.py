import numpy as np
import pandas as pd
import os
from ml_pipeline.collate_dataset import whole_seqs_data_loader
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def mark_sequences(binary_array, k):
    count = 0
    start_index = None
    for i in range(len(binary_array)):
        if binary_array[i] == 1:
            if count == 0:
                start_index = i
            count += 1
            if count > k:
                binary_array[start_index+1:i+1] = 0
        else:
            if count <= k and start_index is not None:
                binary_array[start_index:i] = 0
            count = 0
            start_index = None
    # Handle the last sequence of 1s
    if count <= k and start_index is not None:
        binary_array[start_index:i+1] = 0
    return binary_array



def case_control_matching_test(whole_seqs_out_path, patient_ids, input_shape):

    positive_patients = []
    negative_patients = []

    pos_seq_lengths = []
    neg_seq_lengths = []

    for pat_id in patient_ids:

        vitals_path = whole_seqs_out_path + str(pat_id) + '_raw_vitals.npy'
        target_path = whole_seqs_out_path + str(pat_id) + '_raw_target.npy'

        if os.path.exists(vitals_path): # if the patient wasn't excluded

            vitals = np.load(vitals_path)
            target = np.load(target_path)

            if np.any(target == 1.0) and target.shape[0] < input_shape[0]: # positive patient
                positive_patients.append(pat_id)
                pos_seq_lengths.append(vitals.shape[0])
            elif np.any(target == 1.0) == False and target.shape[0] < input_shape[0]:
                negative_patients.append(pat_id)
                neg_seq_lengths.append(vitals.shape[0])

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

    for pos_seq_len, pos_pt in zip(pos_seq_lengths, positive_patients):
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



    with open(whole_seqs_out_path + 'test_final_patients.txt', 'w') as f:
        for item in final_patients:
            # Write each item on a new line
            f.write("%s\n" % item)

    print('Mean positive patients sequence length: ', np.mean(pos_seq_final_lengths))
    print('Mean negative patients sequence length: ', np.mean(neg_seq_final_lengths))
    print("Median positive patients sequence length: ", np.median(pos_seq_final_lengths))
    print("Median negative patients sequence length: ", np.median(neg_seq_final_lengths))

    print('Finished case-control matching for evaluation. There are ', len(pos_pts_final), ' positive patients in the test set and ', len(neg_pts_final), ' negative patients in the training set')

    return final_patients


def evaluation(whole_seqs_out_path, model, final_patients, batch_size, input_shape, ones_weight, zeros_weight, model_str):

    # data loader
    test_data_loader = whole_seqs_data_loader(batch_size, final_patients, whole_seqs_out_path, 'test', input_shape, ones_weight, zeros_weight, model_str)

    # evaluate
    predictions = model.predict(test_data_loader)
    print(predictions.shape)

    # save raw predictions
    np.save(whole_seqs_out_path + 'raw_preds_' + model_str + '_6_24', predictions)
    model.evaluate(test_data_loader)

    np.save(whole_seqs_out_path + 'final_patients_test_' + model_str + '.npy', np.array(final_patients))

    targets = []
    if model_str == 'LSTM_enc_dec':
        for _, target in test_data_loader:
            targets.append(target.numpy())
    elif model_str == 'transformer' or model_str == 'TCN' or model_str == 'LSTM':
        for _, target, _ in test_data_loader:
            targets.append(target.numpy())

    # Convert the list of targets to a numpy array
    targets = np.concatenate(targets)

    # save targets
    np.save(whole_seqs_out_path + 'targets_test_' + model_str + '_6_24.npy', targets)





