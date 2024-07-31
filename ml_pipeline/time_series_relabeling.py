import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def create_tfrecord(input_dataset, target_dataset):
    feature_dict = {
        'input': tf.train.Feature(float_list=tf.train.FloatList(value=input_dataset)),
        'target': tf.train.Feature(float_list=tf.train.FloatList(value=target_dataset)),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def save_tfrecord_from_dataset(tfrecord_path, tfrecord):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        writer.write(tfrecord)

def save_npy_from_dataset(path, input_dataset, target_dataset):

    for idx, tensor in enumerate(input_dataset):
        npy_path = path + "input_data_" + str(idx)
        # create directory if it doesn't exist already
        if not(os.path.exists(path)):
            os.makedirs(path)
        np.save(npy_path, tensor)

    for idx, tensor in enumerate(target_dataset):
        npy_path = path + "target_data_" + str(idx)
        np.save(npy_path, tensor)


def get_target_array(exclusion_flag, status):
    # now return time of event
    threshold = 3
    result_index = status[status.groupby((status != status.shift()).cumsum()[status.eq(1)]).transform('count') > threshold]
    if result_index.empty == False:
        result_index = status.index.get_loc(result_index.idxmax())
    else:
        result_index = None

    n_steps = len(status.index)

    # now return an array that is full of 1s 8H before the event, and 2H after the event, and 0s otherwise (clip the array 2H after event)
    final_target_series = pd.Series(0, index=status.index)
    target_dataset_negative = None
    target_dataset_positive = None

    nb_timesteps_per_tensor = 4*2 # 2 steps per hour, 4H total --> 8 timesteps per tensor
    chosen_stride = 1

    if result_index != None:

        if result_index < 4*2:
            exclusion_flag = 1 # if onset of sepsis occurs within 4 hours of admission, exclude the patient

        start_event = max(0, result_index - 4*2)
        end_event = min(n_steps, result_index + 2*2)
        final_target_series.iloc[start_event:end_event] = 1

        final_target_series = final_target_series[:end_event]
        full_dataset = tf.keras.utils.timeseries_dataset_from_array(np.array(final_target_series), None, sequence_length = nb_timesteps_per_tensor, sequence_stride = chosen_stride, batch_size = None)

        negative_tensors = []
        positive_tensors = []

        for tensor in full_dataset:
            arr = tensor.numpy()
            if arr.any():
                positive_tensors.append(arr)
            else:
                negative_tensors.append(arr)

        if len(negative_tensors) > 0:
            target_dataset_negative = tf.data.Dataset.from_tensor_slices(negative_tensors)
        if len(positive_tensors) > 0:
            target_dataset_positive = tf.data.Dataset.from_tensor_slices(positive_tensors)

        return exclusion_flag, target_dataset_negative, target_dataset_positive, result_index, len(negative_tensors)

    else:
        negative_tensor = final_target_series
        if negative_tensor.size > 1:
            target_dataset_negative = tf.keras.utils.timeseries_dataset_from_array(np.array(negative_tensor), None, sequence_length = nb_timesteps_per_tensor, sequence_stride = chosen_stride, batch_size = None)
        return exclusion_flag, target_dataset_negative, target_dataset_positive, result_index, None


def get_input_array(patient_id, vitals_folder_path, exclusion_flag, result_index, len_neg_tensors):

    # now get the vitals as inputs
    vitals_file = pd.read_csv(vitals_folder_path + str(patient_id) + '_vitals.csv', index_col=False)
    vitals_file['time'] = pd.to_datetime(vitals_file['time'])
    vitals_file = vitals_file.set_index('time')
    vitals = vitals_file.drop(['Unnamed: 0', 'id'], axis=1).resample('30T').mean() 
    vitals = vitals.ffill().bfill()
    if vitals.isna().any().any():
        exclusion_flag = 1

    n_steps = len(vitals.index)
    input_dataset_negative = None
    input_dataset_positive = None

    nb_timesteps_per_tensor = 4*2 # 2 steps per hour, 4H total --> 8 timesteps per tensor
    chosen_stride = 1

    if result_index != None:     

        end_event = min(n_steps, result_index + 2*2)
        vitals = vitals[:end_event]   

        full_dataset = tf.keras.utils.timeseries_dataset_from_array(np.array(vitals), None, sequence_length = nb_timesteps_per_tensor, sequence_stride = chosen_stride, batch_size = None)

        negative_tensors = []
        positive_tensors = []

        for idx,tensor in enumerate(full_dataset):
            arr = tensor.numpy()
            if idx < len_neg_tensors:
                negative_tensors.append(arr)
            else:
                positive_tensors.append(arr)

        if len(negative_tensors) > 0:
            input_dataset_negative = tf.data.Dataset.from_tensor_slices(negative_tensors)
        if len(positive_tensors) > 0:
            input_dataset_positive = tf.data.Dataset.from_tensor_slices(positive_tensors)

    else:
        vitals_negative_tensor = vitals
        if vitals_negative_tensor.shape[0] > 1:
            input_dataset_negative = tf.keras.utils.timeseries_dataset_from_array(np.array(vitals_negative_tensor), None, sequence_length = nb_timesteps_per_tensor, sequence_stride = chosen_stride, batch_size = None)
        
    return input_dataset_negative, input_dataset_positive, exclusion_flag

def save_data_slices(input_dataset_negative, target_dataset_negative, input_dataset_positive, target_dataset_positive, exclusion_flag, path):

    if exclusion_flag != 1 and target_dataset_positive != None and input_dataset_positive != None:
        positive_path = path + '/positive_samples/' # the path is data_slices/<disease_name>/<patient_id>/positive_samples/

        num_target = tf.data.experimental.cardinality(target_dataset_positive).numpy()
        num_input = tf.data.experimental.cardinality(input_dataset_positive).numpy()

        if (num_target != num_input):
            print("Number of target and input tensors are not equal")

        save_npy_from_dataset(positive_path, input_dataset_positive, target_dataset_positive)

    if exclusion_flag != 1 and target_dataset_negative != None and input_dataset_negative != None:

        num_target = tf.data.experimental.cardinality(target_dataset_negative).numpy()
        num_input = tf.data.experimental.cardinality(input_dataset_negative).numpy()

        negative_path = path + '/negative_samples/'
        if (num_target != num_input):
            print("Number of target and input tensors are not equal")
        save_npy_from_dataset(negative_path, input_dataset_negative, target_dataset_negative)

