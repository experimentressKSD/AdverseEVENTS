import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
from ml_pipeline.models.models_whole_seqs import create_model
from ml_pipeline.whole_seqs_labeling import disease_labeling, raw_data_saving
from ml_pipeline.collate_dataset import pt_split_seqs, whole_seqs_data_loader
from ml_pipeline.dataset_statistics import collect_normalization_stats_seq, case_control_matching
from ml_pipeline.evaluation_whole_seqs import case_control_matching_test, evaluation
from common import read_config
import pandas as pd
import tensorflow as tf
import wandb
import torch
# from wandb.keras import WandbMetricsLogger
from keras.models import load_model
import time
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report


def get_labeling_vars(training_config_path):

    config = read_config(training_config_path)

    dataset = config.get('dataset')
    if dataset == 'mimic':
        data_paths = config.get('data_paths_mimic')
        labs_path = data_paths.get('labs_path')
        out_path = data_paths.get('out_path')
        vitals_path = data_paths.get('vitals_path')
        ids_file = pd.read_csv( labs_path + '0labels.txt')
        all_pt_ids = ids_file['id']

        labelling = config.get('labelling_mimic')
        generate_labels = labelling.get('generate_labels')
        labeled_disease = labelling.get('disease')
        before_onset_positive = labelling.get('before_onset_positive')
        after_onset_positive = labelling.get('after_onset_positive')
        end_pos_seq_after_onset = labelling.get('end_pos_seq_after_onset')
        adjust_neg_seq_length = labelling.get('adjust_neg_seq_length')

        model = config.get('model')


        return generate_labels, labeled_disease, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, vitals_path, all_pt_ids, out_path, labs_path, dataset, model, None
    
    elif dataset == 'eicu':
        data_paths = config.get('data_paths_eicu')
        labels_path = data_paths.get('labels_path')
        out_path = data_paths.get('out_path')
        vitals_path = data_paths.get('vitals_path')

        labelling = config.get('labelling_eicu')
        generate_labels = labelling.get('generate_labels')
        labeled_disease = labelling.get('disease')
        before_onset_positive = labelling.get('before_onset_positive')
        after_onset_positive = labelling.get('after_onset_positive')
        end_pos_seq_after_onset = labelling.get('end_pos_seq_after_onset')
        adjust_neg_seq_length = labelling.get('adjust_neg_seq_length')
        vitals_type = labelling.get('vitals_type')

        labels_file = pd.read_csv(labels_path + labeled_disease + '_labels.csv')
        all_pt_ids = labels_file[labels_file['vitals_available'] == vitals_type]['patient_ID']

        model = config.get('model')

        return generate_labels, labeled_disease, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, vitals_path, all_pt_ids, out_path, labels_file, dataset, model, vitals_type



class CustomValidationAccuracy(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            # Initialize lists to store the predictions and true labels
            preds = []
            ground_truths = []

            # Iterate over the dataset
            predictions_batch = self.model.predict(self.validation_data)
            targets = []
            for _, target, sample_weights in self.validation_data:
                # Append the target to the list
                targets.append(target.numpy())
            targets = np.concatenate(targets)

            for i in range(predictions_batch.shape[0]):
                model_out = predictions_batch[i]
                target = targets[i]
                last_index = np.where(target == 100000.0)[0]
                if last_index.size != 0:
                    last_index = last_index[0]
                    target = target[:last_index]
                    model_out = ((model_out[:last_index]) > 0.5).astype(int)

                else:
                    last_index = None
                
                assert np.sum(target == 100000.0) == 0

                if np.sum(target == 1.0) > 0: # positive patient

                    ground_truths.append(1)
                    if np.sum(model_out) > 0:
                        preds.append(1)
                    else:
                        preds.append(0)
                elif np.sum(target == 1.0) == 0: # negative patient
                    ground_truths.append(0)
                    if np.sum(model_out) > 0:
                        preds.append(1)
                    else:
                        preds.append(0)
            print(confusion_matrix(ground_truths, preds))
            print(classification_report(ground_truths, preds))

def get_data_shapes(dataset, labeled_disease, vitals_type):

    
    minutes_in_a_day = 24*60

    if dataset == 'mimic':
        max_len_stay_days = 7
        sampling_freq = 30
        nb_features = 7
        if labeled_disease == 'circulatory':
            nb_features = 6


    elif dataset == 'eicu':
        max_len_stay_days = 6
        sampling_freq = 5
        if vitals_type == 'central':
            nb_features = 3
        elif vitals_type == 'no_bp':
            nb_features = 4
        elif vitals_type == 'all':
            nb_features = 7

    max_seq_len = int(max_len_stay_days * minutes_in_a_day // sampling_freq)
    input_shape = (max_seq_len, nb_features)
    target_shape = (None, 1)

    print(f"Input shape: {input_shape}")
    print(f"Target shape: {target_shape}")

    return input_shape, target_shape, max_len_stay_days

def get_out_path(out_path, labeled_disease, dataset):

    if dataset == 'mimic':
        whole_seqs_out_path = out_path + labeled_disease + '/'
    elif dataset == 'eicu':
        whole_seqs_out_path = out_path + dataset + '/' + labeled_disease + '/'

    if not os.path.exists(whole_seqs_out_path):
        os.makedirs(whole_seqs_out_path)

    return whole_seqs_out_path

def get_chosen_weights(labeled_disease, dataset):

    if dataset == 'mimic':
        if labeled_disease == 'circulatory':
            chosen_zeros_weight = 0.3
        if labeled_disease == 'respiratory_HiRID':
            chosen_zeros_weight = 0.45
        if labeled_disease == 'kidney':
            chosen_zeros_weight = 0.45
    if dataset == 'eicu':
        if labeled_disease == 'circulatory':
            chosen_zeros_weight = 0.15
        if labeled_disease == 'respiratory_HiRID':
            chosen_zeros_weight = 0.15
        if labeled_disease == 'kidney':
            chosen_zeros_weight = 0.2
    
    return chosen_zeros_weight, 1 - chosen_zeros_weight



if __name__ == '__main__':
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # wandb.init(project="organ_system_failure_prediction_whole_seqs", entity="amelatur")

    # wandb_run_name = wandb.run.name

    training_config_path = '../training_test_config_whole_seqs.yaml'

    generate_labels, labeled_disease, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, vitals_path, all_pt_ids, out_path, labs_path, dataset, model_str, vitals_type = get_labeling_vars(training_config_path)

    input_shape, target_shape, max_len_stay_days = get_data_shapes(dataset, labeled_disease, vitals_type)

    whole_seqs_out_path = get_out_path(out_path, labeled_disease, dataset)

    if generate_labels and dataset == 'mimic':

        disease_labeling(labeled_disease, all_pt_ids, vitals_path, whole_seqs_out_path, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, max_len_stay_days)

    elif generate_labels and dataset == 'eicu':
        raw_data_saving(labeled_disease, all_pt_ids, vitals_path, whole_seqs_out_path, before_onset_positive, after_onset_positive, end_pos_seq_after_onset, adjust_neg_seq_length, vitals_type, labs_path, max_len_stay_days)

    # test, train split
    training_pts, test_pts, val_pts = pt_split_seqs(whole_seqs_out_path, all_pt_ids)

    # collect normalization stats, sequence lengths per patient class and patient ids per class
    collect_normalization_stats_seq(training_pts, whole_seqs_out_path, input_shape)

    # case control matching (sequence length)
    final_patients, ones_weight, zeros_weight = case_control_matching(whole_seqs_out_path, input_shape)

    # create model
    model = create_model(input_shape, model_str)

    # get sample weights
    chosen_zeros_weight, chosen_ones_weight = get_chosen_weights(labeled_disease, dataset)

    # training data loader
    batch_size = 32
    train_data_loader = whole_seqs_data_loader(batch_size, final_patients, whole_seqs_out_path, 'train', input_shape, chosen_ones_weight, chosen_zeros_weight, model_str)

    # load val patients
    val_patients = []
    with open(whole_seqs_out_path + 'validation_pt_ids.txt', 'r') as f:
        for line in f:
            val_patients.append(int(line.strip()))
 
    # validation data loader
    validation_patients = case_control_matching_test(whole_seqs_out_path, val_patients, input_shape)

    val_data_loader = whole_seqs_data_loader(batch_size, validation_patients, whole_seqs_out_path, 'test', input_shape, chosen_ones_weight, chosen_zeros_weight, model_str)

    # validation loss early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    # model training and saving
    model.fit(train_data_loader, epochs=1, callbacks=[WandbMetricsLogger(), CustomValidationAccuracy(val_data_loader)], batch_size=batch_size)

    model.save(whole_seqs_out_path + model_str + f'_weights_{time.time()}.keras')

    # load test patients
    test_patients = []
    with open(whole_seqs_out_path + 'test_pt_ids.txt', 'r') as f:
        for line in f:
            test_patients.append(int(line.strip()))

    # model test set inference
    test_final_patients = case_control_matching_test(whole_seqs_out_path, test_patients, input_shape)

    evaluation(whole_seqs_out_path, model, test_final_patients, batch_size, input_shape, chosen_ones_weight, chosen_zeros_weight, model_str)
 