import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
from ml_pipeline.models.time_series_models import create_model, get_compiled_model
from ml_pipeline.dense_labeling_mimic import disease_labeling
from ml_pipeline.collate_dataset import pt_split_random, get_training_window_paths, convert_pytorch_dataset_to_tensorflow
from ml_pipeline.dataset_statistics import collect_normalization_stats
from common import read_config
import pandas as pd
import keras
from tcn import TCN
import tensorflow as tf
import wandb
import torch
from wandb.keras import WandbMetricsLogger

    
def get_training_vars(training_config_path):
    config = read_config(training_config_path)
    training_config = config.get('training')
    data_path = config.get('data_paths')
    labs_path = data_path.get('labs_path')
    out_path = data_path.get('out_path')
    vitals_path = data_path.get('vitals_path')
    ids_file = pd.read_csv( labs_path + '0labels.txt')
    all_pt_ids = ids_file['id']
    diseases = training_config.get('diseases')
    model = training_config.get('model')
    whole_sequence_training = training_config.get('whole_sequence_training')

    labelling = config.get('labelling')

    return vitals_path, all_pt_ids, diseases, out_path, model, labelling, whole_sequence_training


def model_fit(input_shape, target_shape, tf_dataset, out_path, disease, model, run_output_path):

    model_path = run_output_path + str(model) + '_weights.keras'
    # if os.path.exists(model_path):
    #     saved_model = keras.models.load_model(model_path, custom_objects={'TCN': TCN})

    lr = 0.001
    step_lr_epoch_div = 101
    step_lr_div_factor = 0.75
    nb_epochs = 100
    nb_batches = len(list(tf_dataset))
    save_freq_checkpoint = 20 # save every 20 epochs

    # else:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=run_output_path + str(model) + '_model_checkpoint.weights.h5',
        save_freq=nb_batches * save_freq_checkpoint,
        save_weights_only=True,
        verbose=1  # Verbosity level
    )


    if True:
        model_created = create_model(input_shape, target_shape, model)
        model_compiled, lr_scheduler = get_compiled_model(model_created, lr, step_lr_epoch_div, step_lr_div_factor)
        saved_model = model_compiled

    # Log hyperparameters

    saved_model.fit(tf_dataset, epochs=nb_epochs, callbacks=[WandbMetricsLogger('epoch'), lr_scheduler, checkpoint_callback])
    saved_model.save(run_output_path + str(model) + '_weights.keras')



if __name__ == '__main__':
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    wandb.init(project="organ_system_failure_prediction", entity="amelatur")

    wandb_run_name = wandb.run.name

    training_config_path = '../training_test_config.yaml'


    # read relevant data to produce and store data windows
    vitals_path, all_pt_ids, diseases, out_path, model, labelling, whole_sequence_training = get_training_vars(training_config_path)

    for disease in diseases:
        nb_features = 7
        if disease == 'circulatory':
            nb_features = 6
        input_shape = (48, nb_features)
        target_shape = (48, 1)

        if labelling:
            disease_labeling(disease, all_pt_ids, vitals_path, out_path, whole_sequence_training)

        # output folder path for particular run
        run_output_path = out_path + disease + '/' + str(wandb_run_name) + '/'
        if not os.path.exists(run_output_path):
            os.makedirs(run_output_path)

        # select patients used for training
        training_pt_ids, test_pt_ids = pt_split_random(disease, out_path, run_output_path)
        ls_inputs, ls_targets = get_training_window_paths(disease, training_pt_ids, out_path) # select data windows used for training

        print("Generated " + str(len(ls_inputs)) + " windows used for training for disease " + disease + " failure, on " +str(len(training_pt_ids)) + " training patients.")

        # collect normalization statistics across all training windows for each input feature
        collect_normalization_stats(ls_inputs, out_path, disease, run_output_path)

        # generate dataset and normalize samples as they are served
        tf_dataset = convert_pytorch_dataset_to_tensorflow(ls_inputs, ls_targets, input_shape, target_shape, out_path, disease, shuffle_choice=True, run_output_path=run_output_path)

        # fit model
        model_fit(input_shape, target_shape, tf_dataset, out_path, disease, model, run_output_path)
        