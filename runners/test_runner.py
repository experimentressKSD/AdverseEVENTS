import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import keras
from ml_pipeline.collate_dataset import get_test_window_paths, convert_pytorch_dataset_to_tensorflow
from common import read_config
from ml_pipeline.evaluation import iterate_patients, compute_results, evaluate_indiv
from ml_pipeline.models.time_series_models import create_model
import pandas as pd
from tcn import TCN
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    

def get_test_vars(test_config_path):
    config = read_config(test_config_path)
    test_config = config.get('test')
    data_path = config.get('data_paths')
    labs_path = data_path.get('labs_path')
    out_path = data_path.get('out_path')
    vitals_path = data_path.get('vitals_path')
    diseases = test_config.get('diseases')
    model = test_config.get('model')

    wandb_run_name = test_config.get('wandb_run_name')

    return diseases, out_path, model, vitals_path, wandb_run_name

def get_test_pt_ids(run_output_path):

    test_pt_ids = []
    path = run_output_path + 'test_pt_ids.txt'
    with open (path, 'r') as f:
        for line in f:
            test_pt_ids.append(line.strip())

    return test_pt_ids

def model_test(input_shape, target_shape, tf_dataset, out_path, disease, model, run_output_path):

    saved_model = keras.models.load_model(run_output_path + str(model) + '_weights.keras', custom_objects={'TCN': TCN})

    # inference on test data
    predictions = saved_model.evaluate(tf_dataset, batch_size=128)

    # save predictions, their order should be the same as the data window paths
    # np.save(out_path + disease + '/' + str(model) + '_predictions.npy', predictions)
    # print(predictions.shape)


if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    test_config_path = '../training_test_config.yaml'
    input_shape = (48, 6)
    target_shape = (48,1)

    # read relevant data to produce and store data windows
    diseases, out_path, model, vitals_folder_path, wandb_run_name = get_test_vars(test_config_path)

    for disease in diseases:

        run_output_path = out_path + disease + '/' + str(wandb_run_name) + '/'
        test_pt_ids = get_test_pt_ids(run_output_path)

        # collect paths of all saved data windows of all test patients
        #ls_inputs, ls_targets = get_test_window_paths(disease, out_path, test_pt_ids)

        #print("There are " + str(len(ls_inputs)) + " test windows and " + str(len(test_pt_ids)) + " test patients for the disease: " + disease)

        #tf_dataset = convert_pytorch_dataset_to_tensorflow(ls_inputs, ls_targets, input_shape, target_shape, out_path, disease, shuffle_choice=False, run_output_path=run_output_path) # don'&t shuffle test samples

        # inference
        #model_test(input_shape, target_shape, tf_dataset, out_path, disease, model, run_output_path)

        # # evaluation
        # iterate_patients(ls_inputs, ls_targets, disease, out_path, model)
        # compute_results(disease, out_path, model)

        print("There are " + str(len(set(test_pt_ids))) + " test patients for the failure type: " + ", within the run " + str(wandb_run_name))


        evaluate_indiv(test_pt_ids, vitals_folder_path, out_path, disease, model, run_output_path, target_shape)






