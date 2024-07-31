from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial
import argparse

# supppress warnings
import warnings
warnings.filterwarnings("ignore")

def process_data(i, root_path, preds_path, dataset, disease, targets_path, k):
    data_path = root_path + str(i) + '/'

    predictions = np.load(data_path + preds_path)
    targets = np.load(data_path + targets_path)
    if dataset == 'eicu':
        labels_file = pd.read_csv('/datasets/amelatur/eicu_labels/'+ disease + '_labels.csv')
        n_hours = 12
    if dataset == 'mimic':
        labels_file = pd.read_csv(data_path + 'onset_index.csv', header=None, index_col=False).rename({0: 'patient_ID', 1: 'indices_to_onset'}, axis=1)
        n_hours = 2
    final_patients = np.load(data_path + 'final_patients_test.npy')


    auroc, auprc, precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, fpr_ls, recalls, precisions, total_patients = eval(predictions, targets, labels_file, final_patients, n_hours, k)

    return auroc, auprc, precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, fpr_ls, recalls, precisions, total_patients


def mark_sequences(binary_array, k):
    count = 0
    start_index = None
    for i in range(len(binary_array)):
        if binary_array[i] == 1:
            if count == 0:
                start_index = i
            count += 1
        else:
            if count > 0:
                if count > k:
                    binary_array[start_index:start_index+k-1] = 0  # delete all 1s before the kth one
                    binary_array[start_index+k:i] = 0  # delete all 1s after the kth one
                else:
                    binary_array[start_index:i] = 0  # delete all 1s in the sequence
            count = 0
            start_index = None
    # Handle the last sequence of 1s
    if count > 0:
        if count > k:
            binary_array[start_index:start_index+k-1] = 0  # delete all 1s before the kth one
            binary_array[start_index+k:] = 0  # delete all 1s after the kth one
        else:
            binary_array[start_index:] = 0
    return binary_array

def count_predictions(predictions, targets, threshold, labels_file, final_patients, n_hours, alarm_earliness, k):

    out = []
    ground_truth = []
    times_to_onset = []
    cases_caught = 0
    total_cases = 0
    total_patients = predictions.shape[0]

    for i in range(predictions.shape[0]):

        target = targets[i]
        predicted = predictions[i]

        predicted_thresholded = (predicted > threshold).astype(int)

        last_index = np.where(target == 100000.0)[0]
        if last_index.size != 0:
            last_index = last_index[0]
            target = target[:last_index]
        else:
            last_index = None

        assert np.sum(target == 100000.0) == 0

        if k != None:
            predicted_thresholded = mark_sequences(predicted_thresholded, k)
        
        if np.sum(target == 1.0) > 0: # positive patient
            total_cases += 1

            if last_index != None:
                predicted_thresholded = predicted_thresholded[:last_index]

            ground_truth.append(1)
            if np.sum(predicted_thresholded) > 0: # TRUE POSITIVE
                
                out.append(1)

                if alarm_earliness:
                    onset_index = int((labels_file[labels_file['patient_ID'] == final_patients[i]]['indices_to_onset'].values[0]))
                    earliest_alarm = np.argmax(predicted_thresholded == 1)
                    time_to_onset = onset_index - earliest_alarm
                    times_to_onset.append(time_to_onset/n_hours)
                    cases_caught += 1
            else:
                out.append(0)
        elif np.sum(target == 1.0) == 0: # negative patient
            ground_truth.append(0)
            if last_index != None:
                predicted_thresholded = predicted_thresholded[:last_index]

            if np.sum(predicted_thresholded) > 0:
                out.append(1)
            else:
                out.append(0)


    tn, fp, fn, tp = confusion_matrix(ground_truth, out).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    median_alarm_earliness = np.median(np.array(times_to_onset))

    return precision, recall, specificity, median_alarm_earliness, cases_caught, total_cases, total_patients

def eval(predictions, targets, labels_file, final_patients, n_hours, k):

    thresholds = np.arange(0, 1, 0.01).tolist()
    precisions = []
    recalls = []
    specificities = []
    fpr_ls = []


    for threshold in thresholds:

        alarm_earliness = False
        precision, recall, specificity, median_alarm_earliness, cases_caught, total_cases, total_patients = count_predictions(predictions, targets, threshold, labels_file, final_patients, n_hours, alarm_earliness, k)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        fpr_ls.append(1 - specificity)

    # compute alarm earliness at 0.5 threshold
    threshold = 0.5
    alarm_earliness = True
    precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, total_patients = count_predictions(predictions, targets, threshold, labels_file, final_patients, n_hours, alarm_earliness, k)
    # auroc
    auroc = auc(fpr_ls, recalls)


    # Use a list comprehension to filter out pairs where either value is NaN
    filtered_pairs = [(x, y) for x, y in zip(precisions, recalls) if not (np.isnan(x) or np.isnan(y))]

    # Split the filtered pairs back into two lists
    precision_no_nan, recalls_no_nan = zip(*filtered_pairs)

    # auprc
    auprc = auc(recalls_no_nan, precision_no_nan)

    return auroc, auprc, precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, fpr_ls, recalls, precisions, total_patients

def alarm_earliness(predictions, targets, labels_file, final_patients, precisions, recalls, specificities):

    #recalls_target = [0.9, 0.8, 0.7, 0.6, 0.5]
    recalls_target = [0.8]
    precisions_target = []
    median_onset = []
    recalls_final = []

    target_recall = 0.8

    closest_index = min(range(len(recalls)), key=lambda index: abs(recalls[index]-target_recall))
    thresholds = np.arange(0, 1, 0.01).tolist()
    target_threshold = thresholds[closest_index] 
    target_recall = recalls[closest_index]

    out = []
    ground_truth = []
    times_to_onset = []
    for i in range(len(final_patients)):

        target = targets[i]
        predicted = predictions[i]

        predicted_thresholded = (predicted > target_threshold).astype(int)

        last_index = np.where(target == 100000.0)[0]
        if last_index.size != 0:
            last_index = last_index[0]
            target = target[:last_index]
        else:
            last_index = None

        assert np.sum(target == 100000.0) == 0
        
        if np.sum(target == 1.0) > 0: # positive patient

            if last_index != None:
                predicted_thresholded = predicted_thresholded[:last_index]

            onset_index = int((labels_file[labels_file['patient_ID'] == final_patients[i]]['indices_to_onset'].values[0]))

            ground_truth.append(1)
            if np.sum(predicted_thresholded) > 0:
                out.append(1)
                earliest_alarm = np.argmax(predicted_thresholded == 1)
                time_to_onset = onset_index - earliest_alarm
                times_to_onset.append(time_to_onset/12)
            else:
                out.append(0)
        elif np.sum(target == 1.0) == 0: # negative patient
            ground_truth.append(0)
            if last_index != None:
                predicted_thresholded = predicted_thresholded[:last_index]

            if np.sum(predicted_thresholded) > 0:
                out.append(1)
            else:
                out.append(0)

        median_onset.append(times_to_onset)
    
    median_to_plot = np.median(np.array(median_onset))

    return median_to_plot, recalls[closest_index], precisions[closest_index], target_threshold, specificities[closest_index]


def get_stats(disease, dataset, model):

    print("Disease: " + disease)
    print("Dataset: " + dataset)
    print("Model: " + model)


    root_path = '/datasets/amelatur/whole_sequences/'
    if dataset == 'mimic':
        root_path = root_path + disease + '/k_fold/fold_'
    elif dataset == 'eicu':
        root_path = root_path + 'eicu/eicu/'+ disease + '/k_fold/fold_'
    else:
        print("Invalid dataset")
        exit()

    if model == 'lstm':
        preds_path = 'raw_preds_online_lstm_6_24.npy'
        targets_path = 'targets_test_6_24.npy'
    elif model == 'enc_dec':
        preds_path = 'raw_preds_enc_dec_6_24.npy'
        targets_path = 'targets_test_transformer_6_24.npy'
    elif model == 'transformer':
        preds_path = 'raw_preds_transformer_6_24.npy'
        targets_path = 'targets_test_transformer_6_24.npy'
    else:
        print("Invalid model")
        exit()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        func = partial(process_data, root_path=root_path, preds_path=preds_path, dataset=dataset, disease=disease, targets_path=targets_path, k=None)
        results = list(executor.map(func, range(5)))

    auroc, auprc, precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, fpr_ls, recalls, precisions, total_patients = zip(*results)


    print("AUROC: " + str(np.mean(auroc)) + " +/- " + str(np.std(auroc)))
    print("AUPRC: " + str(np.mean(auprc)) + " +/- " + str(np.std(auprc)))
    print("Precision at 0.5 threshold: " + str(np.mean(precision_at_50)) + " +/- " + str(np.std(precision_at_50)))
    print("Recall at 0.5 threshold: " + str(np.mean(recall_at_50)) + " +/- " + str(np.std(recall_at_50)))
    print("Specificity at 0.5 threshold: " + str(np.mean(specificity_at_50)) + " +/- " + str(np.std(specificity_at_50)))
    print("Median alarm earliness at 0.5 threshold: " + str(np.mean(median_alarm_earliness_at_50)) + " +/- " + str(np.std(median_alarm_earliness_at_50)) + "for an average of " + str(np.mean(cases_caught)) + " cases caught out of an average of " + str(np.mean(total_cases)) + " total cases")

    mean_auroc = np.mean(auroc)
    mean_auprc = np.mean(auprc)
    mean_precision = np.mean(precision_at_50)
    mean_recall = np.mean(recall_at_50)
    mean_specificity = np.mean(specificity_at_50)
    mean_median_alarm_earliness = np.mean(median_alarm_earliness_at_50)
    mean_caught_cases = np.mean(cases_caught)
    mean_total_cases = np.mean(total_cases)
    mean_total_patients = np.mean(total_patients)

    std_auroc = np.std(auroc)
    std_auprc = np.std(auprc)
    std_precision = np.std(precision_at_50)
    std_recall = np.std(recall_at_50)
    std_specificity = np.std(specificity_at_50)
    std_median_alarm_earliness = np.std(median_alarm_earliness_at_50)
    std_caught_cases = np.std(cases_caught)
    std_total_cases = np.std(total_cases)
    std_total_patients = np.std(total_patients)

    # compute average of the list of lists
    fpr_ls = np.mean(fpr_ls, axis=0)
    recalls = np.mean(recalls, axis=0)
    precisions = np.mean(precisions, axis=0)

    return mean_auroc, mean_auprc, mean_precision, mean_recall, mean_specificity, mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, std_auroc, std_auprc, std_precision, std_recall, std_specificity, std_median_alarm_earliness, std_caught_cases, std_total_cases, fpr_ls, recalls, precisions, mean_total_patients, std_total_patients

def get_k_agg_stats(disease, dataset, model, k):

    print("Disease: " + disease)
    print("Dataset: " + dataset)
    print("Model: " + model)


    root_path = '/datasets/amelatur/whole_sequences/'
    if dataset == 'mimic':
        root_path = root_path + disease + '/k_fold/fold_'
    elif dataset == 'eicu':
        root_path = root_path + 'eicu/eicu/'+ disease + '/k_fold/fold_'
    else:
        print("Invalid dataset")
        exit()

    if model == 'lstm':
        preds_path = 'raw_preds_online_lstm_6_24.npy'
        targets_path = 'targets_test_6_24.npy'
    elif model == 'transformer':
        preds_path = 'raw_preds_transformer_6_24.npy'
        targets_path = 'targets_test_transformer_6_24.npy'
    else:
        print("Invalid model")
        exit()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        func = partial(process_data, root_path=root_path, preds_path=preds_path, dataset=dataset, disease=disease, targets_path=targets_path, k=k)
        results = list(executor.map(func, range(5)))

    auroc, auprc, precision_at_50, recall_at_50, specificity_at_50, median_alarm_earliness_at_50, cases_caught, total_cases, fpr_ls, recalls, precisions, total_patients = zip(*results)

    mean_recall = np.mean(recall_at_50)
    mean_specificity = np.mean(specificity_at_50)
    std_recall = np.std(recall_at_50)
    std_specificity = np.std(specificity_at_50)

    return mean_recall, mean_specificity, std_recall, std_specificity

def runner_plotter(disease, dataset):

    values_to_plot = ['Precision', 'Recall', 'Specificity']
    earliness_values = ['Median alarm earliness at 0.5 thr', 'Caught cases', 'Total cases']

    for model in ['lstm', 'transformer']:
        mean_auroc, mean_auprc, mean_precision, mean_recall, mean_specificity, mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, std_auroc, std_auprc, std_precision, std_recall, std_specificity, std_median_alarm_earliness, std_caught_cases, std_total_cases, fpr_ls, recalls, precisions, mean_total_patients, std_total_patients = get_stats(disease, dataset, model)

        if model == 'lstm':

            lstm_values = [mean_precision, mean_recall, mean_specificity]
            lstm_std = [std_precision, std_recall, std_specificity]
            lstm_earliness = [mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, mean_total_patients]
            lstm_earliness_std = [std_median_alarm_earliness, std_caught_cases, std_total_cases, std_total_patients]

            lstm_fpr = fpr_ls
            lstm_recalls = recalls
            lstm_precisions = precisions
            lstm_auroc = mean_auroc
            lstm_auprc = mean_auprc
            lstm_auroc_std = std_auroc
            lstm_auprc_std = std_auprc

        elif model == 'transformer':
                
            transformer_values = [mean_precision, mean_recall, mean_specificity]
            transformer_std = [std_precision, std_recall, std_specificity]
            transformer_earliness = [mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, mean_total_patients]
            transformer_earliness_std = [std_median_alarm_earliness, std_caught_cases, std_total_cases, std_total_patients]

            transformer_fpr = fpr_ls
            transformer_recalls = recalls
            transformer_precisions = precisions
            transformer_auroc = mean_auroc
            transformer_auprc = mean_auprc
            transformer_auroc_std = std_auroc
            transformer_auprc_std = std_auprc


    font_size = 15
    # plot ROC curve
    plt.plot(lstm_fpr, lstm_recalls, label='LSTM AUROC = %0.2f +/- %0.2f' % (lstm_auroc, lstm_auroc_std))
    plt.plot(transformer_fpr, transformer_recalls, label='Transformer AUROC = %0.2f +/- %0.2f' % (transformer_auroc, transformer_auroc_std))
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)

    # plot the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Model')
    plt.legend(fontsize='large')
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.savefig(str(disease) + '_' + str(dataset) + '_roc_curve.svg', format='svg')

    # plot PRC curve
    plt.figure()
    plt.plot(lstm_recalls, lstm_precisions, label='LSTM AUPRC = %0.2f +/- %0.2f' % (lstm_auprc, lstm_auprc_std))
    plt.plot(transformer_recalls, transformer_precisions, label='Transformer AUPRC = %0.2f +/- %0.2f' % (transformer_auprc, transformer_auprc_std)) 
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    # plot baseline straight line
    baseline = lstm_earliness[2] / lstm_earliness[3]
    plt.plot([0, 1], [baseline, baseline], color='navy', linestyle='--', label='Random Model')
    plt.ylim([0, 1])

    plt.legend(fontsize='large')
    plt.savefig(str(disease) + '_' + str(dataset) + '_prc_curve.svg', format='svg')

    # plot precision, recall, specificity
    plt.figure()
    x = np.arange(len(values_to_plot))
    width = 0.4 

    plt.bar(x - width/2, lstm_values, width=width, yerr=lstm_std, capsize=5, label='LSTM')
    plt.bar(x + width/2, transformer_values, width=width, yerr=transformer_std, capsize=5, label='Transformer')
    plt.axhline(y=0.5, color='red', linestyle='--') 
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.xticks(x, values_to_plot)
    plt.ylabel('Value at thr = 0.5', fontsize=font_size)
    plt.legend(fontsize='large')
    plt.savefig(str(disease) + '_' + str(dataset) + '_values.svg', format='svg')

    # plot earliness values
    plt.figure()
    # Define the x and y values and their corresponding errors
    x_value = [lstm_earliness[0], transformer_earliness[0]]
    y_value = [lstm_earliness[1], transformer_earliness[1]]
    x_error = [lstm_earliness_std[0], transformer_earliness_std[0]]
    y_error = [lstm_earliness_std[1], transformer_earliness_std[1]]

    print(transformer_earliness[0], transformer_earliness[1], transformer_earliness[2], transformer_earliness[3])

    # Create a dot plot with error bars
    plt.errorbar(x_value[0], y_value[0], xerr=x_error[0], yerr=y_error[0], fmt='o', color='blue', label='LSTM', capsize=5, elinewidth=2, alpha=0.5)
    plt.errorbar(x_value[1], y_value[1], xerr=x_error[1], yerr=y_error[1], fmt='o', color='red', label='Transformer', capsize=5, elinewidth=2, alpha=0.5)
    plt.xlabel('Median alarm earliness (hours) at 0.5 threshold', fontsize=font_size)
    plt.ylabel('Number of captured cases', fontsize=font_size)
    # add spacing for y label
    plt.tight_layout(pad=3.0)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.legend(fontsize='large')
    plt.savefig(str(disease) + '_' + str(dataset) + '_earliness.svg', format='svg')

def runner_plotter_enc_dec(disease, dataset):

    values_to_plot = ['Precision', 'Recall', 'Specificity']
    earliness_values = ['Median alarm earliness at 0.5 thr', 'Caught cases', 'Total cases']

    model = 'enc_dec'
    mean_auroc, mean_auprc, mean_precision, mean_recall, mean_specificity, mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, std_auroc, std_auprc, std_precision, std_recall, std_specificity, std_median_alarm_earliness, std_caught_cases, std_total_cases, fpr_ls, recalls, precisions, mean_total_patients, std_total_patients = get_stats(disease, dataset, model)


    lstm_values = [mean_precision, mean_recall, mean_specificity]
    lstm_std = [std_precision, std_recall, std_specificity]
    lstm_earliness = [mean_median_alarm_earliness, mean_caught_cases, mean_total_cases, mean_total_patients]
    lstm_earliness_std = [std_median_alarm_earliness, std_caught_cases, std_total_cases, std_total_patients]

    lstm_fpr = fpr_ls
    lstm_recalls = recalls
    lstm_precisions = precisions
    lstm_auroc = mean_auroc
    lstm_auprc = mean_auprc
    lstm_auroc_std = std_auroc
    lstm_auprc_std = std_auprc


    font_size = 15
    # plot ROC curve
    plt.plot(lstm_fpr, lstm_recalls, label='Seq2Seq LSTM AUROC = %0.2f +/- %0.2f' % (lstm_auroc, lstm_auroc_std))
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)

    # plot the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Model')
    plt.legend(fontsize='large')
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.savefig(str(disease) + '_' + str(dataset) + '_roc_curve.svg', format='svg')

    # plot PRC curve
    plt.figure()
    plt.plot(lstm_recalls, lstm_precisions, label='Seq2Seq LSTM AUPRC = %0.2f +/- %0.2f' % (lstm_auprc, lstm_auprc_std))
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    # plot baseline straight line
    baseline = lstm_earliness[2] / lstm_earliness[3]
    plt.plot([0, 1], [baseline, baseline], color='navy', linestyle='--', label='Random Model')
    plt.ylim([0, 1])

    plt.legend(fontsize='large')
    plt.savefig(str(disease) + '_' + str(dataset) + '_prc_curve.svg', format='svg')

    # plot precision, recall, specificity
    plt.figure()
    x = np.arange(len(values_to_plot))
    width = 0.4 

    plt.bar(x - width/2, lstm_values, width=width, yerr=lstm_std, capsize=5, label='Seq2Seq LSTM')
    plt.axhline(y=0.5, color='red', linestyle='--') 
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    plt.xticks(x, values_to_plot)
    plt.ylabel('Value at thr = 0.5', fontsize=font_size)
    plt.legend(fontsize='large')
    plt.savefig(str(disease) + '_' + str(dataset) + '_values.svg', format='svg')

def runner_k_aggregates(disease, dataset):

    if dataset == 'mimic':
        k_values = [1, 2, 3, 4]
    elif dataset == 'eicu':
        k_values = [6, 12, 18, 24]

    lstm_sensitivities_means = []
    lstm_specificities_means = []
    lstm_sensitivities_stds = []
    lstm_specificities_stds = []

    transformer_sensitivities_means = []
    transformer_specificities_means = []
    transformer_sensitivities_stds = []
    transformer_specificities_stds = []

    for model in ['lstm', 'transformer']:
        for k in k_values:
            mean_recall, mean_specificity, std_recall, std_specificity = get_k_agg_stats(disease, dataset, model, k)

            if model == 'lstm':

                lstm_sensitivities_means.append(mean_recall)
                lstm_specificities_means.append(mean_specificity)
                lstm_sensitivities_stds.append(std_recall)
                lstm_specificities_stds.append(std_specificity)

            elif model == 'transformer':
                    
                transformer_sensitivities_means.append(mean_recall)
                transformer_specificities_means.append(mean_specificity)
                transformer_sensitivities_stds.append(std_recall)
                transformer_specificities_stds.append(std_specificity)

    print("LSTM sensitivities: ", lstm_sensitivities_means, lstm_sensitivities_stds)
    print("LSTM specificities: ", lstm_specificities_means, lstm_specificities_stds)
    print("Transformer sensitivities: ", transformer_sensitivities_means, transformer_sensitivities_stds)
    print("Transformer specificities: ", transformer_specificities_means, transformer_specificities_stds)
    
    # plot sensitivity and specificity as two lines on the same plot, with K values on x-axis
    font_size = 15
    plt.figure()
    plt.errorbar(k_values, lstm_sensitivities_means, yerr=lstm_sensitivities_stds, label='LSTM Sensitivity', fmt='o-', markersize=5, capsize=3, elinewidth=0.5, alpha=0.5,)
    plt.errorbar(k_values, lstm_specificities_means, yerr=lstm_specificities_stds, label='LSTM Specificity', fmt='o-', markersize=5, capsize=3, elinewidth=0.5, alpha=0.5,)

    plt.xlabel('K values', fontsize=font_size)
    plt.ylabel('Value', fontsize=font_size)
    plt.xticks(k_values)
    plt.ylim([0.3, 1.0])
    plt.legend(fontsize='large')
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.savefig(str(disease) + '_' + str(dataset) + '_lstm_k_values.svg', format='svg')


    plt.figure()
    plt.errorbar(k_values, transformer_sensitivities_means, yerr=transformer_sensitivities_stds, label='Transformer Sensitivity', fmt='o-', markersize=5, capsize=3, elinewidth=0.5, alpha=0.5,)
    plt.errorbar(k_values, transformer_specificities_means, yerr=transformer_specificities_stds, label='Transformer Specificity', fmt='o-', markersize=5, capsize=3, elinewidth=0.5, alpha=0.5,)

    plt.xlabel('K values', fontsize=font_size)
    plt.ylabel('Value', fontsize=font_size)
    plt.xticks(k_values)
    plt.ylim([0.3, 1.0])
    plt.legend(fontsize='large')
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.savefig(str(disease) + '_' + str(dataset) + '_transformer_k_values.svg', format='svg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, required=True, help='The disease to evaluate', choices=['circulatory', 'respiratory_HiRID', 'kidney'])
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to evaluate', choices=['mimic', 'eicu'])
    parser.add_argument('--evaluation', type=str, required=True, help='The evaluation to perform', choices=['k_aggregates', 'plotter_real_time', 'plotter_retrospective'])

    args = parser.parse_args()

    disease = args.disease
    dataset = args.dataset
    evaluation = args.evaluation

    if evaluation == 'k_aggregates':
        runner_k_aggregates(disease, dataset)
    elif evaluation == 'plotter_real_time':
        runner_plotter(disease, dataset)
    elif evaluation == 'plotter_retrospective':
        runner_plotter_enc_dec(disease, dataset)