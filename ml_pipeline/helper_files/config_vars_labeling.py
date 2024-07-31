import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from common import read_config
import pandas as pd

def get_path_vars(config_path):
    config = read_config(config_path)
    data_paths = config.get('data_paths', {})
    main_folder_path = data_paths.get('labs_path')

    return config, main_folder_path

def get_respi_fail_vars(config_path):
    config, main_folder_path = get_path_vars(config_path)

    respiratory_vars = config.get('respiratory', {})
    sampling_rate_minutes = respiratory_vars.get('sampling_rate_minutes')
    steps_per_hour= 60//sampling_rate_minutes
    peep_interpolate_limit = respiratory_vars.get('peep_interpolate_limit_hours') * steps_per_hour
    ffill_p_f_hours = respiratory_vars.get('ffill_p_f_hours') * steps_per_hour
    window_length = respiratory_vars.get('window_length_hrs') * steps_per_hour

    p_f_ratio_threshold = respiratory_vars.get('p_f_ratio_threshold')
    window_ratio = respiratory_vars.get('window_ratio')

    pao2_normal = respiratory_vars.get('pao2_normal')
    fio2_normal = respiratory_vars.get('fio2_normal')
    paco2_normal = respiratory_vars.get('paco2_normal')
    ph_normal = respiratory_vars.get('ph_normal')
    return main_folder_path, peep_interpolate_limit, ffill_p_f_hours, window_length, p_f_ratio_threshold, window_ratio, pao2_normal, fio2_normal, paco2_normal, ph_normal

def get_circ_fail_vars(config_path):
    config, main_folder_path = get_path_vars(config_path)

    circulatory_vars = config.get('circulatory', {})
    sampling_rate_minutes = circulatory_vars.get('sampling_rate_minutes')
    steps_per_hour = 60//sampling_rate_minutes
    ffill_lactate_hours = circulatory_vars.get('ffill_lactate_hours') * steps_per_hour
    ffill_drugs_hours = circulatory_vars.get('ffill_drugs_hours') * steps_per_hour
    ffill_map_hours = circulatory_vars.get('ffill_map_hours') * steps_per_hour

    lactate_normal_value = circulatory_vars.get('lactate_normal_value')
    map_normal_value = circulatory_vars.get('map_normal_value')

    lactate_threshold = circulatory_vars.get('lactate_threshold')
    map_threshold = circulatory_vars.get('map_threshold')
    window_ratio = circulatory_vars.get('window_ratio')
    window_length = circulatory_vars.get('window_length')
    hr_normal_value = circulatory_vars.get('hr_normal_value')
    hr_threshold = circulatory_vars.get('hr_threshold')
    return main_folder_path, ffill_lactate_hours, ffill_drugs_hours, ffill_map_hours, lactate_normal_value, map_normal_value, lactate_threshold, map_threshold, window_ratio, window_length, hr_normal_value, hr_threshold

def get_sepsis_vars(config_path):

    _, main_folder_path = get_path_vars(config_path)
    ids_file = pd.read_csv( main_folder_path + '0labels.txt')

    return main_folder_path, ids_file
