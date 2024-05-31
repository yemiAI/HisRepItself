import numpy as np
import csv
import math
import os
import re
from scipy.spatial.transform import Rotation as R
import pandas as pd
import argparse

# class Loader:
#     def __init__(self, filename):
#         with open(filename, "r") as fp:
#             reader = csv.reader(fp)
#             self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, :]
#             print(self.rawvals.shape)
#             #exit(0)
#             self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

annotation = {}
annotation_footanchor = {}
path_template1 = "S%s"
path_template2 = "%s_%s.txt"
file_pattern = "(.+)([12])_s([0-9]+).txt"
csv_file_annoation = 'human3.6_retimed_interpolation_annotation_working.csv'
csv_file_mpjpe ='S5_walking_1_MPJPE_annotated.csv'
def interpolation(dataset, anchors_to, anchors_from):

    out_f = anchors_to[-2]# number of steps
    # anchors_to[len(anchors_to) - 1]

    big_array = np.zeros([int(out_f), dataset.shape[1]])
    for idx, sframe in enumerate(anchors_from[:-2]):
        eframe = anchors_from[idx + 1]

        input_step = dataset[sframe:eframe + 1, :]

        output_step_start = anchors_to[idx]
        output_step_end = anchors_to[idx + 1]

        input_step_size = eframe - sframe
        if (input_step.shape[0] <= input_step_size):
            continue
        print("Input step vs size: ", input_step.shape[0], input_step_size)

        for i in range(output_step_end - output_step_start):

            t_prime = i * input_step_size / (output_step_end - output_step_start)
            t_prime_before = math.floor(t_prime)
            t_prime_after = math.ceil(t_prime)
            t = t_prime - t_prime_before
            output_frame_line = output_step_start + i
            if t == 0:

                big_array[output_frame_line, :] = input_step[t_prime_before, :]

            else:
                #print (sframe, eframe, input_step.shape)
                expmap_first_frame_0 = input_step[t_prime_before, :]
                expmap_second_frame_0 = input_step[t_prime_after, :]

                bone_0 = (1 - t) * expmap_first_frame_0
                interpo = (1 - t) * expmap_first_frame_0 + t * expmap_second_frame_0
                big_array[output_frame_line, :] = interpo
    return big_array

with open(csv_file_annoation, 'r') as file:
     reader = csv.reader(file)
     for row in reader:

         if len(row)> 1:
             annotation[row[0]] = { 'period' : math.floor(float(row[1])), 'foot_anchors' : [int (c) for c in row[2:] if len(c) > 0]}
target = annotation['walking1_s5.txt']
retimed_anchors = [target['period'] * i for i in range(len(target['foot_anchors']))]


with open(csv_file_mpjpe, 'r') as file:
    reader = csv.reader(file)
    #header = next(reader)  # Assuming the first row contains column headers

    # Assuming the first column is used as keys for both dictionaries
    data = []
    for row in reader:
        data.append([float(c) for c in row[1:] if len(c) > 0])
    data_values = np.array(data)

#rescaled_errors = interpolation(data_values, target['foot_anchors'], retimed_anchors)
rescaled_errors = interpolation(data_values, retimed_anchors, target['foot_anchors'])
print(data_values.shape)
print(rescaled_errors)
transposed_fix = np.expand_dims(np.arange(rescaled_errors.shape[0]), 0)

out_array = np.zeros([rescaled_errors.shape[0], 11])
out_array[:, 0] = transposed_fix
out_array[:, 1:] = rescaled_errors
#np.savetxt('interpolated_errors_test_wave_73.csv', out_array, delimiter= ",")
#np.savetxt('interpolated_errors_S5_original_to_retimed_annotated.csv', out_array, delimiter= ",")
np.savetxt('try_2.csv', out_array, delimiter= ",")
#print(data_values.shape)

