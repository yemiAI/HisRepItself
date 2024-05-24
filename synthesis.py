import numpy as np 
import csv
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math
import os
import re


class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, :]
            print(self.rawvals.shape)
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

            

    # def xyz(self):
    #     rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
    #     print(rm.shape)
    #     return rotmat2xyz_torch(rm)


def interpolation(inputfile, outputfile, datafile):
    loader = Loader(inputfile)
    exp_size_a = np.linalg.norm(loader.nvals, axis=2)
    period = annotation[datafile]['period']
    rotation_axis = R.from_rotvec(loader.nvals[2, 2], degrees=False)
    out_f = len(annotation[datafile]['foot_anchors']) * annotation[datafile]['period']  # number of steps
    big_array = np.zeros([int(out_f) ,33, 3])
    for idx, sframe in enumerate(annotation[datafile]['foot_anchors'][:-1]):
        eframe = annotation[datafile]['foot_anchors'][idx + 1]

        input_step = loader.nvals[sframe:eframe+1, :, :]
        output_step_start = period * idx
        #print(output_step_start)
        output_step_end = period *(idx + 1)
        #print(output_step_end)

        input_step_size = eframe - sframe
        for i in range(period):
            t_prime = i * input_step_size / (output_step_end - output_step_start)
            #print(t_prime)

            t_prime_before = math.floor(t_prime)
            #print("before",t_prime_before)
            t_prime_after = math.ceil(t_prime)
            #print("after", t_prime_after)
            t = t_prime - t_prime_before
            #print("This is T ",t)

            output_frame_line = output_step_start + i

            if t== 0:

                big_array[output_frame_line, :, :] = input_step[t_prime_before, :, :]

            else:
                expmap_first_frame_0 = input_step[t_prime_before ,0 ,:]
                expmap_second_frame_0 = input_step[t_prime_after, 0, :]

                bone_0 = (1-t) * expmap_first_frame_0
                interpo = (1-t) * expmap_first_frame_0 + t * expmap_second_frame_0
                big_array[output_frame_line, 0, :] = interpo
            #print(output_frame_line)
            for b in range(1, 33):
                expmap_first_frame = input_step[t_prime_before ,b ,:]
                expmap_second_frame = input_step[t_prime_after, b, :]
                key_rots = R.from_rotvec([expmap_first_frame, expmap_second_frame])
                key_times = [0,1]

                #slerp
                slerp = Slerp(key_times,key_rots)
                times = [t]
                #print(slerp)
                interpolated_rots = slerp(times)
                #print(interpolated_rots)

                key_rots_vector = R.as_rotvec(interpolated_rots, degrees=False)

                #print("First",expmap_first_frame)
                #print("second",expmap_second_frame)
                #print("rot vec",key_rots_vector)


                big_array[output_frame_line, b, :] = key_rots_vector
    big_array_reshaped = np.reshape(big_array, [big_array.shape[0], -1])

    print(big_array.shape)
    output_dir = os.path.split(outputfile)[0]
    os.makedirs(output_dir, exist_ok = True)
    np.savetxt(outputfile, big_array_reshaped, delimiter= ",")
    print("File has been saved successfully")



in_base_path = "datasets/h3.6m_periodic"
out_base_path = "datasets/MPJPE_squashed"



path_template1 = "S%s"
path_template2 = "%s_%s.txt"
file_pattern = "(.+)([12])_s([0-9]+).txt"
annotation = {}
annotation_footanchor = {}
csv_file = 'human3.6_retimed_interpolation_annotation_working.csv'



with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    #header = next(reader)  # Assuming the first row contains column headers

    # Assuming the first column is used as keys for both dictionaries
    for row in reader:

        if len(row)> 1:            
            annotation[row[0]] = { 'period' : math.floor(float(row[1])), 'foot_anchors' : [int (c) for c in row[2:] if len(c) > 0]}

            m = re.match(file_pattern, row[0])
            if (m):
                action, subaction, subject = m.groups()
                output_path = os.path.join(out_base_path, path_template1 % (subject),
                                          path_template2 % (action, subaction))
                input_path = os.path.join(in_base_path, path_template1 % (subject),
                                          path_template2 % (action, subaction))
                print("Successfully built path %s" % output_path)
                interpolation(input_path, output_path, row[0])
            else:
                print("Failure reading %s" % row[0])


# datafile = 'walkingtogether2_s9.txt'
datafile = row[0]

print("THis is the datafile",datafile)
#exit(0)


#print(out_f)

# create numpy array for new interpolated values
#big_array = np.zeros([int(out_f) ,33, 3])
#print(big_array.shape)

#period = annotation[datafile]['period']

