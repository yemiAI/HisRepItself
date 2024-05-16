import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])



def slerp(frame_to, frame_from, no_inter): ##no of interpolation

    big_array = np.zeros([no_inter, 33, 3])
    #t= times
    times = np.arange(1, no_inter+1) / (no_inter+1)


    for i,t in enumerate(times):
        interpo = (1 - t) * frame_from[0,:] + t * frame_to[0,:]
        big_array[i, 0, :] =interpo


    for b in range(1, 33):
        key_times = [0, 1]
        # slerp
        key_rots = R.from_rotvec([frame_to[b], frame_from[b]])

        slerp_data = Slerp(key_times, key_rots)

        #print(times)
        interpolation = slerp_data(times)
        key_rots_vector = R.as_rotvec(interpolation, degrees = False)
        #print(key_rots_vector)

        big_array[:,b,:] = key_rots_vector

    return big_array


if (__name__ == '__main__'):

    #loader = Loader('walking_2_S11.txt')
    loader = Loader('datasets/h3.6m_periodic/S11/walkingtogether_2.txt')
    #print(loader.rawvals.shape)

    lookaheadmin = 60  # This variable appears to be unused
    lookaheadmax = 100

    per_frame_intervals = []
    best_val = {'from': None, 'to': None, 'difference': float('inf')}  # Initialize best_val with infinite difference

    for frame_from in range(loader.rawvals.shape[0] - lookaheadmax):
        frame_to = frame_from + 60 + np.argmin(
            np.linalg.norm(loader.rawvals[frame_from] - loader.rawvals[frame_from + 60:frame_from + 120], axis=1))
        diff = np.linalg.norm(loader.rawvals[frame_from] - loader.rawvals[frame_to])
        per_frame_intervals.append({'from': frame_from, 'to': frame_to, 'difference': diff})
        if diff < best_val['difference']:
            best_val = {'from': frame_from, 'to': frame_to, 'difference': diff}

    max_frames = loader.rawvals.shape[0]
    start_frame = best_val['from']
    end_frame = best_val['to']
    sliced_data  = loader.rawvals[start_frame:end_frame]
    repetitions = 30



    first_frame = sliced_data[0, :]
    first_reshaped = np.reshape(first_frame,[-1, 3])
    #print(first_reshaped.shape)
    last_frame = sliced_data[-1, :]
    last_reshaped= np.reshape(last_frame,[-1, 3])
    #print(last_reshaped.shape)

    interpolated_frames= slerp(first_reshaped, last_reshaped, 1)
    print(interpolated_frames.shape)
    interpolated_reshaped = np.reshape(interpolated_frames,[1,-1])
    print(interpolated_reshaped.shape)

    repeated_data = np.concatenate([sliced_data,interpolated_reshaped] * repetitions, axis=0)
    #filename = "forced_period/S11/walking_2.txt"
    filename = "datasets/h3.6m_periodic/S11/synthetic/walkingtogether_2.txt"
    np.savetxt(filename, repeated_data, delimiter=',', fmt='%f')
    print(f"Repeated data has been saved to {filename}")


