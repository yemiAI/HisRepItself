import numpy as np
import csv
import os
import re
import argparse



joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RSiteF", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "LSiteF", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "LSiteT", "L_Wrist_End", "LSiteH", "RightShoulder", "RightArm",
                      "RightForeArm","RightHand", "RightHandThumb", "RSiteT", "R_Wrist_End", "RSiteH"]


joint_table = []

for i, j in enumerate(joint_name):
    if j[:4] == 'Left':
        orig = joint_name.index("".join(["Right", j[4:]]))
    elif j[:5] == 'Right':
        orig = joint_name.index("".join(["Left", j[5:]]))
    elif j[0] == 'L':
        orig = joint_name.index("".join(["R", j[1:]]))
    elif j[0] == 'R':
        orig = joint_name.index("".join(["L", j[1:]]))
    else:
        orig = i
    joint_table.append(orig)


def bone_swap(animation):
    big_array = np.zeros_like(animation)
    for i, o in enumerate(joint_table):
        big_array[:, i, :] = animation[:, o, :]
    return big_array

def add_gaussian_noise(data, mean=0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    print(noise.shape)
    #exit(0)
    return data + noise

class Loader:
    def __init__(self, filename, flip_x=False, flip_z=False, augment=False):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])
            print(f"{filename}: Original shape {self.rawvals.shape}")
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])
            print(self.nvals.shape)
            #exit(0)


            # Augmentation
            if flip_x:
                self.nvals[:, :, 1:] *= -1
                # self.nvals[:, :, 0] *= -1
                self.nvals = bone_swap(self.nvals)
                print(f"Flipping x coordinates for {filename}")

            # if False: # flip_y
            #     #self.nvals[:, :, [0, 2]] *= -1
            #     self.nvals[:, :, 0] *= -1
            #     self.nvals[:, :, 2] *= -1
            #     self.nvals = bone_swap(self.nvals)
            #     print(f"Flipping x coordinates for {filename}")



            if flip_z:
                self.nvals[:, :, :2] *= -1
                self.nvals = bone_swap(self.nvals)
                print(f"Flipping z coordinates for {filename}")

            if augment:
                self.nvals = self.apply_augmentation(self.nvals)

            print(f"{filename}: Reshaped to {self.nvals.shape}")

    def apply_augmentation(self, data):
        data = add_gaussian_noise(data)
        return data

    def save(self, output_path):
        np.savetxt(output_path, self.nvals.reshape(self.nvals.shape[0], -1), delimiter=',')
        print(f"Saved augmented data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--flip_x', action='store_true', help='Flip the x coordinates')
    parser.add_argument('--flip_z', action='store_true', help='Flip the z coordinates')
    parser.add_argument('--flip_x_folder', type=str, help='Folder to dump flipped x coordinate data')
    parser.add_argument('--flip_z_folder', type=str, help='Folder to dump flipped z coordinate data')
    parser.add_argument('--gaussian_noise', action='store_true', help='Apply Gaussian noise augmentation')
    parser.add_argument('--gaussian_noise_folder', type=str, help='Folder to dump Gaussian noise augmented data')
    args = parser.parse_args()

    # Paths and patterns
    in_base_path = "datasets/h3.6m_retimed_interpolation_nozero"
    path_template1 = "S%s"
    path_template2 = "%s_%s.txt"
    file_pattern = re.compile(r"(.+)_(\d+).txt")

    # List of subjects
    subjects = ["1", "5", "6", "7", "8", "9", "11"]

    # Iterate over the directory structure
    for subject in subjects:
        subject_dir = os.path.join(in_base_path, path_template1 % subject)
        if os.path.exists(subject_dir) and os.path.isdir(subject_dir):
            for file_name in os.listdir(subject_dir):
                if file_pattern.match(file_name):
                    action, number = file_pattern.match(file_name).groups()
                    input_path = os.path.join(subject_dir, path_template2 % (action, number))
                    if os.path.exists(input_path):
                        print(f"Loading file from path {input_path}")
                        loader = Loader(input_path, flip_x=args.flip_x, flip_z=args.flip_z, augment=args.gaussian_noise)

                        # Save flipped x data if flip_x is set and flip_x_folder is provided
                        if args.flip_x and args.flip_x_folder:
                            flip_x_output_dir = os.path.join(args.flip_x_folder, path_template1 % subject)
                            os.makedirs(flip_x_output_dir, exist_ok=True)
                            flip_x_output_path = os.path.join(flip_x_output_dir, path_template2 % (action, number))
                            loader.save(flip_x_output_path)

                        # Save flipped z data if flip_z is set and flip_z_folder is provided
                        if args.flip_z and args.flip_z_folder:
                            flip_z_output_dir = os.path.join(args.flip_z_folder, path_template1 % subject)
                            os.makedirs(flip_z_output_dir, exist_ok=True)
                            flip_z_output_path = os.path.join(flip_z_output_dir, path_template2 % (action, number))
                            loader.save(flip_z_output_path)

                        # Save Gaussian noise data if gaussian_noise is set and gaussian_noise_folder is provided
                        if args.gaussian_noise and args.gaussian_noise_folder:
                            gaussian_noise_output_dir = os.path.join(args.gaussian_noise_folder, path_template1 % subject)
                            os.makedirs(gaussian_noise_output_dir, exist_ok=True)
                            gaussian_noise_output_path = os.path.join(gaussian_noise_output_dir, path_template2 % (action, number))
                            loader.save(gaussian_noise_output_path)
                    else:
                        print(f"File not found: {input_path}")
        else:
            print(f"Directory not found: {subject_dir}")

if __name__ == "__main__":
    main()
