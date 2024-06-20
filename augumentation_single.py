import numpy as np
import csv
import os
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

class Loader:
    def __init__(self, filename, flip_x=False):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])
            print(f"{filename}: Original shape {self.rawvals.shape}")
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])
            print(self.nvals.shape)

            # Flipping X coordinates if needed
            if flip_x:
                self.nvals[:, :, 0] *= -1
                self.nvals = bone_swap(self.nvals)
                print(f"Flipping x coordinates for {filename}")

            print(f"{filename}: Reshaped to {self.nvals.shape}")

    def save(self, output_path):
        np.savetxt(output_path, self.nvals.reshape(self.nvals.shape[0], -1), delimiter=',')
        print(f"Saved augmented data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process a single file with X-flip.')
    parser.add_argument('--flip_x', action='store_true', help='Flip the x coordinates')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    # Load and process the file
    loader = Loader(args.input_file, flip_x=args.flip_x)

    # Save the processed file
    loader.save(args.output_file)

if __name__ == "__main__":
    main()
