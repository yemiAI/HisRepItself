import numpy as np
import torch
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

if (__name__ == '__main__'):
    from misc import expmap2rotmat_torch, rotmat2xyz_torch
else:
    from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28,
           27, 30]


class AnimationData:
    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])
        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()
        x = keypoints[:, :, 0].reshape([-1])
        y = keypoints[:, :, 1].reshape([-1])
        z = keypoints[:, :, 2].reshape([-1])
        df = pd.DataFrame({'time': t, 'x': x, 'y': y, 'z': z})
        return df

    def unpack_extras(self, data, used):
        clones = {
            31: 30,
            28: 27,
            24: 13,
            16: 13,
            23: 22,
            20: 19
        }
        fixed = {1: np.array([-132.9486, 0, 0]),
                 6: np.array([132.94882, 0, 0]),
                 11: np.array([0, 0.1, 0])}
        retval = np.zeros([data.shape[0], 32, 3])
        for fromi, toi in enumerate(used):
            retval[:, toi, :] = data[:, fromi, :]
        for f in fixed:
            retval[:, f, :] = fixed[f]
        for c in clones:
            retval[:, c, :] = retval[:, clones[c], :]
        return retval

    def build_lines(self, num):
        linex = []
        liney = []
        linez = []
        for f in self.used_bones:
            t = parents[f]
            if (t >= 0):
                linex.append([self.df.x[num * 32 + f], self.df.x[num * 32 + t]])
                liney.append([self.df.y[num * 32 + f], self.df.y[num * 32 + t]])
                linez.append([self.df.z[num * 32 + f], self.df.z[num * 32 + t]])
        return [linex, liney, linez]

    def __init__(self, data, extra_bones):
        self.used_bones = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
        self.extra_bones = extra_bones
        if (not extra_bones):
            self.data = self.unpack_extras(data, self.used_bones)
        else:
            self.data = data
        self.df = self.build_frame(self.data)


class Animation:
    def drawlines(self, aidx, frame):
        linex, liney, linez = self.animdata[aidx].build_lines(frame)
        for idx in range(len(linex)):
            self.animlines[aidx].append(self.ax[aidx].plot(linex[idx], liney[idx], linez[idx]))

    def update_plot(self, frame):
        if frame == self.pauseatframe:
            print("Pausing")
            self.paused = True
            self.ani.pause()
        self.framecounter.set_text("frame=%d" % frame)
        for aidx, adata in enumerate(self.animdata):
            if (self.skellines):
                linex, liney, linez = adata.build_lines(frame)
                for idx in range(len(linex)):
                    self.animlines[aidx][idx][0].set_data_3d(linex[idx], liney[idx], linez[idx])
            if (self.dots):
                newdata = adata.df[adata.df['time'] == frame]
                self.animdots[aidx]._offsets3d = (newdata.x, newdata.y, newdata.z)

        # Update rolling graphs
        self.update_rolling_graph(frame)

    def update_rolling_graph(self, frame):
        try:
            # Extract the relevant rows for the current frame
            row_data_1 = self.rolling_data.iloc[frame-50:frame+50, 1]
            row_data_5 = self.rolling_data.iloc[frame-50:frame+50, 5]
            row_data_10 = self.rolling_data.iloc[frame-50:frame+50, 10]

            frame_no = frame
            #print("The frame number is %d" % frame_no)

            # Update main rolling graph
            #self.rolling_line.set_data(range(1, len(row_data_1) + 1), row_data_1)
            # self.rolling_ax.relim()
            # self.rolling_ax.autoscale_view()

            # Update additional subplots
            self.rolling_line.set_data(range(1, len(row_data_1) + 1), row_data_1)
            self.rolling_ax.relim()
            self.rolling_ax.autoscale_view()
            self.rolling_ax.axline([50, 0], [50, 30], color="orange")

            self.rolling_line1.set_data(range(1, len(row_data_5) + 1), row_data_5)
            self.rolling_ax1.relim()
            self.rolling_ax1.autoscale_view()

            self.rolling_ax1.axline([50, 0], [50, 30], color= "orange")


            self.rolling_line2.set_data(range(1, len(row_data_10) + 1), row_data_10)
            self.rolling_ax2.relim()
            self.rolling_ax2.autoscale_view()
            self.rolling_ax2.axline([50, 0], [50, 30], color="orange")
            #print(f"Updating rolling graph for frame {frame_no}: {row_data_5.values}")

        except Exception as e:
            print(f"Error updating rolling graph: {e}")

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def __init__(self, animations, rolling_file, dots=True, skellines=False, scale=1.0, unused_bones=True,
                 pauseatframe=-1, save=None):
        self.fig = plt.figure(figsize=(10, 10), facecolor='white', edgecolor= 'white')  # Adjust the figure size as needed
        self.skellines = skellines
        self.dots = dots
        self.scale = scale
        self.paused = False
        self.pauseatframe = pauseatframe
        self.ax = []
        self.extra_bones = unused_bones
        self.frames = animations[0].shape[0]
        self.animdata = [AnimationData(anim, self.extra_bones) for anim in animations]
        self.animlines = []
        self.animdots = []
        self.savefile = save
        self.rolling_texts = []

        for idx, adata in enumerate(self.animdata):
            # Place the main animation at the top
            self.fig.suptitle('Human 3.6m S5 walking')
            self.ax.append(self.fig.add_axes([0.1, 0.57, 0.8, 0.42], projection='3d'))  # [left, bottom, width, height]
            self.animlines.append([])
            idata = adata.df[adata.df['time'] == 0]
            if (self.skellines):
                self.drawlines(idx, 0)
            if (self.dots):
                self.animdots.append(self.ax[idx].scatter(idata.x, idata.y, idata.z))
            self.ax[idx].set_xlim(-self.scale, self.scale)
            self.ax[idx].set_ylim(-self.scale, self.scale)
            self.ax[idx].set_zlim(-self.scale, self.scale)
            self.ax[idx].grid(visible=True)

            self.ax[idx].set_xticks([i for i in range(-1000, 1000, 250)], ["" for i in range(-1000, 1000, 250)])
            self.ax[idx].set_yticks([i for i in range(-1000, 1000, 250)], ["" for i in range(-1000, 1000, 250)])
            self.ax[idx].set_zticks([i for i in range(-1000, 1000, 250)], ["" for i in range(-1000, 1000, 250)])

            self.ax[idx].view_init(elev=147, azim=-90, roll=0)

        # Add vertically aligned rolling graph subplots
        # 1
        self.rolling_ax = self.fig.add_axes([0.1, 0.49, 0.8, 0.13])  # [left, bottom, width, height]
        self.rolling_line, = self.rolling_ax.plot([], [], marker='o', linestyle='-')
        self.rolling_ax.set_xlim(0,50)  # Adjust limits based on your data
        self.rolling_ax.set_ylim(0, 30)  # Adjust limits based on your data

        #self.rolling_ax.set_subylabel('Value')
        # self.rolling_ax.set_subylabel('Value')
        self.rolling_ax.set_xticks([i for i in range(0,101,10)], [i - 50 for i in range(0,101,10)])

        self.rolling_ax.set_title('Frame 1 prediction', loc= 'right', fontsize= 10)
        #self.rolling_ax.set_ylabel('Te')
        # 5
        self.rolling_ax1 = self.fig.add_axes([0.1, 0.28, 0.8, 0.15])  # [left, bottom, width, height]
        self.rolling_line1, = self.rolling_ax1.plot([], [], marker='o', linestyle='-')
        self.rolling_ax1.set_xlim(0, 100)  # Adjust limits based on your data
        self.rolling_ax1.set_ylim(0, 95)  # Adjust limits based on your data
        self.rolling_ax1.set_xticks([i for i in range(0, 101, 10)], [i - 50 for i in range(0, 101, 10)])
        self.rolling_ax1.set_title('Frame 5 prediction', loc='right', fontsize= 10)
        #self.rolling_ax.set_ylabel('Time')
        # 10
        self.rolling_ax2 = self.fig.add_axes([0.1, 0.07, 0.8, 0.15])  # [left, bottom, width, height]
        self.rolling_line2, = self.rolling_ax2.plot([], [], marker='o', linestyle='-')
        self.rolling_ax2.set_xlim(0, 100)  # Adjust limits based on your data
        self.rolling_ax2.set_ylim(0, 160)  # Adjust limits based on your data
        self.rolling_ax2.set_xticks([i for i in range(0, 101, 10)], [i - 50 for i in range(0, 101, 10)])
        self.rolling_ax2.set_title('Frame 10 prediction', loc= 'right', fontsize= 10)
        #self.rolling_ax.set_ylabel('Time')
        self.rolling_file = rolling_file



        try:
            # Read the initial rolling graph data from CSV, skipping the first column
            self.rolling_data = pd.read_csv(self.rolling_file)
            #print("Initial rolling data read:")
            #print(self.rolling_data.head())
            if self.rolling_data.shape[
                1] == 11:  # Expecting 11 columns, with the first being time/frame and the rest being test results
                self.rolling_line.set_data(range(1, self.rolling_data.shape[1]), self.rolling_data.iloc[0, 1:])
            else:
                raise ValueError(
                    "The rolling file must contain exactly 11 columns, with the first column as time/frame and the next 10 columns as test results.")
        except ValueError as ve:
            print(f"Error reading CSV file: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        self.framecounter = plt.figtext(0.1, 0.95, "frame=0")  # Position frame counter at the top left
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.frames, interval=0)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

        # Add shared X and Y labels
        self.fig.text(0.5, 0.02, 'Frame Offsets', ha='center', va='center')
        self.fig.text(0.02, 0.35, 'MPJPE', ha='center', va='center', rotation='vertical')

        if self.savefile:
            self.ani.save(filename=self.savefile, writer="ffmpeg", fps= 30)

        plt.show()


def phase(keypoints):
    foot = keypoints[:, 4, :]
    spine = keypoints[:, 0, :]
    footxz = foot[:, [0, 2]]
    spinexz = spine[:, [0, 2]]
    distance_diff = footxz - spinexz
    distances = np.linalg.norm(distance_diff, axis=1)
    frame = np.argmax(distances)
    #print(frame)
    return frame


class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, 3:]
            #print(self.rawvals.shape)
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

    def xyz(self):
        rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
        #print(rm.shape)
        return rotmat2xyz_torch(rm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, help="Scaling factor", default=1000.0)
    parser.add_argument("--lineplot", action='store_true', help="Draw a skel")
    parser.add_argument("--nodots", action='store_true', help="Line only, no dots")
    parser.add_argument("--keypoint", action='store_true', help="Line only, no dots")
    parser.add_argument("--output", action='store_true', help="Visualize model output too")
    parser.add_argument("--model_pth", type=str, help="Draw a skel")
    parser.add_argument("--padzeros", action='store_true', help="zero padding")
    parser.add_argument("--save", type=str, help="save the output of a model")
    parser.add_argument("--file", type=str)
    parser.add_argument("--rolling_file", type=str, help="CSV file for rolling graph", required=True)

    args = parser.parse_args()

    l = Loader(args.file)
    l_copy = l.nvals.copy()
    if args.padzeros:
        zeros = np.zeros([l.nvals.shape[0], 1, 3])
        pred = np.append(zeros, l_copy, axis=1)
    else:
        gt = l.nvals

    if args.keypoint:
        anim = Animation([pred], args.rolling_file, dots=not args.nodots, skellines=args.lineplot, scale=args.scale,
                         save=args.save)
    else:
        anim = Animation([l.xyz()], args.rolling_file, dots=not args.nodots, skellines=args.lineplot, scale=args.scale,
                         save=args.save)
