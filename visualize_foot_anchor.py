import numpy as np
import torch
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter
import io
from mpl_toolkits.mplot3d import Axes3D

if (__name__ == '__main__'):
    from misc import expmap2rotmat_torch, rotmat2xyz_torch
else:
    from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28,
           27, 30]


stickmancolours = ['#015482', '#154406', 'black', 'orange', 'magenta'] # '#015482', '#a8ff04',

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
    def drawlines(self, aidx, frame, color='#015482'):
        #print("Drawn stickman with colour %s"%color)
        linex, liney, linez = self.animdata[aidx].build_lines(frame)
        #linex, liney, linez = self.animdata[0].build_lines(frame)
        for idx in range(len(linex)):
            # Plot the main line with default color and no blur
            self.animlines[aidx].append(self.ax[aidx].plot(linex[idx], liney[idx], linez[idx], color = color))
            # self.animlines[0].append(self.ax[0].plot(linex[idx], liney[idx], linez[idx], color=color, alpha=1.0))

            #Create a blur effect by plotting multiple lines with decreasing alpha and slight offsets
            # for offset in np.linspace(-0.05, 0.05, 10):
            #     self.animlines[0].append(
            #         self.ax[0].plot(
            #             np.array(linex[idx]) + offset,
            #             np.array(liney[idx]) + offset,
            #             np.array(linez[idx]) + offset,
            #             color=color,
            #             alpha=0.1
            #         )
            #     )

    def update_plot(self, frame):
        if frame == self.pauseatframe:
            print("Pausing")
            self.paused = True
            self.ani.pause()
        self.framecounter.set_text("frame=%d" % frame)
        count = 0
        for aidx, adata in enumerate(self.animdata):
            if (self.skellines):
                linex, liney, linez = adata.build_lines(frame)
                for idx in range(len(linex)):
                    self.animlines[aidx][idx][0].set_data_3d(linex[idx], liney[idx], linez[idx])
                if self.dots:
                    newdata = adata.df[adata.df['time'] == frame]
                    self.animdots[aidx]._offsets3d = (newdata.x, newdata.y, newdata.z)
                    # Update the main line
                    xlist = [aidx + x for x in linex[idx]]
                    #xlist = [100.0*aidx + x for x in linex[idx]]
                    # self.animlines[0][count][0].set_data(xlist, liney[idx])
                    # # self.animlines[0][count][0].set_data(linex[idx], liney[idx])
                    # self.animlines[0][count][0].set_3d_properties(linez[idx])
                    # count += 1
                    # Update the blur effect lines with slight offsets
                    # for offset in np.linspace(-0.05, 0.05, 10):
                    #     self.animlines[0][count][0].set_data(np.array(linex[idx]) + offset,
                    #                                          np.array(liney[idx]) + offset)
                    #     self.animlines[0][count][0].set_3d_properties(np.array(linez[idx]) + offset)
                    #     count += 1


        # Update rolling graphs
        self.update_rolling_graph(frame)

    def update_rolling_graph(self, frame):
        try:
            # Extract the relevant rows for the current frame
            #if frame < 50:
                #row_data_5 = np.zeros([100])
                #row_data_5_2 = np.zeros([100])

                #foot = np.zeros([100])

                #width1 = self.rolling_data.iloc[0:frame + 50, 1].shape[0]

                #endval = min(50 - frame + width1, 100)

                #row_data_5[50 - frame:endval] = self.rolling_data.iloc[0:frame + 50, 1]
                #row_data_5_2[50 - frame:endval] = self.rolling_data_2.iloc[0:frame + 50, 1]
                #foot[50 - frame:100] = self.foot_anchor.iloc[0:frame + 50, 5]
            #else:
                #row_data_5 = self.rolling_data.iloc[frame - 50:frame + 50, 1]
                #row_data_5_2 = self.rolling_data_2.iloc[frame - 50:frame + 50, 1]
                #foot = self.foot_anchor.iloc[frame - 50:frame + 50, 5]

            # Update combined rolling graph
            #self.rolling_line.set_data(range(1, len(row_data_5) + 1), row_data_5)
            #self.rolling_line_2.set_data(range(1, len(row_data_5_2) + 1), row_data_5_2)

            # Update foot anchors
            x_list = [i - frame for i in foot_list_1 if (i > frame - 50 and i < frame + 100)]
            x_list_1 = [i - frame for i in foot_list_2 if (i > frame - 50 and i < frame + 100)]
            #print(x_list_1)
            #exit(0)
            #y_list = [row_data_5[i] for i in x_list]
            #y_list = [row_data_5_2[i - (frame - 50)] for i in foot_list if (i > frame - 50 and i < frame + 50)]
            y_list = [0 for i in x_list]
            y_list_1 = [0 for i in x_list_1]
            #y_list_2 = [0 for i in x_list_1]
            self.foot_anchor1.set_data(x_list, y_list)
            self.foot_anchor2.set_data(x_list_1, y_list_1)
            #exit(0)
            self.rolling_ax.relim()
            self.rolling_ax.autoscale_view()
            self.rolling_ax.axline([50, 0], [50, 50], color="#f0944d")

        except Exception as e:
            print(f"Error updating rolling graph: {e}")


    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def __init__(self, animations, rolling_file, rolling_file_2, labels, dots=True, skellines=False, scale=1.0,
                 unused_bones=True, pauseatframe=-1, save=None):
        self.fig = plt.figure(figsize=(14, 7), facecolor='white', edgecolor='white')  # Adjust the figure size as needed
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
        #self.foot_anchor = foot_anchor

        #self.ax.append(self.fig.add_subplot(1, 1, 1, projection='3d'))
        #self.ax = self.fig.add_axes([0.1, 0.55, 0.8, 0.42])

        for idx, adata in enumerate(self.animdata):
            # Create subplots for each animation
            self.ax.append(self.fig.add_subplot(1, 2, idx + 1, projection='3d'))
            self.animlines.append([])
            idata = adata.df[adata.df['time'] == 0]
            if (self.skellines):
                print("Drawing stickman %d"%idx)
                #self.drawlines(idx, 0)
                self.drawlines(idx, 0, color = stickmancolours[idx])
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
            self.ax[idx].set_title(labels[idx])

        # Add a single rolling graph with two lines below the animations
        self.rolling_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.1])  # [left, bottom, width, height]
        #self.rolling_line, = self.rolling_ax.plot([], [], marker='.', linestyle='-', label='mpjpe_h3.6m_periodic')
        #self.rolling_line_2, = self.rolling_ax.plot([], [],  marker='.', linestyle='-', label='mpjpe_h3.6m_retimed_interpolation_errors_S5', color='green')
        self.rolling_line_3, = self.rolling_ax.plot([], [], marker='None', linestyle='-', label = 'foot_anchor_fixedDCT', color = '#015482')
        self.rolling_line_4, = self.rolling_ax.plot([], [], marker='None', linestyle='-', label = 'foot_anchor_OurRetimedAdaptiveDCT', color= 'green')# Add your new plot line here
        self.rolling_ax.set_xlim(0, 50)  # Adjust limits based on your data
        self.rolling_ax.set_ylim(0, 15)  # Adjust limits based on your data
        self.rolling_ax.set_xticks([i for i in range(0, 101, 10)], [i - 50 for i in range(0, 101, 10)])
        #self.rolling_ax.set_title('Frame predictions', loc='right', fontsize=10)
        self.rolling_ax.legend()

        self.rolling_file = rolling_file
        self.rolling_file_2 = rolling_file_2

        self.foot_anchor1, = self.rolling_ax.plot([], [],  marker='|', linestyle='None', markersize=30, markeredgewidth=2, label='Foot Anchors', color='green')  # Initialize foot anchor plot
        self.foot_anchor2, = self.rolling_ax.plot([], [], marker='|', linestyle='None', markersize=30,
                                                                    markeredgewidth=2, label='Foot Anchors',
                                                                    color='#015482')
        try:
            # Read the initial rolling graph data from CSV, skipping the first column
            self.rolling_data = pd.read_csv(self.rolling_file)
            self.rolling_data_2 = pd.read_csv(self.rolling_file_2)
            # if self.rolling_data.shape[
            #     1] == 11:  # Expecting 11 columns, with the first being time/frame and the rest being test results
            #
            #     #self.rolling_line.set_data(range(1, self.rolling_data.shape[1]), self.rolling_data.iloc[0, 1:])
            # else:
            #     raise ValueError(
            #         "The rolling file must contain exactly 11 columns, with the first column as time/frame and the next 10 columns as test results.")
        except ValueError as ve:
            print(f"Error reading CSV file: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        self.framecounter = plt.figtext(0.1, 0.95, "Frame=0")  # Position frame counter at the top left
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.frames, interval=0)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

        # Add shared X and Y labels
        self.fig.text(0.5, 0.02, 'Frame Offsets', ha='center', va='center')
        #self.fig.text(0.07, 0.15, 'MPJPE', ha='center', va='center', rotation='vertical')

        if self.savefile:
            self.ani.save(filename=self.savefile, writer="ffmpeg", fps=30)

        plt.show()



def phase(keypoints):
    foot = keypoints[:, 4, :]
    spine = keypoints[:, 0, :]
    footxz = foot[:, [0, 2]]
    spinexz = spine[:, [0, 2]]
    distance_diff = footxz - spinexz
    distances = np.linalg.norm(distance_diff, axis=1)
    frame = np.argmax(distances)
    # print(frame)
    return frame


class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, 3:]
            # print(self.rawvals.shape)
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

    def xyz(self):
        rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
        # print(rm.shape)
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
    parser.add_argument("--file1", type=str, required=True, help="First CSV file")
    parser.add_argument("--file2", type=str, required=True, help="Second CSV file")
    #parser.add_argument("--foot_anchor", type=str, required=True, help="Foot anchor positions")
    parser.add_argument("--label1", type=str, required=True, help="Label for the first animation")
    parser.add_argument("--label2", type=str, required=True, help="Label for the second animation")
    parser.add_argument("--rolling_file", type=str, help="CSV file for rolling graph", required=False)
    parser.add_argument("--rolling_file_2", type=str, help="CSV file for rolling graph", required=False)

    args = parser.parse_args()

    l1 = Loader(args.file1)
    l2 = Loader(args.file2)

    if args.padzeros:
        zeros1 = np.zeros([l1.nvals.shape[0], 1, 3])
        zeros2 = np.zeros([l2.nvals.shape[0], 1, 3])
        pred1 = np.append(zeros1, l1.nvals.copy(), axis=1)
        pred2 = np.append(zeros2, l2.nvals.copy(), axis=1)
    else:
        pred1 = l1.nvals
        pred2 = l2.nvals

    labels = [args.label1, args.label2]

    h36_anno_dict = {}
    with open('human3.6_retimed_interpolation_annotation.csv') as h36_anno:
        reader = csv.reader(h36_anno)
        header = True
        for row in reader:
            if (header):
                header = False
            else:
                # annotations = row['dataset'][:-4], row['period']
                dataset = row[0][:-4]
                #print(row)
                manual= [int(i) for i in row[2:] if len(i) > 0]
                period = float(row[1])
                h36_anno_dict[dataset] = {'period' : period, 'manual' : manual}
                #print(period)

    #if args.foot_anchor == 'periodic':
    p = h36_anno_dict['walkingtogether1_s5']['period']
    foot_list_1 = [int(p) * i + 50 for i in np.arange(0, len(h36_anno_dict['walkingtogether1_s5']['manual']))]
    print(foot_list_1)
   # else:
    foot_list_2a = h36_anno_dict['walkingtogether1_s5']['manual']
    print(foot_list_2a)
    foot_list_2 = [i + 50 for i in foot_list_2a]
    #exit(0)

    if args.keypoint:
        anim = Animation([pred1, pred2], args.rolling_file, args.rolling_file_2, labels=labels, dots=not args.nodots, skellines=args.lineplot,
                         scale=args.scale, save=args.save)
    else:
        anim = Animation([l1.xyz(), l2.xyz()], args.rolling_file, args.rolling_file_2, labels=labels, dots=not args.nodots,
                         skellines=args.lineplot, scale=args.scale, save=args.save)
