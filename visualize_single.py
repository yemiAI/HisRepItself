import numpy as np
import torch

import argparse
import csv

if (__name__ == '__main__'):
    from misc import expmap2rotmat_torch, rotmat2xyz_torch
else:
    from utils.misc import expmap2rotmat_torch, rotmat2xyz_torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28,
           27, 30]

parents_noextra = []

class AnimationData:

    def build_frame(self, keypoints):
        numpoints = len(keypoints[0])

        t = np.array([np.ones(numpoints) * i for i in range(len(keypoints))]).flatten()

        x = keypoints[:, :, 0].reshape([-1])
        y = keypoints[:, :, 1].reshape([-1])
        z = keypoints[:, :, 2].reshape([-1])

        df = pd.DataFrame({'time': t,
                           'x': x,
                           'y': y,
                           'z': z})
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

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def __init__(self, animations, dots=True, skellines=False, scale=1.0, unused_bones=True, pauseatframe=-1,
                 save=None):

        self.fig = plt.figure()
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

        for idx, adata in enumerate(self.animdata):
            self.ax.append(self.fig.add_subplot(10 * len(animations) + 100 + (idx + 1), projection='3d'))
            self.animlines.append([])
            idata = adata.df[adata.df['time'] == 0]

            if (self.skellines):
                self.drawlines(idx, 0)

            if (self.dots):
                self.animdots.append(self.ax[idx].scatter(idata.x, idata.y, idata.z))

            self.ax[idx].set_xlim(-self.scale, self.scale)
            self.ax[idx].set_ylim(-self.scale, self.scale)
            self.ax[idx].set_zlim(-self.scale, self.scale)

            self.ax[idx].view_init(elev=92, azim=-90, roll=0)

        self.framecounter = plt.figtext(0.1, 0.1, "frame=0")
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.frames, interval=16)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

        if self.savefile:
            self.ani.save(filename=self.savefile, writer="ffmpeg")

        plt.show()


def phase(keypoints):

    foot = keypoints[:, 4, :]
    spine = keypoints[:, 0, :]
    footxz = foot[:, [0, 2]]
    spinexz = spine[:, [0, 2]]
    distance_diff = footxz - spinexz
    distances = np.linalg.norm(distance_diff, axis=1)
    frame = np.argmax(distances)
    print(frame)

    return frame


class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])[:, 3:]
            print(self.rawvals.shape)
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

    def xyz(self):
        rm = expmap2rotmat_torch(torch.tensor(self.nvals.reshape(-1, 3))).float().reshape(self.nvals.shape[0], 32, 3, 3)
        print(rm.shape)
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

    args = parser.parse_args()

    l = Loader(args.file)
    l_copy = l.nvals.copy()
    if args.padzeros:
        
        zeros = np.zeros([l.nvals.shape[0], 1, 3])
        #gt = np.append(zeros, l.nvals, axis=1)
        pred= np.append(zeros, l_copy, axis=1)


    else:
        gt = l.nvals

    if args.keypoint:
        anim = Animation([pred], dots=not args.nodots, skellines=args.lineplot, scale=args.scale, save=args.save)
    else:
        anim = Animation([l.xyz()], dots=not args.nodots, skellines=args.lineplot, scale=args.scale, save=args.save)
