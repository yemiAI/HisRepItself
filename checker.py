import numpy as np
import pandas as pd
import csv

def get_csv(filename):

    with open(filename, 'r') as fp:
        reader = csv.reader(fp)

        clist = [[float(c) for c in row] for row in reader]
    return np.array([v[1:] for v in clist])

foot_anchors = [0,95,161,232,303,376,451,521,591,663,733,812,896,981,1062,1136,1214,1287,1359,1433,1511,1583,1659,1732,1807,1879,1952,2024,2092,2156,2218,2282,2350,2423,2499,2572,2647,2721,2790,2862,2932]


retimed_anchors = [72 * i for i in range(len(foot_anchors))]

retimedfile = "S5_walking_1_retimed_interpolation_MPJPE_working.csv"

origfile = "S5_walking_1_MPJPE.csv"

origdata = get_csv(origfile)
retimeddata = get_csv(retimedfile)

count = 0
total = 0
for i, ofa_ in enumerate(foot_anchors):
    ofa = ofa_ - 50
    rfa = retimed_anchors[i] - 50
    ol = origdata[ofa, 5]
    rl = retimeddata[rfa, 5]
    total += 1
    if (rl > ol):
        count += 1
    print("%d, %d, %f, %d, %f, %f"%(i, ofa, ol, rfa, rl, rl - ol))


print("Count is %d/%d"%(count, total))