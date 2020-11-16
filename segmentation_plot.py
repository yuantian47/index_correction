import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

import os


def seg_plot(segdir, imdir, idx, xdim):
    top_seg_raw_up = np.array(pd.read_csv(segdir + "top/result_top_" +
                                          str(idx) + ".csv", header=None))
    bot_seg_raw_up = np.array(pd.read_csv(segdir + "top/result_bot_" +
                                          str(idx) + ".csv", header=None))
    top_seg_raw_dn = np.array(pd.read_csv(segdir + "bot/result_top_" +
                                          str(idx) + ".csv", header=None))
    bot_seg_raw_dn = np.array(pd.read_csv(segdir + "bot/result_bot_" +
                                          str(idx) + ".csv", header=None))
    top_seg_up, bot_seg_up = np.zeros((xdim, 2)), np.zeros((xdim, 2))
    top_seg_dn, bot_seg_dn = np.zeros((xdim, 2)), np.zeros((xdim, 2))
    top_seg, bot_seg = np.zeros((xdim, 2)), np.zeros((xdim, 2))
    for j in range(xdim):
        same_x_top_up = top_seg_raw_up[list([*np.where(
            top_seg_raw_up[:, 0] == j)[0]])]
        top_seg_up[j] = same_x_top_up[np.argmax(same_x_top_up[:, 1])]
        same_x_bot_up = bot_seg_raw_up[list([*np.where(
            bot_seg_raw_up[:, 0] == j)[0]])]
        bot_seg_up[j] = same_x_bot_up[np.argmax(same_x_bot_up[:, 1])]
        same_x_top_dn = top_seg_raw_dn[list([*np.where(
            top_seg_raw_dn[:, 0] == j)[0]])]
        top_seg_dn[j] = same_x_top_dn[np.argmin(same_x_top_dn[:, 1])]
        same_x_bot_dn = bot_seg_raw_dn[list([*np.where(
            bot_seg_raw_dn[:, 0] == j)[0]])]
        bot_seg_dn[j] = same_x_bot_dn[np.argmin(same_x_bot_dn[:, 1])]
        top_seg[j] = top_seg_up[j]
        bot_seg[j] = bot_seg_up[j]
        if top_seg_up[j][1] < top_seg_dn[j][1]:
            top_seg[j][1] = float(top_seg_up[j][1] + top_seg_dn[j][1]) / 2
        else:
            top_seg[j][1] = float(top_seg_dn[j][1] - 3)
        if bot_seg_up[j][1] < bot_seg_dn[j][1]:
            bot_seg[j][1] = float(bot_seg_up[j][1] + bot_seg_dn[j][1]) / 2
        else:
            bot_seg[j][1] = float(bot_seg_dn[j][1] - 3)
    img = cv.imread(imdir + "0_" + str(idx) + "_bscan.png",
                    cv.IMREAD_COLOR)
    for i in range(xdim):
        img[int(top_seg_up[i][1])][int(top_seg_up[i][0])] =\
            np.array([255, 0, 0])
        img[int(top_seg_dn[i][1])][int(top_seg_dn[i][0])] =\
            np.array([0, 0, 255])
        img[int(top_seg[i][1])][int(top_seg[i][0])] = np.array([255, 255, 0])
        img[int(bot_seg_up[i][1])][int(bot_seg_up[i][0])] =\
            np.array([255, 0, 0])
        img[int(bot_seg_dn[i][1])][int(bot_seg_dn[i][0])] =\
            np.array([0, 0, 255])
        img[int(bot_seg[i][1])][int(bot_seg[i][0])] = np.array([255, 255, 0])
    return img


if __name__ == "__main__":
    segdir = "../data/seg_res/seg_res_bss_"
    imdir = "../data/images/bss_crop/"
    vis_dir = "../data/images/vis_seg_bss/"
    if os.path.isdir(vis_dir) is False:
        os.makedirs(vis_dir)
    for k in tqdm(range(200, 601)):
        img = seg_plot(segdir, imdir, k, 416)
        cv.imwrite(vis_dir + "0_" + str(k) + "_bscan.png", img)

