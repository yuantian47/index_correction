import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib as plt
from scipy import interpolate


class Interpolation2D:

    def __init__(self, xdim, zdim, yidx, xlength, zlength, n1, n2, n3):
        self.xdim, self.zdim = xdim, zdim
        self.yidx = yidx
        self.xlength, self.zlength = xlength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        top_seg_raw = np.array(pd.read_csv("../data/seg_res/seg_res_calib_760/result_top_" + str(yidx) + ".csv",
                                           header=None))
        bot_seg_raw = np.array(pd.read_csv("../data/seg_res/seg_res_calib_760/result_bot_" + str(yidx) + ".csv",
                                           header=None))
        self.top_seg, self.bot_seg = np.zeros((xdim, 2)), np.zeros((xdim, 2))
        for i in range(xdim):
            same_x_top = top_seg_raw[list([*np.where(top_seg_raw[:, 0] == i)[0]])]
            self.top_seg[i] = same_x_top[np.argmax(same_x_top[:, 1])]
            same_x_bot = bot_seg_raw[list([*np.where(bot_seg_raw[:, 0] == i)[0]])]
            self.bot_seg[i] = same_x_bot[np.argmax(same_x_bot[:, 1])]
        self.top_seg_mm = np.multiply(self.top_seg, [float(xlength) / self.xdim, float(zlength) / self.zdim])
        self.bot_seg_mm = np.multiply(self.bot_seg, [float(xlength) / self.xdim, float(zlength) / self.zdim])
        self.images = cv.imread("../data/images/contact_lens_crop_calib_760/0_" + str(yidx) + "_bscan.png",
                                cv.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    inter_2d = Interpolation2D(416, 310, 400, 5.73, 1.68, 1., 1.466, 1.)
