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
        self.top_fit, self.bot_fit = None, None
        self.top_normal, self.bot_normal = None, None
        self.top_refract, self.bot_refract = None, None
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

    def fit_circle(self, layer):
        seg_mm = None
        if layer == 'top':
            seg_mm = self.top_seg_mm
        # elif layer == 'corr_bot':
        #     seg_mm = self.bot_seg_mm
        co_mat = np.zeros((seg_mm.shape[0], 3))
        co_mat[:, 0] = seg_mm[:, 0] * 2
        co_mat[:, 1] = seg_mm[:, 1] * 2
        co_mat[:, 2] = 1
        ordinate = np.zeros((seg_mm.shape[0], 1))
        ordinate[:, 0] = np.sum(np.power(seg_mm, 2), axis=1)
        res, err, _, _ = np.linalg.lstsq(co_mat, ordinate, rcond=None)
        print("The circle fitting error is:", err)
        rad = np.sqrt(np.power(res[0], 2) + np.power(res[1], 2) + res[2])
        print("The radius of the circle is:", rad)
        seg_fit = np.copy(seg_mm)
        for i in range(seg_fit.shape[0]):
            seg_fit[i, 1] = -np.sqrt(np.power(rad, 2) - np.power(seg_mm[i, 0] - res[0], 2)) + res[1]
        seg_normal = np.zeros_like(seg_mm)
        for i in range(seg_normal.shape[0]):
            normal = np.array([seg_fit[i][0] - res[0], seg_fit[i][1] - res[1]]).T
            seg_normal[i] = normal / rad
        if layer == 'top':
            self.top_fit = seg_fit
            self.top_normal = seg_normal
        # elif layer == 'corr_bot':
        #     self.bot_fit = seg_fit

    def cal_refract(self, layer):
        incidents, points, normals, r = None, None, None, None
        refracts = np.zeros_like(self.top_normal)
        if layer == 'top':
            incidents = np.repeat([[0.0, 1.0]], self.top_normal.shape[0], axis=0)
            points = self.top_fit
            normals = self.top_normal
            r = self.n1 / self.n2
            self.top_refract = np.zeros_like(normals)
        elif layer == 'bot':
            incidents = self.top_refract
            points = self.bot_fit
            normals = self.bot_normal
            r = self.n2 / self.n3
            self.bot_refract = np.zeros_like(normals)
        for i in range(points.shape[0]):
            c = -np.dot(normals[i], incidents[i])
            refract = r * incidents[i] + (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
            refracts[i] = refract / np.linalg.norm(refract)
        if layer == 'top':
            self.top_refract = refracts
        elif layer == 'bot':
            self.bot_refract = refracts


if __name__ == "__main__":
    inter_2d = Interpolation2D(416, 310, 400, 5.73, 1.68, 1., 1.466, 1.)
    inter_2d.fit_circle(layer='top')
    inter_2d.cal_refract(layer='top')
