import numpy as np
import pandas as pd
import cv2 as cv
from scipy import interpolate, spatial
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Interpolation2D:

    def __init__(self, xdim, zdim, yidx, xlength, zlength, n1, n2, n3):
        self.xdim, self.zdim = xdim, zdim
        self.yidx = yidx
        self.xlength, self.zlength = xlength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.top_fit, self.bot_fit = None, None
        self.top_normal, self.bot_normal = None, None
        self.top_refract, self.bot_refract = None, None
        self.linear_interpolator = None
        self.poly_order = 4
        top_seg_raw = np.array(pd.read_csv(
            "../data/seg_res/seg_res_bss/result_top_" + str(
                yidx) + ".csv",
            header=None))
        bot_seg_raw = np.array(pd.read_csv(
            "../data/seg_res/seg_res_bss/result_bot_" + str(
                yidx) + ".csv",
            header=None))
        self.top_seg, self.bot_seg = np.zeros((xdim, 2)), np.zeros((xdim, 2))
        for i in range(xdim):
            same_x_top = top_seg_raw[list([*np.where(top_seg_raw[:, 0] == i)[0]])]
            self.top_seg[i] = same_x_top[np.argmax(same_x_top[:, 1])]
            same_x_bot = bot_seg_raw[list([*np.where(bot_seg_raw[:, 0] == i)[0]])]
            self.bot_seg[i] = same_x_bot[np.argmax(same_x_bot[:, 1])]
        self.top_seg_mm = np.multiply(self.top_seg,
                                      [float(xlength) / self.xdim,
                                       float(zlength) / self.zdim])
        self.bot_seg_mm = np.multiply(self.bot_seg,
                                      [float(xlength) / self.xdim,
                                       float(zlength) / self.zdim])
        self.corr_bot_seg_mm = None
        self.images = cv.imread(
            "../data/images/bss_760_crop/0_" + str(
                yidx) + "_bscan.png",
            cv.IMREAD_GRAYSCALE)
        self.values, self.rays = None, None
        self.bot_rays = np.zeros((xdim, 2))

    def fit_circle(self, layer):
        seg_mm = None
        if layer == 'top':
            seg_mm = self.top_seg_mm
        elif layer == 'corr_bot':
            seg_mm = self.corr_bot_seg_mm
        co_mat = np.zeros((seg_mm.shape[0], 3))
        co_mat[:, 0] = seg_mm[:, 0] * 2
        co_mat[:, 1] = seg_mm[:, 1] * 2
        co_mat[:, 2] = 1
        ordinate = np.zeros((seg_mm.shape[0], 1))
        ordinate[:, 0] = np.sum(np.power(seg_mm, 2), axis=1)
        res, _, _, _ = np.linalg.lstsq(co_mat, ordinate, rcond=None)
        rad = np.sqrt(np.power(res[0], 2) + np.power(res[1], 2) + res[2])
        print("The radius of the circle is:", rad)
        seg_fit = np.copy(seg_mm)
        for i in range(seg_fit.shape[0]):
            seg_fit[i, 1] = -np.sqrt(
                np.power(rad, 2) - np.power(seg_mm[i, 0] - res[0], 2)) + res[1]
        fitting_error = np.sum(np.power(seg_mm[:, 1] - seg_fit[:, 1], 2))
        print("The circle fitting error is:", fitting_error)
        seg_normal = np.zeros_like(seg_mm)
        for i in range(seg_normal.shape[0]):
            normal = np.array(
                [seg_fit[i][0] - res[0], seg_fit[i][1] - res[1]]).T
            seg_normal[i] = normal / rad
        if layer == 'top':
            self.top_fit = seg_fit
            self.top_normal = seg_normal
        elif layer == 'corr_bot':
            self.bot_fit = seg_fit
            self.bot_normal = seg_normal

    def fit_poly(self, layer, order):
        seg_mm = None
        if layer == 'top':
            seg_mm = self.top_seg_mm
        elif layer == 'corr_bot':
            seg_mm = self.corr_bot_seg_mm
        p, _, _, _, _ = np.polyfit(seg_mm[:, 0], seg_mm[:, 1], order,
                                 rcond=False, full=True)
        print("The coffeicients are:", p.tolist())
        p_class = np.poly1d(p, r=False)
        p2_class = np.polyder(p_class)
        seg_fit = np.copy(seg_mm)
        seg_normal = np.zeros_like(seg_mm)
        for i in range(seg_fit.shape[0]):
            seg_fit[i][1] = p_class(seg_fit[i][0])
            seg_normal[i] = np.array([p2_class(seg_fit[i][0]), -1.]) / \
                            np.linalg.norm([p2_class(seg_fit[i][0]), -1.])
        fitting_error = np.sum(np.power(seg_mm[:, 1] - seg_fit[:, 1], 2))
        print("The poly fitting error is:", fitting_error)
        if layer == 'top':
            self.top_fit = seg_fit
            self.top_normal = seg_normal
        elif layer == 'corr_bot':
            self.bot_fit = seg_fit
            self.bot_normal = seg_normal

    def cal_refract(self, layer):
        if layer == 'top':
            # self.fit_circle(layer='top')
            self.fit_poly(layer='top', order=self.poly_order)
        incidents, points, normals, r = None, None, None, None
        refracts = np.zeros_like(self.top_normal)
        if layer == 'top':
            incidents = np.repeat([[0.0, 1.0]], self.top_normal.shape[0],
                                  axis=0)
            points = self.top_fit
            normals = self.top_normal
            r = self.n1 / self.n2
            self.top_refract = np.zeros_like(normals)
        elif layer == 'corr_bot':
            incidents = self.top_refract
            points = self.bot_fit
            normals = self.bot_normal
            r = self.n2 / self.n3
            self.bot_refract = np.zeros_like(normals)
        for i in range(points.shape[0]):
            c = -np.dot(normals[i], incidents[i])
            refract = r * incidents[i] + (r * c - np.sqrt(
                1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
            refracts[i] = refract / np.linalg.norm(refract)
        if layer == 'top':
            self.top_refract = refracts
        elif layer == 'corr_bot':
            self.bot_refract = refracts

    def top_refraction_correction(self, refract, origin, points, group_idx):
        distance = np.absolute(origin[1] - points[:, 1]) / group_idx
        corrected_points = origin + (
                    distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def bot_refraction_correction(self, refract, origin, points, group_idx):
        distance = np.asarray([np.linalg.norm(origin - points[i]) for i in
                               range(points.shape[0])]) / group_idx
        corrected_points = origin + (
                    distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def linear_inter_pairs(self):
        values = cv.imread(
            "../data/images/bss_760_crop/0_" + str(
                self.yidx) + "_bscan.png",
            cv.IMREAD_GRAYSCALE).transpose()
        rays = np.zeros((self.xdim, self.zdim, 2))
        print("Building linear interpolation function")
        self.corr_bot_seg_mm = np.zeros_like(self.bot_seg_mm)
        for i in tqdm(range(self.xdim)):
            rays[i] = np.asarray([[(float(i) / self.xdim) * self.xlength,
                                   (float(j) / self.zdim) * self.zlength] for j
                                  in range(self.zdim)])
            top_point = np.argwhere(rays[i][:, 1] > self.top_fit[i][1])[0, 0]
            rays[i][top_point:] = self.top_refraction_correction(
                self.top_refract[i], self.top_fit[i],
                rays[i][top_point:], self.n2)
            self.corr_bot_seg_mm[i] = rays[i][int(self.bot_seg[i][1])]
        # self.fit_circle(layer='corr_bot')
        self.fit_poly(layer='corr_bot', order=self.poly_order)
        self.cal_refract(layer='corr_bot')
        for i in tqdm(range(self.xdim)):
            bot_point = np.argwhere(rays[i][:, 1] > self.bot_fit[i][1])[0, 0]
            rays[i][bot_point:] = self.bot_refraction_correction(
                self.bot_refract[i], self.bot_fit[i],
                rays[i][bot_point:], self.n3 / self.n2)
        self.bot_rays = rays[:, -1]
        self.rays, self.values = rays.reshape((-1, 2)), values.reshape((-1, 1))
        self.linear_interpolator = \
            interpolate.LinearNDInterpolator(self.rays,
                                             self.values,
                                             fill_value=-1.0,
                                             rescale=True)

    def bot_convex_hull_check(self, pos):
        if self.bot_rays[0, 0] < pos[0] < self.bot_rays[-1, 0]:
            first_idx = np.argwhere(self.bot_rays[:, 0] > pos[0])[0, 0]
            second_idx = first_idx - 1
            if self.bot_rays[first_idx, 1] > pos[1] and self.bot_rays[
                second_idx, 1] > pos[1]:
                return True
            else:
                return False
        else:
            return True

    def convex_hull_check(self, pos):
        try:
            first_idx = np.argwhere(self.rays[:self.zdim, 1] > pos[1])[0, 0]
        except IndexError:
            return self.bot_convex_hull_check(pos)
        if first_idx is not None and first_idx <= self.zdim - 1:
            second_idx = first_idx + 1
            if pos[0] < self.rays[first_idx, 0] or pos[0] < self.rays[
                second_idx, 0]:
                return False
        elif first_idx is not None:
            second_idx = first_idx
            if pos[0] < self.rays[first_idx, 0] or pos[0] < self.rays[
                second_idx, 0]:
                return False
        try:
            first_idx = np.argwhere(self.rays[-self.zdim:, 1] > pos[1])[0, 0]
        except IndexError:
            return self.bot_convex_hull_check(pos)
        if first_idx is not None and first_idx <= self.zdim - 1:
            second_idx = first_idx + 1
            if pos[0] > self.rays[-self.zdim + first_idx - 1, 0] or pos[0] > \
                    self.rays[-self.zdim + second_idx - 1, 0]:
                return False
        elif first_idx is not None:
            second_idx = first_idx
            if pos[0] > self.rays[-self.zdim + first_idx - 1, 0] or pos[0] > \
                    self.rays[-self.zdim + second_idx - 1, 0]:
                return False
        return self.bot_convex_hull_check(pos)

    def reconstruction(self, x_padding=15):
        img = np.full((self.xdim + 2 * x_padding, self.zdim), 255,
                      dtype=np.uint8)
        print("Reconstructing the image.")
        for i in tqdm(range(-x_padding, self.xdim + x_padding)):
            for j in range(self.zdim):
                pos = np.multiply([i, j], [self.xlength / self.xdim,
                                           self.zlength / self.zdim])
                if self.convex_hull_check(pos):
                    img[i + x_padding][j] =\
                        np.uint8(self.linear_interpolator(pos))
        img = cv.cvtColor(img.transpose(), cv.COLOR_GRAY2RGB)
        # Visualize raw segmentation result
        for i in self.top_seg:
            img[int(i[1])][int(i[0]) + x_padding] = np.array([255, 0, 0])
        for i in self.bot_seg:
            img[int(i[1])][int(i[0]) + x_padding] = np.array([0, 255, 0])
        # Visualize fitting segmentation result
        for i, j in zip(self.top_fit, self.bot_fit):
            img[int((i[1] / self.zlength) * self.zdim)][
                int((i[0] / self.xlength) * self.xdim) + x_padding] = \
                np.array([255, 128, 0])
            img[int((j[1] / self.zlength) * self.zdim)][
                int((j[0] / self.xlength) * self.xdim) + x_padding] = \
                np.array([128, 255, 0])
        return img


if __name__ == "__main__":
    inter_2d = Interpolation2D(416, 677, 400, 5.81, 3.67, 1., 1.466, 1.3350)
    inter_2d.cal_refract(layer='top')
    inter_2d.linear_inter_pairs()
    img = inter_2d.reconstruction()
    plt.imshow(img)
    plt.show()

    hull = spatial.ConvexHull((inter_2d.rays * np.array([1., -1.])))
    plt.plot(inter_2d.rays[:, 0], inter_2d.rays[:, 1] * -1., '.')
    for idx, simplex in enumerate(hull.simplices):
        plt.plot(inter_2d.rays[simplex, 0], inter_2d.rays[simplex, 1] * -1.,
                 'r-')
    plt.show()
