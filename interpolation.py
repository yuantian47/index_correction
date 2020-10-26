import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import scipy.interpolate
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import csv_pcd


class Interpolation:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength,
                 ylength, zlength, n1, n2, n3, dp_x, dp_y, dp_z):
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.dp_x, self.dp_y, self.dp_z = dp_x, dp_y, dp_z
        self.directory = directory
        self.idx_range = idx_range
        self.seg = csv_pcd.RealPCD(directory, idx_range, xdim, ydim, zdim,
                                   xlength, ylength, zlength, n1, n2, n3)
        self.seg.remove_outlier(layer='top')
        self.seg.pcd_fit_sphere(method='ls')
        self.top_smooth_pcd = self.seg.get_top_smooth_pcd()
        self.top_smooth_pcd.paint_uniform_color([1, 0, 0])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(
            self.top_smooth_pcd.points).shape[0], axis=0))
        self.seg.refraction_correction()
        self.seg.remove_outlier(layer='corrected_bot')
        self.seg.pcd_fit_sphere(layer='bot', method='ls')
        self.bot_smooth_pcd = self.seg.get_bot_smooth_pcd()
        self.bot_smooth_pcd.paint_uniform_color([0, 1, 0])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(
            self.top_smooth_pcd.points).shape[0], axis=0), layer="top")
        self.seg.ray_tracing(self.seg.refracts_top, layer='bot')
        mesh_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                              origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [self.top_smooth_pcd, self.bot_smooth_pcd, mesh_frame],
            window_name="smooth fit a sphere on bottom layer",
            point_show_normal=False)
        self.positions_nd, self.values_nd, self.values_gr = None, None, None
        self.nninter, self.gridinter = None, None
        self.lowest_layer = np.zeros((((self.xdim // self.dp_x) + 1) *
                                      ((self.ydim // self.dp_y) + 1), 3))
        self.left_layer = np.zeros((((self.ydim // self.dp_y) + 1) *
                                    (self.zdim // self.dp_z), 3))
        self.right_layer = np.zeros((((self.ydim // self.dp_y) + 1) *
                                     (self.zdim // self.dp_z), 3))

    def grid_inter_pairs(self, imgs_dir):
        values_gr = np.zeros((self.xdim, self.ydim, self.zdim))
        print("Building Values Matrix")
        for i in range(self.idx_range[0], self.idx_range[1] + 1):
            img = cv.imread(imgs_dir + "0_" + str(i) + "_bscan.png",
                            cv.IMREAD_GRAYSCALE)
            values_gr[:, i-self.idx_range[0], :] = img.transpose()
        self.values_gr = values_gr
        x = np.linspace(0, self.xlength, self.xdim, endpoint=False)
        y = np.linspace(0, self.ylength, self.ydim, endpoint=False)
        z = np.linspace(0, self.zlength, self.zdim, endpoint=False)
        self.gridinter = \
            scipy.interpolate.RegularGridInterpolator((x, y, z),
                                                      self.values_gr,
                                                      method='linear',
                                                      bounds_error=False,
                                                      fill_value=255)
        print("Interpolator Built.")

    def nn_inter_pairs(self):
        positions = np.zeros(((self.xdim // self.dp_x) + 1,
                              (self.ydim // self.dp_y) + 1,
                              self.zdim // self.dp_z, 3))
        values = np.zeros(((self.xdim // self.dp_x) + 1,
                           (self.ydim // self.dp_y) + 1,
                           self.zdim // self.dp_z, 3))
        record_count = 0
        print("Building interpolation matrix.")
        idx_x, idx_y = 0, 0
        top_smooth_points = np.asarray(self.top_smooth_pcd.points)
        bot_smooth_points = np.asarray(self.bot_smooth_pcd.points)
        bot_points = np.asarray(self.seg.bot_points_mm)
        for i in tqdm(range(top_smooth_points.shape[0])):
            if idx_y == (self.ydim // self.dp_y) + 1:
                break
            if ((i+1)//(self.xdim+1)) % self.dp_y != 0:
                continue
            if (i % self.xdim) % self.dp_x != 0 and (i + 1) % self.xdim != 0:
                continue
            top_refract = self.seg.refracts_top[i]
            bot_refract = self.seg.refracts_bot[i]
            positions[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            z_positions = np.linspace(0, self.zlength, self.zdim // self.dp_z,
                                      endpoint=False)
            positions[idx_x, idx_y, :, 2] = z_positions
            top_point = np.argwhere(z_positions >= top_smooth_points[i][2])
            bot_point = np.argwhere(z_positions >= bot_points[i][2])
            values[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            values[idx_x, idx_y, :top_point[0, 0], 2] = \
                np.linspace(0, top_smooth_points[i][2], top_point[0, 0])
            values[idx_x, idx_y, top_point[0, 0]:, :] = \
                self.refract_correction_top(top_refract, top_smooth_points[i],
                                            top_smooth_points[i],
                                            positions[idx_x, idx_y,
                                            top_point[0, 0]:, :],
                                            self.n2)
            values[idx_x, idx_y, bot_point[0, 0]:, :] = \
                self.refract_correction_bot(bot_refract,
                                            bot_smooth_points[i],
                                            values[idx_x, idx_y,
                                            bot_point[0, 0]:, :],
                                            self.n3 / self.n2)
            self.lowest_layer[record_count, :] = values[idx_x, idx_y, -1, :]
            if idx_x == 0:
                self.left_layer[(record_count * self.zdim): ((record_count +
                1) * self.zdim), :] = values[idx_x, idx_y, :, :]
            idx_x += 1
            if idx_x == self.xdim//self.dp_x + 1:
                self.right_layer[(record_count * self.zdim): ((record_count +
                1) * self.zdim), :] = values[idx_x - 1, idx_y, :, :]
                idx_y += 1
                idx_x = 0
                record_count += 1
        self.positions_nd, self.values_nd = positions.reshape(
            (-1, 3)), values.reshape((-1, 3))
        self.nninter = \
            scipy.interpolate.LinearNDInterpolator(self.values_nd,
                                                   self.positions_nd,
                                                   fill_value=-1.)
        print("Interpolator Built.")

    def refract_correction_top(self, refract, dis_origin, origin, points,
                               group_idx):
        raw_distance = np.absolute(dis_origin[2] - points[:, 2])
        distance = raw_distance / group_idx
        corrected_points = origin + \
                           (distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def refract_correction_bot(self, refract, origin, points, group_idx):
        raw_distance = np.absolute(np.linalg.norm(points - origin, axis=1))
        distance = raw_distance / group_idx
        corrected_points = origin + \
                           (distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def bot_convexhull_check(self, pos):
        xy_diff_arr = np.subtract(self.lowest_layer[:, :2], pos[:2])
        xy_dis_arr = np.linalg.norm(xy_diff_arr, axis=1)
        small_3_idx = np.argpartition(xy_dis_arr, 3)
        if pos[2] - self.lowest_layer[small_3_idx[0]][2] > 0 or \
           pos[2] - self.lowest_layer[small_3_idx[1]][2] > 0 or \
           pos[2] - self.lowest_layer[small_3_idx[2]][2] > 0:
            return False
        else:
            return True

    def reconstruction(self, y_idx):
        img = np.full((self.xdim, self.zdim), 255, dtype=np.uint8)
        print("Reconstructing image")
        for i in tqdm(range(0, self.xdim)):
            for j in range(0, self.zdim):
                pos = np.multiply([i, y_idx, j], [self.xlength/self.xdim,
                                                  self.ylength/self.ydim,
                                                  self.zlength/self.zdim])
                if self.bot_convexhull_check(pos):
                    img[i][j] = np.uint8(self.gridinter(self.nninter(pos)))
        img = cv.cvtColor(img.transpose(), cv.COLOR_GRAY2RGB)
        top_seg = np.array(pd.read_csv(self.directory + "result_top_" +
                                       str(y_idx+200) + ".csv", header=None))
        bot_seg = np.array(pd.read_csv(self.directory + "result_bot_" +
                                       str(y_idx+200) + ".csv", header=None))
        for i in range(top_seg.shape[0]):
            img[top_seg[i][1]][top_seg[i][0]] = np.array([255, 0, 0])
        for i in range(bot_seg.shape[0]):
            img[bot_seg[i][1]][bot_seg[i][0]] = np.array([0, 255, 0])
        return img


if __name__ == "__main__":
    inter = Interpolation("../data/seg_res/seg_res_bss/",
                          [200, 600], 416, 401, 577, 5.81, 5.0,
                          3.13, 1.0003, 1.4815, 1.3432, 10, 10, 1)
    inter.nn_inter_pairs()
    inter.grid_inter_pairs('../data/images/bss_760_crop/')
    img = inter.reconstruction(140)
    plt.imshow(img)
    plt.show()

    print("Program Finished.")
