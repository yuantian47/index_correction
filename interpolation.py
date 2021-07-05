import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import scipy.interpolate
import scipy.linalg
import cv2 as cv
import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import csv_pcd


def points_dis2plane(a, b, c, d, pts):
    dis = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / np.sqrt(
        np.power(a, 2) + np.power(b, 2) + np.power(c, 2))
    return dis


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


class Interpolation:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength,
                 ylength, zlength, n1, n2, n3, dp_x, dp_y, dp_z, group_idx):
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.dp_x, self.dp_y, self.dp_z = dp_x, dp_y, dp_z
        self.directory = directory
        self.idx_range = idx_range
        self.group_idx = group_idx
        self.seg = csv_pcd.RealPCD(directory, idx_range, xdim, ydim, zdim,
                                   xlength, ylength, zlength, n1, n2, n3,
                                   self.group_idx)
        self.seg.remove_outlier(layer='top', neighbors=100)
        self.seg.pcd_fit_sphere(method='ls')
        tmp_pcd = self.seg.get_top_smooth_pcd()
        tmp_pcd.paint_uniform_color([0, 1, 0])
        self.seg.pcd_fit_spline(layer='top')
        self.top_smooth_pcd = self.seg.get_top_smooth_pcd()
        self.top_smooth_pcd.paint_uniform_color([1, 0, 0])
        mean_dis = np.mean(np.abs(np.asarray(self.top_smooth_pcd.points) -
                                  np.asarray(tmp_pcd.points)), axis=0)
        print("Anterior mean distance is :", mean_dis)
        mesh_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                              origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.top_smooth_pcd, tmp_pcd,
                                           mesh_frame])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(
            self.top_smooth_pcd.points).shape[0], axis=0))
        self.seg.refraction_correction()
        # self.seg.remove_outlier(layer='corrected_bot', neighbors=100)
        self.seg.pcd_fit_sphere(layer='bot', method='ls')
        tmp_pcd = self.seg.get_bot_smooth_pcd()
        tmp_pcd.paint_uniform_color([0, 0, 1])
        self.seg.pcd_fit_spline(layer='bot')
        self.bot_smooth_pcd = self.seg.get_bot_smooth_pcd()
        self.bot_smooth_pcd.paint_uniform_color([0, 1, 0])
        mean_dis = np.mean(np.abs(np.asarray(self.bot_smooth_pcd.points) -
                                  np.asarray(tmp_pcd.points)), axis=0)
        print("Posterior mean distance is:", mean_dis)
        o3d.visualization.draw_geometries([self.bot_smooth_pcd, tmp_pcd,
                                           mesh_frame])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(
            self.top_smooth_pcd.points).shape[0], axis=0), layer="top")
        self.seg.ray_tracing(self.seg.refracts_top, layer='bot')
        mesh_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                              origin=[0, 0, 0])
        self.seg.tar_pcd.points = o3d.utility.Vector3dVector(
            self.target_correction())
        self.seg.tar_pcd.paint_uniform_color([0, 0, 1])
        self.seg.emp_pcd.paint_uniform_color([0.5, 0.5, 0])
        self.seg.remove_outlier('tar')
        # self.seg.remove_outlier('emp')
        o3d.visualization.draw_geometries(
            [self.top_smooth_pcd, self.bot_smooth_pcd, self.seg.tar_pcd,
             self.seg.emp_pcd, mesh_frame],
            window_name="index correction result",
            point_show_normal=False)
        self.positions_nd, self.values_nd, self.values_gr = None, None, None
        self.nninter, self.gridinter = None, None
        self.lowest_layer = np.zeros(((self.xdim // self.dp_x) *
                                      (self.ydim // self.dp_y), 3))
        self.left_layer = np.zeros(((self.ydim // self.dp_y) *
                                    (self.zdim // self.dp_z), 3))
        self.right_layer = np.zeros(((self.ydim // self.dp_y) *
                                     (self.zdim // self.dp_z), 3))

    def svd_fit_plane(self):
        tar_points = np.asarray(self.seg.tar_pcd.points)
        emp_points = np.asarray(self.seg.emp_pcd.points)
        tar_mean = np.mean(tar_points, axis=0)
        emp_mean = np.mean(emp_points, axis=0)
        cov_tar = np.cov(tar_points - tar_mean, rowvar=False)
        cov_emp = np.cov(emp_points - emp_mean, rowvar=False)
        tar_w, tar_v = np.linalg.eigh(cov_tar, UPLO='U')
        tar_normal = tar_v[np.argmin(tar_w)]
        emp_w, emp_v = np.linalg.eigh(cov_emp, UPLO='U')
        emp_normal = emp_v[np.argmin(emp_w)]
        d_tar = -1 * (tar_normal[0] * tar_mean[0] +
                      tar_normal[1] * tar_mean[1] +
                      tar_normal[2] * tar_mean[2])
        d_emp = -1 * (emp_normal[0] * emp_mean[0] +
                      emp_normal[1] * emp_mean[1] +
                      emp_normal[2] * emp_mean[2])
        fit_tar_pcd = o3d.geometry.PointCloud()
        fit_tar_points = np.copy(tar_points)
        fit_tar_points[:, 2] = -1 * (tar_normal[0] * tar_points[:, 0] +
                                     tar_normal[1] * tar_points[:, 1] +
                                     d_tar) / tar_normal[2]
        fit_tar_pcd.points = o3d.utility.Vector3dVector(fit_tar_points)
        fit_tar_pcd.paint_uniform_color([0, 0, 1])
        fit_emp_pcd = o3d.geometry.PointCloud()
        fit_emp_points = np.copy(emp_points)
        fit_emp_points[:, 2] = -1 * (emp_normal[0] * emp_points[:, 0] +
                                     emp_normal[1] * emp_points[:, 1] +
                                     d_emp) / emp_normal[2]
        fit_emp_pcd.points = o3d.utility.Vector3dVector(fit_emp_points)
        fit_emp_pcd.paint_uniform_color([0.5, 0.5, 0])
        mesh_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                              origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [self.top_smooth_pcd, self.bot_smooth_pcd, fit_tar_pcd,
             fit_emp_pcd, mesh_frame],
            window_name="fit planes on corrected air target and empty target",
            point_show_normal=False)
        dis_tar = points_dis2plane(tar_normal[0], tar_normal[1],
                                   tar_normal[2], d_tar, tar_points)
        dis_emp = points_dis2plane(emp_normal[0], emp_normal[1],
                                   emp_normal[2], d_emp, emp_points)
        print("BSS target fitting error mean is {:.4f}, std is {:.4f}".format(
            np.mean(dis_tar), np.std(dis_tar)))
        print("Empty target fitting error mean is {:.4f}, std is {:.4f}".
              format(np.mean(dis_emp), np.std(dis_emp)))
        if tar_normal[2] * emp_normal[2] < 0:
            emp_normal *= -1
        normal_diff = np.arccos(np.dot(tar_normal, emp_normal) /
                                (np.linalg.norm(tar_normal) *
                                 np.linalg.norm(emp_normal)))
        print("The normal difference is {:.4f}".format(normal_diff))
        print("The mean difference of two raw point clouds is {:.4f}".format(
            tar_mean[2] - emp_mean[2]))
        return normal_diff, tar_mean[2] - emp_mean[2]

    def icp_registration(self):
        trans_init = np.eye(4)
        reg_p2p = o3d.registration.registration_icp(self.seg.tar_pcd,
                                                    self.seg.emp_pcd,
                                                    1.0,
                                                    trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(max_iteration=2000))
        draw_registration_result(self.seg.tar_pcd, self.seg.emp_pcd,
                                 reg_p2p.transformation)
        return reg_p2p

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

    def nn_inter_pairs_pro(self):
        positions_raw = np.zeros((self.xdim, self.ydim, self.zdim, 3))
        top_refract_raw = np.zeros((self.xdim, self.ydim, 3))
        bot_refract_raw = np.zeros((self.xdim, self.ydim, 3))
        top_smooth_raw = np.zeros((self.xdim, self.ydim, 3))
        bot_smooth_raw = np.zeros((self.xdim, self.ydim, 3))
        bot_points_raw = np.zeros((self.xdim, self.ydim, 3))
        z_positions = np.linspace(0, self.zlength, self.zdim, endpoint=False)
        positions_raw[:, :, :, 2] = z_positions
        top_smooth_points = np.asarray(self.top_smooth_pcd.points)
        bot_smooth_points = np.asarray(self.bot_smooth_pcd.points)
        bot_points = np.asarray(self.seg.bot_points_mm)
        ydim_idx = 0
        for i in tqdm(range(0, top_smooth_points.shape[0], self.xdim)):
            for z in range(0, self.zdim):
                positions_raw[:, ydim_idx, z, :2] = \
                    top_smooth_points[i: i+self.xdim, :2]
            top_refract_raw[:, ydim_idx, :] = \
                self.seg.refracts_top[i: i+self.xdim]
            bot_refract_raw[:, ydim_idx, :] = \
                self.seg.refracts_bot[i: i+self.xdim]
            top_smooth_raw[:, ydim_idx, :] = \
                top_smooth_points[i: i+self.xdim, :]
            bot_smooth_raw[:, ydim_idx, :] = \
                bot_smooth_points[i: i + self.xdim, :]
            bot_points_raw[:, ydim_idx, :] = bot_points[i: i + self.xdim, :]
            ydim_idx += 1
        positions = positions_raw[::self.dp_x, ::self.dp_y, ::self.dp_z]
        top_refract = top_refract_raw[::self.dp_x, ::self.dp_y, :]
        bot_refract = bot_refract_raw[::self.dp_x, ::self.dp_y, :]
        top_smooth_dp = top_smooth_raw[::self.dp_x, ::self.dp_y, :]
        bot_smooth_dp = bot_smooth_raw[::self.dp_x, ::self.dp_y, :]
        bot_points_dp = bot_points_raw[::self.dp_x, ::self.dp_y, :]
        values = np.zeros_like(positions)
        for i in tqdm(range(positions.shape[0])):
            for j in range(positions.shape[1]):
                top_point = np.argwhere(z_positions >= top_smooth_dp[i][j][2])
                bot_point = np.argwhere(z_positions >= bot_points_dp[i][j][2])
                values[i, j, :, :2] = top_smooth_dp[i, j, :2]
                values[i, j, :top_point[0, 0], 2] = \
                    np.linspace(0, top_smooth_dp[i][j][2], top_point[0, 0])
                values[i, j, top_point[0, 0]:, :] = \
                    self.refract_correction_top(top_refract[i, j],
                                                top_smooth_dp[i, j, :],
                                                top_smooth_dp[i, j, :],
                                                positions[i, j,
                                                top_point[0, 0]:, :],
                                                self.n2 / self.n1)
                values[i, j, bot_point[0, 0]:, :] = \
                    self.refract_correction_bot(bot_refract[i, j],
                                                bot_smooth_dp[i, j, :],
                                                values[i, j,
                                                bot_point[0, 0]:, :],
                                                self.n3 / self.n2)
        self.lowest_layer = np.asarray(values[:, :, -1, :]).reshape((-1, 3))
        self.left_layer = np.asarray(values[0, :, :, :]).reshape((-1, 3))
        self.right_layer = np.asarray(values[-1, :, :, :]).reshape((-1, 3))
        self.positions_nd, self.values_nd = positions.reshape(
            (-1, 3)), values.reshape((-1, 3))
        self.nninter = \
            scipy.interpolate.LinearNDInterpolator(self.values_nd,
                                                   self.positions_nd,
                                                   fill_value=-1.)
        print("Interpolator Built.")
        return values, bot_smooth_dp, top_smooth_dp

    def refract_correction_top(self, refract, dis_origin, origin, points,
                               group_idx):
        raw_distance = np.absolute(dis_origin[2] - points[:, 2])
        distance = raw_distance / group_idx
        corrected_points = origin + \
                           (distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def refract_correction_bot(self, refract, origin, points, group_idx):
        raw_distance = np.linalg.norm(points - origin, axis=1)
        distance = raw_distance / group_idx
        corrected_points = origin + \
                           (distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points

    def target_correction(self):
        raw_distance_top = np.linalg.norm(self.seg.tar_points_mm -
                                          np.asarray(
                                              self.seg.top_smooth_pcd.points),
                                          axis=1)
        distance_top = raw_distance_top / (self.n2 / self.n1)
        corrected_top = np.asarray(self.seg.top_smooth_pcd.points) + \
                        (distance_top.reshape(distance_top.shape[0], 1) *
                         self.seg.refracts_top)
        bot_points_mm = np.asarray(self.seg.bot_smooth_pcd.points)
        raw_distance_bot = np.linalg.norm(corrected_top - bot_points_mm,
                                          axis=1)
        distance_bot = raw_distance_bot / (self.n3 / self.n2)
        corrected_bot = bot_points_mm + \
                        (distance_bot.reshape(distance_bot.shape[0], 1) *
                         self.seg.refracts_bot)
        return corrected_bot

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

    def left_convexhull_check(self, pos):
        yz_diff_arr = np.subtract(self.left_layer[:, 1:], pos[1:])
        yz_dis_arr = np.linalg.norm(yz_diff_arr, axis=1)
        small_3_idx = np.argpartition(yz_dis_arr, 3)
        if pos[0] - self.left_layer[small_3_idx[0]][0] < 0 or \
           pos[0] - self.left_layer[small_3_idx[1]][0] < 0 or \
           pos[0] - self.left_layer[small_3_idx[2]][0] < 0:
            return False
        else:
            return True

    def right_convexhull_check(self, pos):
        yz_diff_arr = np.subtract(self.right_layer[:, 1:], pos[1:])
        yz_dis_arr = np.linalg.norm(yz_diff_arr, axis=1)
        small_3_idx = np.argpartition(yz_dis_arr, 3)
        if pos[0] - self.right_layer[small_3_idx[0]][0] > 0 or \
           pos[0] - self.right_layer[small_3_idx[1]][0] > 0 or \
           pos[0] - self.right_layer[small_3_idx[2]][0] > 0:
            return False
        else:
            return True

    def reconstruction(self, y_idx, x_padding=10):
        img = np.full((self.xdim + 2 * x_padding,
                       self.zdim), 255, dtype=np.uint8)
        print("Reconstructing image")
        for i in tqdm(range(-x_padding, self.xdim+x_padding)):
            for j in range(0, self.zdim):
                pos = np.multiply([i, y_idx, j], [self.xlength/self.xdim,
                                                  self.ylength/self.ydim,
                                                  self.zlength/self.zdim])
                if self.bot_convexhull_check(pos) and \
                        self.left_convexhull_check(pos) and \
                        self.right_convexhull_check(pos):
                    img[i + x_padding][j] =\
                        np.uint8(self.gridinter(self.nninter(pos)))
        img = cv.cvtColor(img.transpose(), cv.COLOR_GRAY2RGB)
        return img


if __name__ == "__main__":

    air_normal_diffs = np.zeros(5)
    air_tar_dists = np.zeros(5)

    for i in range(1, 6):
        if i == 3:
            inter = Interpolation(
                "../data/seg_res/" + str(i) + "_c/water_seg_res",
                [210, 610], 500, 401, 877, 7.022, 5.0125,
                4.701, 1.0003, 1.369, 1.340, 10, 10, 1, i)
        else:
            inter = Interpolation(
                "../data/seg_res/" + str(i) + "_c/water_seg_res",
                [200, 600], 500, 401, 877, 7.022, 5.0125,
                4.701, 1.0003, 1.369, 1.340, 10, 10, 1, i)
        normal_diff, tar_dist = inter.svd_fit_plane()
        air_normal_diffs[i-1], air_tar_dists[i-1] = normal_diff, tar_dist
        print("\n ******************** \n")
        # values, bot, top = inter.nn_inter_pairs_pro()
        # plt.scatter(values[:, 0, :, 0], values[:, 0, :, 2] * -1)
        # plt.scatter(bot[:, 0, 0], bot[:, 0, 2] * -1)
        # plt.scatter(top[:, 0, 0], top[:, 0, 2] * -1)
        # plt.show()
        # inter.grid_inter_pairs(
        #     "../data/seg_res/" + str(i) + "_c/" + str(i) + "_water_crop/")
        # img = inter.reconstruction(200)
        # plt.imsave("test.png", img)
        # plt.imshow(img)
        # plt.show()

    print("Air Normal: ", np.mean(air_normal_diffs),
          "+-", np.std(air_normal_diffs), air_normal_diffs)
    print("Air Dists: ", np.mean(air_tar_dists), "+-",
          np.std(air_tar_dists), air_tar_dists)

    print("Program Finished.")
