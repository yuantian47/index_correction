import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import padasip as pa
from scipy import interpolate

import normal_calculation as nc


# reference: Open3D Point Cloud Outlier Removal
def display_inlier_outlier(pcd, ind):
    inlier_pcd = pcd.select_by_index(ind)
    outlier_pcd = pcd.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_pcd.paint_uniform_color([1, 0, 0])
    inlier_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd, mesh_frame])


def downsample_compare(raw_pcd, dp_rate):
    pcd_dp = raw_pcd.uniform_down_sample(dp_rate)
    # pcd_dp, unknown_arr, unknow_list = raw_pcd.voxel_down_sample_and_trace(0.02, raw_pcd.get_min_bound(), raw_pcd.get_max_bound())
    pcd_dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(100 / dp_rate)),
                            fast_normal_computation=False)
    pcd_dp.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
    pcd_dp.normalize_normals()
    raw_normals = np.asarray(raw_pcd.normals)[::dp_rate]
    dp_normals = np.asarray(pcd_dp.normals)
    if raw_normals.shape != dp_normals.shape:
        raise ValueError("Two normals' arrays have different shape!")
    angle_list = []
    for i in range(raw_normals.shape[0]):
        angle_list.append(np.dot(raw_normals[i], dp_normals[i]) /
                          (np.linalg.norm(raw_normals[i]) * np.linalg.norm(dp_normals[i])))
    angle_array = np.arccos(np.around(angle_list, 5))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh_frame, pcd_dp],
                                      "Down Sampled Point Cloud with rate " + str(dp_rate),
                                      point_show_normal=True)
    return angle_array


class RealPCD:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength, n1, n2, n3):
        self.directory = directory
        self.idx_range = idx_range
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.top_points, self.bot_points = np.zeros((xdim * ydim, 3)), np.zeros((xdim * ydim, 3))
        self.tar_points = np.zeros((xdim * ydim, 3))
        self.emp_points = np.zeros((xdim * ydim, 3))
        self.top_points_mm, self.bot_points_mm = np.zeros((xdim * ydim, 3)), np.zeros((xdim * ydim, 3))
        self.tar_points_mm = np.zeros((xdim * ydim, 3))
        self.emp_points_mm = np.zeros((xdim * ydim, 3))
        self.top_pcd, self.bot_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        self.tar_pcd = o3d.geometry.PointCloud()
        self.emp_pcd = o3d.geometry.PointCloud()
        self.top_smooth_pcd, self.bot_smooth_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        self.corrected_bot_pcd = o3d.geometry.PointCloud()
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.refracts_top = np.zeros((self.xdim * self.ydim, 3))
        self.refracts_raw_top = None
        self.refracts_bot = np.zeros((self.xdim * self.ydim, 3))
        self.top_ind, self.bot_ind, self.corr_bot_ind = None, None, None
        point_idx = 0
        for i in tqdm(range(self.idx_range[0], self.idx_range[1] + 1)):
            top_seg_raw_up = np.array(pd.read_csv(self.directory +
                                                  "/result_top_up_" + str(i) +
                                                  ".csv", header=None))
            bot_seg_raw_up = np.array(pd.read_csv(self.directory +
                                                  "/result_bot_up_" + str(i) +
                                                  ".csv", header=None))
            top_seg_raw_dn = np.array(pd.read_csv(self.directory +
                                                  "/result_top_dn_" + str(i) +
                                                  ".csv", header=None))
            bot_seg_raw_dn = np.array(pd.read_csv(self.directory +
                                                  "/result_bot_dn_" + str(i) +
                                                  ".csv", header=None))
            tar_seg_raw = np.array(pd.read_csv(self.directory +
                                               "/result_tar_" +
                                               str(i) + ".csv",
                                               header=None))
            emp_seg_raw = np.array(pd.read_csv(
                "../data/seg_res/2/tar_seg_res/result_top_" + str(i)
                + ".csv", header=None) + np.array([0, 200]))
            top_seg_up, bot_seg_up = np.zeros((xdim, 3)), np.zeros((xdim, 3))
            top_seg_dn, bot_seg_dn = np.zeros((xdim, 3)), np.zeros((xdim, 3))
            top_seg, bot_seg = np.zeros((xdim, 3)), np.zeros((xdim, 3))
            tar_seg = np.zeros((xdim, 3))
            emp_seg = np.zeros((xdim, 3))
            for j in range(xdim):
                same_x_top_up = top_seg_raw_up[list([*np.where(
                    top_seg_raw_up[:, 0] == j)[0]])]
                top_seg_up[j] = np.insert(same_x_top_up[np.argmax(
                    same_x_top_up[:, 1])], 1, i-self.idx_range[0])
                same_x_bot_up = bot_seg_raw_up[list([*np.where(
                    bot_seg_raw_up[:, 0] == j)[0]])]
                bot_seg_up[j] = np.insert(same_x_bot_up[np.argmax(
                    same_x_bot_up[:, 1])], 1, i-self.idx_range[0])
                same_x_top_dn = top_seg_raw_dn[list([*np.where(
                    top_seg_raw_dn[:, 0] == j)[0]])]
                top_seg_dn[j] = np.insert(same_x_top_dn[np.argmin(
                    same_x_top_dn[:, 1])], 1, i - self.idx_range[0])
                same_x_bot_dn = bot_seg_raw_dn[list([*np.where(
                    bot_seg_raw_dn[:, 0] == j)[0]])]
                bot_seg_dn[j] = np.insert(same_x_bot_dn[np.argmin(
                    same_x_bot_dn[:, 1])], 1, i - self.idx_range[0])
                top_seg[j] = top_seg_up[j]
                bot_seg[j] = bot_seg_up[j]
                if top_seg_up[j][2] < top_seg_dn[j][2]:
                    top_seg[j][2] = float(top_seg_up[j][2] + top_seg_dn[j][2])\
                                    / 2
                else:
                    top_seg[j][2] = float(top_seg_up[j][2] - 3)
                if bot_seg_up[j][2] < bot_seg_dn[j][2]:
                    bot_seg[j][2] = float(bot_seg_up[j][2] + bot_seg_dn[j][2])\
                                    / 2
                else:
                    bot_seg[j][2] = float(bot_seg_dn[j][2] - 3)
                same_x_tar = tar_seg_raw[list([*np.where(tar_seg_raw[:,
                                                         0] == j)[0]])]
                tar_seg[j] = np.insert(same_x_tar[np.argmax(same_x_tar[:, 1])],
                                       1, i-self.idx_range[0])
                same_x_emp = emp_seg_raw[list([*np.where(emp_seg_raw[:,
                                                         0] == j)[0]])]
                emp_seg[j] = np.insert(same_x_emp[np.argmax(same_x_emp[:, 1])],
                                       1, i - self.idx_range[0])
            top_seg_mm = np.multiply(top_seg,
                                     [float(self.xlength) / self.xdim,
                                      float(self.ylength) / self.ydim,
                                      float(self.zlength) / self.zdim])
            bot_seg_mm = np.multiply(bot_seg,
                                     [float(self.xlength) / self.xdim,
                                      float(self.ylength) / self.ydim,
                                      float(self.zlength) / self.zdim])
            tar_seg_mm = np.multiply(tar_seg,
                                     [float(self.xlength) / self.xdim,
                                      float(self.ylength) / self.ydim,
                                      float(self.zlength) / self.zdim])
            emp_seg_mm = np.multiply(emp_seg,
                                     [float(self.xlength) / self.xdim,
                                      float(self.ylength) / self.ydim,
                                      float(self.zlength) / self.zdim])
            self.top_points[point_idx:point_idx + xdim] = np.asarray(
                top_seg, dtype=np.int)
            self.bot_points[point_idx:point_idx + xdim] = np.asarray(
                bot_seg, dtype=np.int)
            self.tar_points[point_idx:point_idx + xdim] = tar_seg
            self.emp_points[point_idx:point_idx + xdim] = emp_seg
            self.top_points_mm[point_idx:point_idx + xdim], self.bot_points_mm[point_idx:point_idx + xdim] = \
                top_seg_mm, bot_seg_mm
            self.tar_points_mm[point_idx:point_idx + xdim] = tar_seg_mm
            self.emp_points_mm[point_idx:point_idx + xdim] = emp_seg_mm
            point_idx += xdim
        self.top_pcd.points = o3d.utility.Vector3dVector(self.top_points_mm)
        self.bot_pcd.points = o3d.utility.Vector3dVector(self.bot_points_mm)
        self.tar_pcd.points = o3d.utility.Vector3dVector(self.tar_points_mm)
        self.emp_pcd.points = o3d.utility.Vector3dVector(self.emp_points_mm)

    def edit_pcd(self, layer='top'):
        if layer == 'top':
            o3d.visualization.draw_geometries_with_editing([self.top_pcd], 'Edit Top Point Cloud')
            self.top_pcd = o3d.io.read_point_cloud('../data/pcd/top_pcd.ply')
        elif layer == 'bot':
            o3d.visualization.draw_geometries_with_editing([self.bot_pcd], 'Edit Bottom Point Cloud')
            self.bot_pcd = o3d.io.read_point_cloud('../data/pcd/bot_pcd.ply')
        else:
            raise ValueError("Please input vaild layer's name.")

    def remove_outlier(self, layer='top', method='statistical', neighbors=100, std_ratio=0.5, radius=0.1):
        if layer == 'top':
            if method == 'statistical':
                cl, self.top_ind = self.top_pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
            elif method == 'radius':
                cl, self.top_ind = self.top_pcd.remove_radius_outlier(nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.top_pcd, self.top_ind)
            self.top_pcd = self.top_pcd.select_by_index(self.top_ind)
        elif layer == 'corrected_bot':
            if method == 'statistical':
                cl, self.corr_bot_ind = self.corrected_bot_pcd.remove_statistical_outlier(nb_neighbors=neighbors,
                                                                                          std_ratio=std_ratio)
            elif method == 'radius':
                cl, self.corr_bot_ind = self.corrected_bot_pcd.remove_radius_outlier(nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.corrected_bot_pcd, self.corr_bot_ind)
            self.corrected_bot_pcd = self.corrected_bot_pcd.select_by_index(self.corr_bot_ind)
        elif layer == 'bot':
            if method == 'statistical':
                cl, self.bot_ind = self.bot_pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
            elif method == 'radius':
                cl, self.bot_ind = self.bot_pcd.remove_radius_outlier(nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.bot_pcd, self.bot_ind)
            self.bot_pcd = self.bot_pcd.select_by_index(self.bot_ind)
        elif layer == 'tar':
            if method == 'statistical':
                cl, self.tar_ind = self.tar_pcd.remove_statistical_outlier(
                    nb_neighbors=neighbors, std_ratio=std_ratio)
            elif method == 'radius':
                cl, self.tar_ind = self.tar_pcd.remove_radius_outlier(
                    nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.tar_pcd, self.tar_ind)
            self.tar_pcd = self.tar_pcd.select_by_index(self.tar_ind)
        elif layer == 'emp':
            if method == 'statistical':
                cl, self.emp_ind = self.emp_pcd.remove_statistical_outlier(
                    nb_neighbors=neighbors, std_ratio=std_ratio)
            elif method == 'radius':
                cl, self.emp_ind = self.emp_pcd.remove_radius_outlier(
                    nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.emp_pcd, self.emp_ind)
            self.emp_pcd = self.emp_pcd.select_by_index(self.emp_ind)
        else:
            raise ValueError("Please input valid layer's name.")

    def get_top_pcd(self):
        return self.top_pcd

    def get_top_smooth_pcd(self):
        return self.top_smooth_pcd

    def get_bot_pcd(self):
        return self.bot_pcd

    def get_corrected_bot_pcd(self):
        return self.corrected_bot_pcd

    def get_bot_smooth_pcd(self):
        return self.bot_smooth_pcd

    def pcd_fit_sphere(self, layer='top', method='lms'):
        if method == 'ls':
            if layer == 'top':
                top_points_mm_s = np.array(np.asarray(self.top_pcd.points),
                                           copy=True)
                co_mat = np.zeros((top_points_mm_s.shape[0], 4))
                co_mat[:, 0] = top_points_mm_s[:, 0] * 2
                co_mat[:, 1] = top_points_mm_s[:, 1] * 2
                co_mat[:, 2] = top_points_mm_s[:, 2] * 2
                co_mat[:, 3] = 1
                ordinate = np.zeros((top_points_mm_s.shape[0], 1))
                ordinate[:, 0] = np.sum(np.power(top_points_mm_s, 2), axis=1)
                res, err, _, _ = np.linalg.lstsq(co_mat, ordinate, rcond=None)
                print("The error is:", err)
                rad = np.sqrt(res[0] * res[0] +
                              res[1] * res[1] +
                              res[2] * res[2] + res[3])
                print("The radius is: {}".format(rad))
                top_points_mm_s = np.array(np.asarray(self.top_points_mm),
                                           copy=True)
                for i in range(top_points_mm_s.shape[0]):
                    top_points_mm_s[i, 2] = \
                        -np.sqrt(np.power(rad, 2) -
                                 np.power(top_points_mm_s[i, 0] - res[0], 2) -
                                 np.power(top_points_mm_s[i, 1] - res[1],
                                          2)) + res[2]
                self.top_smooth_pcd = o3d.geometry.PointCloud()
                self.top_smooth_pcd.points =\
                    o3d.utility.Vector3dVector(top_points_mm_s)
                self.top_smooth_pcd.normals =\
                    self.sphere_normal(top_points_mm_s, res[:-1].T[0])
                self.top_smooth_pcd.normalize_normals()
            elif layer == 'bot':
                bot_points_mm_s = np.array(np.asarray(self.corrected_bot_pcd.points), copy=True)
                co_mat = np.zeros((bot_points_mm_s.shape[0], 4))
                co_mat[:, 0] = bot_points_mm_s[:, 0] * 2
                co_mat[:, 1] = bot_points_mm_s[:, 1] * 2
                co_mat[:, 2] = bot_points_mm_s[:, 2] * 2
                co_mat[:, 3] = 1
                ordinate = np.zeros((bot_points_mm_s.shape[0], 1))
                ordinate[:, 0] = np.sum(np.power(bot_points_mm_s, 2), axis=1)
                res, err, _, _ = np.linalg.lstsq(co_mat, ordinate, rcond=None)
                print("The error is:", err)
                rad = np.sqrt(res[0] * res[0] +
                              res[1] * res[1] + res[2] * res[2] + res[3])
                print("The radius is: {}".format(rad))
                bot_points_mm_s = np.array(np.asarray(self.bot_points_mm), copy=True)
                for i in range(bot_points_mm_s.shape[0]):
                    bot_points_mm_s[i, 2] = -np.sqrt(np.power(rad, 2) -
                                                     np.power(bot_points_mm_s[i, 0] - res[0], 2) -
                                                     np.power(bot_points_mm_s[i, 1] - res[1], 2)) + res[2]
                self.bot_smooth_pcd = o3d.geometry.PointCloud()
                self.bot_smooth_pcd.points =\
                    o3d.utility.Vector3dVector(bot_points_mm_s)
                self.bot_smooth_pcd.normals =\
                    self.sphere_normal(bot_points_mm_s, res[:-1].T[0])
                self.bot_smooth_pcd.normalize_normals()
        if method == 'lms':
            if layer == 'top':
                top_points_mm_s = np.array(np.asarray(self.top_pcd.points), copy=True)
                desired_vec = np.zeros((top_points_mm_s.shape[0], 1))
                desired_vec[:, 0] = np.sum(np.power(top_points_mm_s, 2), axis=1)
                input_mat = np.zeros((top_points_mm_s.shape[0], 4))
                input_mat[:, 0] = top_points_mm_s[:, 0] * 2
                input_mat[:, 1] = top_points_mm_s[:, 1] * 2
                input_mat[:, 2] = top_points_mm_s[:, 2] * 2
                input_mat[:, 3] = 1
                filter = pa.filters.FilterLMS(n=4, mu=0.005, w="random")
                _, _, res_arr = filter.run(desired_vec, input_mat)
                res = res_arr[-1]
                rad = np.sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3])
                print("The radius is: {}".format(rad))
                for i in range(top_points_mm_s.shape[0]):
                    top_points_mm_s[i, 2] = -np.sqrt(np.power(rad, 2) -
                                                     np.power(
                                                         top_points_mm_s[i, 0] - res[0], 2) -
                                                     np.power(
                                                         top_points_mm_s[i, 1] - res[1], 2)) + res[2]
                self.top_smooth_pcd = o3d.geometry.PointCloud()
                self.top_smooth_pcd.points =\
                    o3d.utility.Vector3dVector(top_points_mm_s)
                self.top_smooth_pcd.normals =\
                    self.sphere_normal(top_points_mm_s, res[:-1].T[0])
                self.top_smooth_pcd.normalize_normals()

    def sphere_normal(self, points, center):
        normal = points - center
        return o3d.utility.Vector3dVector(normal)

    def pcd_fit_spline(self, layer='top'):
        if layer == 'top':
            top_points_mm_s = np.array(np.asarray(self.top_pcd.points),
                                       copy=True)
            points_mm_s = np.array(np.asarray(self.top_points_mm), copy=True)
            spline = interpolate.SmoothBivariateSpline(top_points_mm_s[:, 0],
                                                       top_points_mm_s[:, 1],
                                                       top_points_mm_s[:, 2],
                                                       bbox=[0,
                                                             np.max(
                                                                 points_mm_s[:,
                                                                 0]),
                                                             0,
                                                             np.max(
                                                                 points_mm_s[:,
                                                                 1])],
                                                       kx=3, ky=3, s=2)
            # print("The spline coefficients:", spline.get_coeffs().shape)
            print("The spline knots:", spline.get_knots())
            print("Spline fitting residual:", spline.get_residual())
            for idx in tqdm(range(points_mm_s.shape[0])):
                points_mm_s[idx, 2] =\
                    spline(points_mm_s[idx, 0],
                           points_mm_s[idx, 1])
            self.top_smooth_pcd = o3d.geometry.PointCloud()
            self.top_smooth_pcd.points =\
                o3d.utility.Vector3dVector(points_mm_s)
            self.top_smooth_pcd.normals = self.spline_normal(points_mm_s,
                                                             spline)
            self.top_smooth_pcd.orient_normals_to_align_with_direction(
                np.array([0.0, 0.0, -1.0]))
            self.top_smooth_pcd.normalize_normals()
        elif layer == 'bot':
            bot_points_mm_s = np.array(np.asarray(
                self.corrected_bot_pcd.points),
                                       copy=True)
            points_mm_s = np.array(np.asarray(self.bot_points_mm), copy=True)
            spline = interpolate.SmoothBivariateSpline(bot_points_mm_s[:, 0],
                                                       bot_points_mm_s[:, 1],
                                                       bot_points_mm_s[:, 2],
                                                       bbox=[0,
                                                             np.max(
                                                                 points_mm_s[:,
                                                                 0]),
                                                             0,
                                                             np.max(
                                                                 points_mm_s[:,
                                                                 1])],
                                                       kx=3, ky=3, s=2)
            # print("The spline coefficients:", spline.get_coeffs().shape)
            print("The spline knots:", spline.get_knots())
            print("Spline fitting residual:", spline.get_residual())
            for idx in tqdm(range(points_mm_s.shape[0])):
                points_mm_s[idx, 2] = \
                    spline(points_mm_s[idx, 0],
                           points_mm_s[idx, 1])
            self.bot_smooth_pcd = o3d.geometry.PointCloud()
            self.bot_smooth_pcd.points =\
                o3d.utility.Vector3dVector(points_mm_s)
            self.bot_smooth_pcd.normals = self.spline_normal(points_mm_s,
                                                             spline)
            self.bot_smooth_pcd.orient_normals_to_align_with_direction(
                np.array([0.0, 0.0, -1.0]))
            self.bot_smooth_pcd.normalize_normals()

    def spline_normal(self, points, spline):
        der_x, der_y = np.zeros_like(points), np.zeros_like(points)
        der_x[:, 0], der_y[:, 1] = 1, 1
        der_x[:, 2] = spline.ev(points[:, 0], points[:, 1], dx=1, dy=0)
        der_y[:, 2] = spline.ev(points[:, 0], points[:, 1], dx=0, dy=1)
        normals = np.cross(der_x, der_y)
        return o3d.utility.Vector3dVector(normals)

    def cal_normal(self, layer="top", method="marcos"):
        if method == "marcos":
            kernel_h, kernel_v = np.array([[0.5, 0, -0.5]]), np.array([[-0.5], [0], [0.5]])
            if layer == "top":
                normal_cal = nc.Marcos_normal(kernel_h, kernel_v, self.top_points_mm, self.xdim, self.ydim)
                self.top_pcd.normals = o3d.utility.Vector3dVector(normal_cal.get_normal() * [0.0, 0.0, -1.0])
            elif layer == "bot":
                normal_cal = nc.Marcos_normal(kernel_h, kernel_v, self.bot_points_mm, self.xdim, self.ydim)
                self.bot_pcd.normals = o3d.utility.Vector3dVector(normal_cal.get_normal())
            else:
                raise ValueError("Please indicate correct layer's name.")
        elif method == "o3d":
            if layer == "top":
                self.top_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100),
                                              fast_normal_computation=False)
                self.top_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                self.top_pcd.normalize_normals()
            elif layer == "bot":
                self.bot_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100),
                                              fast_normal_computation=False)
                self.bot_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                self.bot_pcd.normalize_normals()
            else:
                raise ValueError("Please indicate correct layer's name.")
        else:
            raise ValueError("Please indicate correct normal calculation method.")

    def ray_tracing(self, incidents, layer='top'):
        # if incidents.shape[0] != self.xdim * self.ydim:
        #     raise ValueError("The incident number: " +
        #                      str(incidents.shape[0]) +
        #                      " is not equal to point cloud's points number: " + str(self.xdim * self.ydim) + ".")
        if layer == 'top':
            points, normals = np.asarray(self.top_smooth_pcd.points), np.asarray(self.top_smooth_pcd.normals)
            r = self.n1 / self.n2
            for i in range(points.shape[0]):
                c = -np.dot(normals[i], incidents[i])
                refract = r * incidents[i] + \
                                       (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
                refract_norm = np.linalg.norm(refract)
                self.refracts_top[i] = refract / refract_norm
        elif layer == 'raw_top':
            points, normals = np.asarray(self.top_pcd.points), np.asarray(self.top_pcd.normals)
            r = self.n1 / self.n2
            self.refracts_raw_top = np.zeros(points.shape)
            for i in range(points.shape[0]):
                c = -np.dot(normals[i], incidents[i])
                refract = r * incidents[i] + \
                                       (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
                refract_norm = np.linalg.norm(refract)
                self.refracts_raw_top[i] = refract / refract_norm
        elif layer == 'bot':
            points, normals = np.asarray(self.bot_smooth_pcd.points), np.asarray(self.bot_smooth_pcd.normals)
            r = self.n2 / self.n3
            for i in range(points.shape[0]):
                c = -np.dot(normals[i], incidents[i])
                refract = r * incidents[i] + \
                          (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
                refract_norm = np.linalg.norm(refract)
                self.refracts_bot[i] = refract / refract_norm
        else:
            raise ValueError("The layer input: " + layer + " does not exist.")

    def refraction_correction(self, raw=False):
        if raw is False:
            top_points = np.asarray(self.top_smooth_pcd.points)
            bot_points = np.asarray(self.bot_pcd.points)
            refracts_top = self.refracts_top
        else:
            top_points = np.asarray(self.top_pcd.points)
            clip_bot_pcd = self.bot_pcd.select_by_index(self.top_ind)
            bot_points = np.asarray(clip_bot_pcd.points)
            refracts_top = self.refracts_raw_top
        if top_points.shape != bot_points.shape:
            raise ValueError("The two point clouds shape is not same!")
        raw_z_distance = np.absolute(bot_points[:, 2] - top_points[:, 2])
        z_distance = raw_z_distance/(self.n2 / self.n1)
        ref_vec_arr = np.zeros(bot_points.shape)
        for i in range(ref_vec_arr.shape[0]):
            ref_vec_arr[i] = z_distance[i] * refracts_top[i]
        corrected_bot_points = top_points + ref_vec_arr
        self.corrected_bot_pcd.points = o3d.utility.Vector3dVector(corrected_bot_points)
        self.corrected_bot_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100),
                                                fast_normal_computation=False)
        self.corrected_bot_pcd.paint_uniform_color([0.5, 0.5, 0.0])


if __name__ == "__main__":
    seg = RealPCD("../data/seg_res/seg_res_calib_760/", [200, 600], 416, 401, 310, 5.73, 5.0, 1.68, 1, 1.466, 1)

    """Remove the outlier"""
    # seg.edit_pcd()
    seg.remove_outlier(layer='top')

    seg.pcd_fit_sphere(method='ls')
    seg.cal_normal(method='o3d')
    seg.cal_normal(layer='bot', method='o3d')
    top_pcd = seg.get_top_pcd()
    top_smooth_pcd = seg.get_top_smooth_pcd()
    bot_pcd = seg.get_bot_pcd()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    top_smooth_pcd.paint_uniform_color([0, 1, 0])
    top_pcd.paint_uniform_color([1, 0, 0])
    bot_pcd.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([top_pcd, bot_pcd, mesh_frame], window_name="contact lens two layers",
    #                                   point_show_normal=False)
    # o3d.visualization.draw_geometries([top_smooth_pcd, top_pcd, mesh_frame], window_name="contact lens compare",
    #                                   point_show_normal=False)

    seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(top_smooth_pcd.points).shape[0], axis=0))
    seg.refraction_correction()
    seg.remove_outlier(layer='corrected_bot')
    corrected_bot_pcd = seg.get_corrected_bot_pcd()
    o3d.visualization.draw_geometries([top_pcd, corrected_bot_pcd, bot_pcd, mesh_frame],
                                      window_name="smooth index correction on bottom layer",
                                      point_show_normal=False)

    seg.pcd_fit_sphere(layer='bot', method='ls')
    bot_smooth_pcd = seg.get_bot_smooth_pcd()
    bot_smooth_pcd.paint_uniform_color([0, 1, 0])
    corrected_bot_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([corrected_bot_pcd, bot_smooth_pcd, mesh_frame],
                                      window_name="smooth fit a sphere on bottom layer",
                                      point_show_normal=False)

    seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(top_pcd.points).shape[0], axis=0), layer="raw_top")
    seg.refraction_correction(raw=True)
    corrected_bot_pcd = seg.get_corrected_bot_pcd()
    o3d.visualization.draw_geometries([top_pcd, corrected_bot_pcd, bot_pcd, mesh_frame],
                                      window_name="raw index correction on bottom layer (K=100)",
                                      point_show_normal=False)

    seg.pcd_fit_sphere(layer='bot', method='ls')
    bot_smooth_pcd = seg.get_bot_smooth_pcd()
    bot_smooth_pcd.paint_uniform_color([0, 1, 0])
    corrected_bot_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([corrected_bot_pcd, bot_smooth_pcd, mesh_frame],
                                      window_name="raw fit a sphere on bottom layer (K=100)",
                                      point_show_normal=False)

    # """Down sample the real data"""
    # for i in range(2, 8, 2):
    #     diff_angle_arr = downsample_compare(top_pcd, i)
    #     print("The mean of two Open3D angle array is: {} + {}".format(np.mean(diff_angle_arr),
    #                                                                   np.std(diff_angle_arr)))
