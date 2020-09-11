import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import padasip as pa

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
        self.top_points_mm, self.bot_points_mm = np.zeros((xdim * ydim, 3)), np.zeros((xdim * ydim, 3))
        self.top_pcd, self.bot_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.refracts_top = np.zeros((self.xdim * self.ydim, 3))
        self.refracts_bot = np.zeros((self.xdim * self.ydim, 3))
        point_idx = 0
        for i in tqdm(range(self.idx_range[0], self.idx_range[1] + 1)):
            top_seg_raw = np.array(pd.read_csv(self.directory + "result_top_" + str(i) + ".csv", header=None))
            bot_seg_raw = np.array(pd.read_csv(self.directory + "result_bot_" + str(i) + ".csv", header=None))
            top_seg, bot_seg = np.zeros((xdim, 3)), np.zeros((xdim, 3))
            for j in range(xdim):
                same_x_top = top_seg_raw[list([*np.where(top_seg_raw[:, 0] == j)[0]])]
                top_seg[j] = np.insert(same_x_top[np.argmax(same_x_top[:, 1])], 1, i)
                same_x_bot = bot_seg_raw[list([*np.where(bot_seg_raw[:, 0] == j)[0]])]
                bot_seg[j] = np.insert(same_x_bot[np.argmax(same_x_bot[:, 1])], 1, i)
            top_seg_mm = np.multiply(top_seg, [float(self.xlength) / self.xdim, float(self.ylength) / self.ydim,
                                               float(self.zlength) / self.zdim])
            bot_seg_mm = np.multiply(bot_seg, [float(self.xlength) / self.xdim, float(self.ylength) / self.ydim,
                                               float(self.zlength) / self.zdim])
            self.top_points[point_idx:point_idx + xdim], self.bot_points[point_idx:point_idx + xdim] = top_seg, bot_seg
            self.top_points_mm[point_idx:point_idx + xdim], self.bot_points_mm[point_idx:point_idx + xdim] = \
                top_seg_mm, bot_seg_mm
            point_idx += xdim
        self.top_pcd.points = o3d.utility.Vector3dVector(self.top_points_mm)
        self.bot_pcd.points = o3d.utility.Vector3dVector(self.bot_points_mm)

    def edit_pcd(self, layer='top'):
        if layer == 'top':
            o3d.visualization.draw_geometries_with_editing([self.top_pcd], 'Edit Top Point Cloud')
            self.top_pcd = o3d.io.read_point_cloud('../data/pcd/top_pcd.ply')
        elif layer == 'bot':
            o3d.visualization.draw_geometries_with_editing([self.bot_pcd], 'Edit Bottom Point Cloud')
            self.bot_pcd = o3d.io.read_point_cloud('../data/pcd/bot_pcd.ply')
        else:
            raise ValueError("Please input vaild layer's name.")

    def remove_outlier(self, layer='top', method='statistical', neighbors=100, std_ratio=2.0, radius=0.1):
        if layer == 'top':
            if method == 'statistical':
                cl, ind = self.top_pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
            elif method == 'radius':
                cl, ind = self.top_pcd.remove_radius_outlier(nb_points=neighbors, radius=radius)
            else:
                raise ValueError("Please input valid outlier removal method.")
            display_inlier_outlier(self.top_pcd, ind)
            self.top_pcd = self.top_pcd.select_by_index(ind)
        else:
            raise ValueError("Please input valid layer's name.")

    def get_top_pcd(self):
        return self.top_pcd

    def get_bot_pcd(self):
        return self.bot_pcd

    def filter_pcd(self, layer='top', method='lms'):
        if method == 'ls':
            if layer == 'top':
                top_potints_mm_s = np.array(np.asarray(self.top_pcd.points), copy=True)
                co_mat = np.zeros((top_potints_mm_s.shape[0], 4))
                co_mat[:, 0] = top_potints_mm_s[:, 0] * 2
                co_mat[:, 1] = top_potints_mm_s[:, 1] * 2
                co_mat[:, 2] = top_potints_mm_s[:, 2] * 2
                co_mat[:, 3] = 1
                ordinate = np.zeros((top_potints_mm_s.shape[0], 1))
                ordinate[:, 0] = np.sum(np.power(top_potints_mm_s, 2), axis=1)
                res, err, _, _ = np.linalg.lstsq(co_mat, ordinate, rcond=None)
                print("The error is:", err)
                rad = np.sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3])
                print("The radius is: {}".format(rad))
                for i in range(top_potints_mm_s.shape[0]):
                    top_potints_mm_s[i, 2] = -np.sqrt(np.power(rad, 2) -
                                                      np.power(top_potints_mm_s[i, 0] - res[0], 2) -
                                                      np.power(top_potints_mm_s[i, 1] - res[1], 2)) + res[2]
                smooth_pcd = o3d.geometry.PointCloud()
                smooth_pcd.points = o3d.utility.Vector3dVector(top_potints_mm_s)
                smooth_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100),
                                            fast_normal_computation=False)
                smooth_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                smooth_pcd.normalize_normals()
                return smooth_pcd
            if layer == 'bot':
                return None
        if method == 'lms':
            if layer == 'top':
                top_potints_mm_s = np.array(np.asarray(self.top_pcd.points), copy=True)
                desired_vec = np.zeros((top_potints_mm_s.shape[0], 1))
                desired_vec[:, 0] = np.sum(np.power(top_potints_mm_s, 2), axis=1)
                input_mat = np.zeros((top_potints_mm_s.shape[0], 4))
                input_mat[:, 0] = top_potints_mm_s[:, 0] * 2
                input_mat[:, 1] = top_potints_mm_s[:, 1] * 2
                input_mat[:, 2] = top_potints_mm_s[:, 2] * 2
                input_mat[:, 3] = 1
                filter = pa.filters.FilterLMS(n=4, mu=0.005, w="random")
                _, _, res_arr = filter.run(desired_vec, input_mat)
                res = res_arr[-1]
                rad = np.sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3])
                print("The radius is: {}".format(rad))
                for i in range(top_potints_mm_s.shape[0]):
                    top_potints_mm_s[i, 2] = -np.sqrt(np.power(rad, 2) -
                                                      np.power(top_potints_mm_s[i, 0] - res[0], 2) -
                                                      np.power(top_potints_mm_s[i, 1] - res[1], 2)) + res[2]
                smooth_pcd = o3d.geometry.PointCloud()
                smooth_pcd.points = o3d.utility.Vector3dVector(top_potints_mm_s)
                smooth_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100),
                                            fast_normal_computation=False)
                smooth_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                smooth_pcd.normalize_normals()
                return smooth_pcd

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
        if incidents.shape[0] != self.xdim * self.ydim:
            raise ValueError("The incident number: " +
                             str(incidents.shape[0]) +
                             " is not equal to point cloud's points number: " + str(self.xdim * self.ydim) + ".")
        if layer == 'top':
            points, normals = np.asarray(self.top_pcd.points), np.asarray(self.top_pcd.normals)
            r = self.n1 / self.n2
            for i in range(points.shape[0]):
                c = -np.dot(normals[i], incidents[i])
                self.refracts_top[i] = r * incidents[i] + \
                                       (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
        elif layer == 'bot':
            points, normals = np.asarray(self.bot_pcd.points), np.asarray(self.bot_pcd.normals)
            r = self.n2 / self.n3
            for i in range(points.shape[0]):
                c = -np.dot(normals[i], incidents[i])
                self.refracts_bot[i] = r * incidents[i] + \
                                       (r * c - np.sqrt(1 - np.power(r, 2) * (1 - np.power(c, 2)))) * normals[i]
        else:
            raise ValueError("The layer input: " + layer + " does not exist.")


if __name__ == "__main__":
    seg = RealPCD("../data/seg_res/", [200, 600], 416, 401, 310, 5.81, 5.00, 1.68, 1, 1.466, 1)

    """Remove the outlier"""
    # seg.edit_pcd()
    seg.remove_outlier()

    smooth_pcd = seg.filter_pcd(method='ls')
    seg.cal_normal(method='o3d')
    seg.cal_normal(layer='bot', method='o3d')
    top_pcd = seg.get_top_pcd()
    bot_pcd = seg.get_bot_pcd()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    smooth_pcd.paint_uniform_color([0, 1, 0])
    top_pcd.paint_uniform_color([1, 0, 0])
    bot_pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([top_pcd, bot_pcd, mesh_frame], window_name="contact lens two layers",
                                      point_show_normal=False)
    o3d.visualization.draw_geometries([smooth_pcd, top_pcd, mesh_frame], window_name="contact lens compare",
                                      point_show_normal=False)
    top_pcd_dp = top_pcd.voxel_down_sample(0.05)
    smooth_pcd_dp = smooth_pcd.voxel_down_sample(0.05)
    o3d.visualization.draw_geometries([top_pcd_dp, mesh_frame], window_name="raw contact lens normal (Open3D)",
                                      point_show_normal=True)
    o3d.visualization.draw_geometries([smooth_pcd_dp, mesh_frame], window_name="smoothed contact lens normal (Open3D)",
                                      point_show_normal=True)
    # angle_raw_smoothed = nc.angle_between_normals(np.asarray(top_pcd.normals), np.asarray(smooth_pcd.normals))
    # print("The difference between raw and smoothed pcd: {} +- {}".format(np.mean(angle_raw_smoothed),
    #                                                                      np.std(angle_raw_smoothed)))
    # seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(top_pcd.points).shape[0], axis=0))
    # top_pcd.normals = o3d.utility.Vector3dVector(seg.refracts_top)
    # o3d.visualization.draw_geometries([top_pcd, mesh_frame], window_name="contact lens ray trace",
    #                                   point_show_normal=True)

    """Down sample the real data"""
    for i in range(2, 10, 2):
        diff_angle_arr = downsample_compare(top_pcd, i)
        print("The mean of two marcos angle array is: {} + {}".format(np.mean(diff_angle_arr),
                                                                      np.std(diff_angle_arr)))
