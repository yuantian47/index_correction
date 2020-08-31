import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

import normal_calculation as nc


class RealPCD:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength):
        self.directory = directory
        self.idx_range = idx_range
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.top_points, self.bot_points = np.zeros((xdim*ydim, 3)), np.zeros((xdim*ydim, 3))
        self.top_points_mm, self.bot_points_mm = np.zeros((xdim*ydim, 3)), np.zeros((xdim*ydim, 3))
        self.top_pcd, self.bot_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        point_idx = 0
        for i in tqdm(range(self.idx_range[0], self.idx_range[1] + 1)):
            top_seg_raw = np.array(pd.read_csv(self.directory+"result_top_" + str(i) + ".csv", header=None))
            bot_seg_raw = np.array(pd.read_csv(self.directory+"result_bot_" + str(i) + ".csv", header=None))
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
            self.top_points[point_idx:point_idx+xdim], self.bot_points[point_idx:point_idx+xdim] = top_seg, bot_seg
            self.top_points_mm[point_idx:point_idx+xdim], self.bot_points_mm[point_idx:point_idx+xdim] = \
                top_seg_mm, bot_seg_mm
            point_idx += xdim
        self.top_pcd.points = o3d.utility.Vector3dVector(self.top_points_mm)
        self.bot_pcd.points = o3d.utility.Vector3dVector(self.bot_points_mm)

    def get_top_pcd(self):
        return self.top_pcd

    def get_bot_pcd(self):
        return self.bot_pcd

    def cal_normal(self, layer="top", method="marcos"):
        if method == "marcos":
            kernel_h, kernel_v = np.array([[0.5, 0, -0.5]]), np.array([[-0.5], [0], [0.5]])
            if layer == "top":
                normal_cal = nc.Marcos_normal(kernel_h, kernel_v, self.top_points_mm, self.xdim, self.ydim)
                self.top_pcd.normals = o3d.utility.Vector3dVector(normal_cal.get_normal())
            elif layer == "bot":
                normal_cal = nc.Marcos_normal(kernel_h, kernel_v, self.bot_points_mm, self.xdim, self.ydim)
                self.bot_pcd.normals = o3d.utility.Vector3dVector(normal_cal.get_normal())
            else:
                raise ValueError("Please indicate correct layer's name.")
        elif method == "o3d":
            if layer == "top":
                self.top_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=35),
                                              fast_normal_computation=False)
                self.top_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                self.top_pcd.normalize_normals()
            elif layer == "bot":
                self.bot_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=35),
                                              fast_normal_computation=False)
                self.bot_pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, -1.0]))
                self.bot_pcd.normalize_normals()
            else:
                raise ValueError("Please indicate correct layer's name.")
        else:
            raise ValueError("Please indicate correct normal calculation method.")


if __name__ == "__main__":
    seg = RealPCD("../data/seg_res/", [200, 600], 416, 401, 310, 5.81, 5.00, 1.68)
    seg.cal_normal(method='o3d')
    seg.cal_normal(layer='bot', method='o3d')
    top_pcd = seg.get_top_pcd()
    bot_pcd = seg.get_bot_pcd()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([top_pcd, mesh_frame], window_name="contact len's normal", point_show_normal=True)
