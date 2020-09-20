import numpy as np
import open3d as o3d
from tqdm import tqdm

import csv_pcd


class Interpolation:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength, n1, n2, n3):
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        seg = csv_pcd.RealPCD(directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength, n1, n2, n3)
        seg.remove_outlier(layer='top')
        seg.pcd_fit_sphere(method='ls')
        self.top_smooth_pcd = seg.get_top_smooth_pcd()
        self.top_smooth_pcd.paint_uniform_color([1, 0, 0])
        seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(self.top_smooth_pcd.points).shape[0], axis=0))
        seg.refraction_correction()
        seg.remove_outlier(layer='corrected_bot')
        seg.pcd_fit_sphere(layer='bot', method='ls')
        self.bot_smooth_pcd = seg.get_bot_smooth_pcd()
        self.bot_smooth_pcd.paint_uniform_color([0, 1, 0])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.top_smooth_pcd, self.bot_smooth_pcd, mesh_frame],
                                          window_name="smooth fit a sphere on bottom layer",
                                          point_show_normal=False)

    def linear_inter_pairs(self):
        positions = np.zeros((self.xdim, self.ydim, self.zdim, 3))
        values = np.zeros((self.xdim, self.ydim, self.zdim, 3))
        print("Building linear interpolation matrix.")
        idx_x, idx_y = 0, 0
        top_smooth_points = np.asarray(self.top_smooth_pcd.points)
        bot_smooth_points = np.asarray(self.bot_smooth_pcd.points)
        for i in tqdm(range(top_smooth_points.shape[0])):
            positions[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            debug = np.linspace(0, self.zdim, self.zdim, endpoint=False)
            z_positions = np.multiply(np.linspace(0, self.zdim, self.zdim, endpoint=False),
                                      (float(self.zlength) / self.zdim))
            positions[idx_x, idx_y, :, 2] = z_positions
            top_point = np.argwhere(z_positions >= top_smooth_points[i][2])
            bot_point = np.argwhere(z_positions >= bot_smooth_points[i][2])
            values[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            values[idx_x, idx_y, :top_point[0, 0], 2] = np.linspace(0, top_smooth_points[i][2], top_point[0, 0])

            idx_x += 1
            idx_y += 1


if __name__ == "__main__":
    inter = Interpolation("../data/seg_res/seg_res_calib_760/", [200, 600], 416, 401, 310, 5.73, 5.0, 1.68, 1, 1.466, 1)
    inter.linear_inter_pairs()
