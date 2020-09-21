import numpy as np
import open3d as o3d
from tqdm import tqdm
import scipy.interpolate

import csv_pcd


class Interpolation:
    def __init__(self, directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength, n1, n2, n3):
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.seg = csv_pcd.RealPCD(directory, idx_range, xdim, ydim, zdim, xlength, ylength, zlength, n1, n2, n3)
        self.seg.remove_outlier(layer='top')
        self.seg.pcd_fit_sphere(method='ls')
        self.top_smooth_pcd = self.seg.get_top_smooth_pcd()
        self.top_smooth_pcd.paint_uniform_color([1, 0, 0])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]], np.asarray(self.top_smooth_pcd.points).shape[0], axis=0))
        self.seg.refraction_correction()
        self.seg.remove_outlier(layer='corrected_bot')
        self.seg.pcd_fit_sphere(layer='bot', method='ls')
        self.bot_smooth_pcd = self.seg.get_bot_smooth_pcd()
        self.bot_smooth_pcd.paint_uniform_color([0, 1, 0])
        self.seg.ray_tracing(np.repeat([[0.0, 0.0, 1.0]],
                                       np.asarray(self.top_smooth_pcd.points).shape[0], axis=0), layer="top")
        self.seg.ray_tracing(self.seg.refracts_top, layer='bot')
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
            top_refract = self.seg.refracts_top[i]
            bot_refract = self.seg.refracts_bot[i]
            positions[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            z_positions = np.multiply(np.linspace(0, self.zdim, self.zdim, endpoint=False),
                                      (float(self.zlength) / self.zdim))
            positions[idx_x, idx_y, :, 2] = z_positions
            top_point = np.argwhere(z_positions >= top_smooth_points[i][2])
            bot_point = np.argwhere(z_positions >= bot_smooth_points[i][2])
            values[idx_x, idx_y, :, :2] = top_smooth_points[i][:2]
            values[idx_x, idx_y, :top_point[0, 0], 2] = np.linspace(0, top_smooth_points[i][2], top_point[0, 0])
            values[idx_x, idx_y, top_point[0, 0]:bot_point[0, 0], :] = \
                self.refract_correction(top_refract, top_smooth_points[i],
                                        positions[idx_x, idx_y, top_point[0, 0]:bot_point[0, 0], :])
            values[idx_x, idx_y, bot_point[0, 0]:, :] = \
                self.refract_correction(bot_refract, bot_smooth_points[i],
                                        positions[idx_x, idx_y, bot_point[0, 0]:, :])
            idx_x += 1
            if idx_x == self.xdim:
                idx_y += 1
                idx_x = 0
        return positions, values

    def refract_correction(self, refract, origin, points):
        raw_distance = np.absolute(origin[2] - points[:, 2])
        distance = raw_distance / self.n2
        corrected_points = origin + (distance.reshape(distance.shape[0], 1) * refract)
        return corrected_points


if __name__ == "__main__":
    inter = Interpolation("../data/seg_res/seg_res_calib_760/", [200, 600], 416, 401, 310, 5.73, 5.0, 1.68, 1, 1.466, 1)
    positions, values = inter.linear_inter_pairs()
    linearinter = scipy.interpolate.LinearNDInterpolator(positions.reshape(-1, 3), values.reshape(-1, 3), fill_value=-1)
    print("Program Finished.")
