import numpy as np
import open3d as o3d

from csv_pcd import RealPCD
import segio


class CorneaPCD(RealPCD):
    def __init__(self, directory, xdim, ydim, zdim, xlength,
                 ylength, zlength, n1, n2, n3, vol_idx):
        self.directory = directory
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.vol_idx = vol_idx

        self.top_points = np.zeros((xdim * ydim, 3))
        self.bot_points = np.zeros((xdim * ydim, 3))
        self.top_points_mm = np.zeros((xdim * ydim, 3))
        self.bot_points_mm = np.zeros((xdim * ydim, 3))
        self.top_pcd = o3d.geometry.PointCloud()
        self.bot_pcd = o3d.geometry.PointCloud()
        self.top_smooth_pcd = o3d.geometry.PointCloud()
        self.bot_smooth_pcd = o3d.geometry.PointCloud()
        self.corrected_bot_pcd = o3d.geometry.PointCloud()
        self.refracts_top = np.zeros((self.xdim * self.ydim, 3))
        self.refracts_raw_top = None
        self.refracts_bot = np.zeros((self.xdim * self.ydim, 3))

        segs = list(segio.volumes(open(directory + '.seg', 'rb')))
        seg_top_layers = segs[self.vol_idx]['topLayers']
        seg_bot_layers = segs[self.vol_idx]['botLayers']
        print("Init function finished.")
        for i in range(seg_top_layers.shape[0]):
            top_seg, bot_seg = np.zeros((xdim, 3)), np.zeros((xdim, 3))
            top_seg[:, 1], bot_seg[:, 1] = i, i
            top_seg[:, 0] = seg_top_layers[i][:, 0]
            bot_seg[:, 0] = seg_bot_layers[i][:, 0]
            top_seg[:, 2] = seg_top_layers[i][:, 1]
            bot_seg[:, 2] = seg_bot_layers[i][:, 1]
            top_seg_mm = np.multiply(top_seg,
                                     [float(xlength) / xdim,
                                      float(ylength) / ydim,
                                      float(zlength) / zdim])
            bot_seg_mm = np.multiply(bot_seg,
                                     [float(xlength) / xdim,
                                      float(ylength) / ydim,
                                      float(zlength) / zdim])
            self.top_points[i * xdim: (i + 1) * xdim] =\
                np.asarray(top_seg, dtype=np.int)
            self.bot_points[i * xdim: (i + 1) * xdim] =\
                np.asarray(bot_seg, dtype=np.int)
            self.top_points_mm[i * xdim: (i + 1) * xdim] = top_seg_mm
            self.bot_points_mm[i * xdim: (i + 1) * xdim] = bot_seg_mm
        self.top_pcd.points = o3d.utility.Vector3dVector(self.top_points_mm)
        self.bot_pcd.points = o3d.utility.Vector3dVector(self.bot_points_mm)


if __name__ == "__main__":
    test = CorneaPCD("../data/201208_DALK/12-08-2020_6_44_23_PM", 250, 48,
                     198, 12, 7.19, 8, 1.0003, 1.4815, 1.0003, 0)
    top_pcd = test.get_top_pcd()
    bot_pcd = test.get_bot_pcd()
    mesh_frame = \
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                          origin=[0, 0, 0])
    o3d.visualization.draw_geometries([top_pcd, bot_pcd, mesh_frame])
    print("Program Finished.")
