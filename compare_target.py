import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from interpolation import points_dis2plane, draw_registration_result


class TargetPCD:
    def __init__(self, directory, idx_range, xdim, ydim, zdim,
                 xlength, ylength, zlength):
        self.directory = directory
        self.idx_range = idx_range
        self.xdim, self.ydim, self.zdim = xdim, ydim, zdim
        self.xlength, self.ylength, self.zlength = xlength, ylength, zlength
        self.emp_points = np.zeros((xdim * ydim, 3))
        self.emp_points_mm = np.zeros((xdim * ydim, 3))
        self.emp_pcd = o3d.geometry.PointCloud()
        point_idx = 0
        for i in tqdm(range(self.idx_range[0], self.idx_range[1] + 1)):
            emp_seg_raw = np.array(pd.read_csv(directory + "/result_top_" +
                                               str(i) + ".csv", header=None))
            emp_seg = np.zeros((xdim, 3))
            for j in range(xdim):
                same_x_emp = emp_seg_raw[list([*np.where(emp_seg_raw[:,
                                                         0] == j)[0]])]
                emp_seg[j] = np.insert(same_x_emp[np.argmax(same_x_emp[:, 1])],
                                       1, i - self.idx_range[0])
            emp_seg_mm = np.multiply(emp_seg,
                                     [float(self.xlength) / self.xdim,
                                      float(self.ylength) / self.ydim,
                                      float(self.zlength) / self.zdim])
            self.emp_points_mm[point_idx:point_idx + xdim] = emp_seg_mm
            point_idx += xdim
        self.emp_pcd.points = o3d.utility.Vector3dVector(self.emp_points_mm)

    def get_point_cloud(self):
        return self.emp_pcd

    def svd_fit_plane(self, test_pcd):
        tar_points = np.asarray(test_pcd.points)
        emp_points = np.asarray(self.emp_pcd.points)
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
        fit_emp_pcd.paint_uniform_color([0.1, 0.5, 0])
        mesh_frame = \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                              origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [fit_tar_pcd, fit_emp_pcd, mesh_frame],
            window_name="fit planes on corrected air target and empty target",
            point_show_normal=False)
        dis_tar = points_dis2plane(tar_normal[0], tar_normal[1],
                                   tar_normal[2], d_tar, tar_points)
        dis_emp = points_dis2plane(emp_normal[0], emp_normal[1],
                                   emp_normal[2], d_emp, emp_points)
        print("Target 1 fitting error mean is {:.4f}, std is {:.4f}".format(
            np.mean(dis_tar), np.std(dis_tar)))
        print("Target 2 fitting error mean is {:.4f}, std is {:.4f}".
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

    def icp_registration(self, test_pcd):
        trans_init = np.eye(4)
        reg_p2p = o3d.registration.registration_icp(test_pcd,
                                                    self.emp_pcd,
                                                    1.0,
                                                    trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(max_iteration=2000))
        draw_registration_result(test_pcd, self.emp_pcd,
                                 reg_p2p.transformation)
        return reg_p2p


if __name__ == "__main__":
    tar_pcd_4 = TargetPCD("../data/seg_res/tar_test/tar_" + str(4) +
                          "_cropseg_res", [200, 600], 500, 401, 877, 7.022,
                          5.0125, 4.701)
    pcd_4 = tar_pcd_4.get_point_cloud()
    pcd_4.paint_uniform_color([0, 1, 0])

    tar_pcd_5 = TargetPCD("../data/seg_res/tar_test/tar_" + str(5) +
                          "_cropseg_res", [200, 600], 500, 401, 877, 7.022,
                          5.0125, 4.701)
    pcd_5 = tar_pcd_5.get_point_cloud()
    pcd_5.paint_uniform_color([1, 1, 1])

    tar_pcd_6 = TargetPCD("../data/seg_res/tar_test/tar_" + str(7) +
                          "_cropseg_res", [200, 600], 500, 401, 877, 7.022,
                          5.0125, 4.701)
    pcd_6 = tar_pcd_6.get_point_cloud()

    normal_diff_5, dis_diff_5 = tar_pcd_4.svd_fit_plane(pcd_5)
    normal_diff_6, dis_diff_6 = tar_pcd_4.svd_fit_plane(pcd_6)
    print("Normal diff 2:", normal_diff_5, "distance 2:", dis_diff_5)
    print("Normal diff 3:", normal_diff_6, "distance 3:", dis_diff_6)

    # reg_5 = tar_pcd_4.icp_registration(pcd_5)
    # print(reg_5.fitness, reg_5.inlier_rmse, reg_5.transformation)
    #
    # reg_6 = tar_pcd_4.icp_registration(pcd_6)
    # print(reg_6.fitness, reg_6.inlier_rmse, reg_6.transformation)

    # mesh_frame = \
    #     o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
    #                                                       origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_4, mesh_frame],
    #                                   window_name="index correction result",
    #                                   point_show_normal=False)
