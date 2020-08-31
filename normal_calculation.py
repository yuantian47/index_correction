import numpy as np
import cv2 as cv
import draw_cornea
import open3d as o3d
import ray_tracing


class Marcos_normal:
    def __init__(self, kernel_h, kernel_v, sphere, dim_x, dim_y):
        self.kernel_h = kernel_h
        self.kernel_v = kernel_v
        self.sphere = sphere
        self.dim_x = dim_x
        self.dim_y = dim_y
        if self.dim_x * self.dim_y != self.sphere.shape[0]:
            raise ValueError("The sphere's points cannot generate a matrix!")
        self.mat_x, self.mat_y, self.mat_z = np.zeros((self.dim_x, self.dim_y)), np.zeros((self.dim_x, self.dim_y)),\
                                             np.zeros((self.dim_x, self.dim_y))
        for i in range(self.sphere.shape[0]):
            y_pos = i // self.dim_x + 1
            x_pos = i % self.dim_x
            self.mat_x[x_pos][-y_pos] = self.sphere[i][0]
            self.mat_y[x_pos][-y_pos] = self.sphere[i][1]
            self.mat_z[x_pos][-y_pos] = self.sphere[i][2]
        self.gxh, self.gyh, self.gzh = None, None, None
        self.gxv, self.gyv, self.gzv = None, None, None
        self.gh, self.gv = np.zeros((self.dim_x, self.dim_y, 3)), np.zeros((self.dim_x, self.dim_y, 3))
        self.normal = np.zeros(sphere.shape)

    def gradient_components_cal(self):
        self.gxh = cv.filter2D(self.mat_x, -1, self.kernel_h, borderType=cv.BORDER_REPLICATE)
        self.gyh = cv.filter2D(self.mat_y, -1, self.kernel_h, borderType=cv.BORDER_REPLICATE)
        self.gzh = cv.filter2D(self.mat_z, -1, self.kernel_h, borderType=cv.BORDER_REPLICATE)
        self.gxv = cv.filter2D(self.mat_x, -1, self.kernel_v, borderType=cv.BORDER_REPLICATE)
        self.gyv = cv.filter2D(self.mat_y, -1, self.kernel_v, borderType=cv.BORDER_REPLICATE)
        self.gzv = cv.filter2D(self.mat_z, -1, self.kernel_v, borderType=cv.BORDER_REPLICATE)

    def cal_normal(self):
        self.gh[:, :, 0], self.gh[:, :, 1], self.gh[:, :, 2] = self.gxh, self.gyh, self.gzh
        self.gv[:, :, 0], self.gv[:, :, 1], self.gv[:, :, 2] = self.gxv, self.gyv, self.gzv
        for i in range(self.normal.shape[0]):
            y_pos = i // self.dim_x + 1
            x_pos = i % self.dim_x
            self.normal[i] = np.cross(self.gv[x_pos, -y_pos], self.gh[x_pos, -y_pos]) /\
                             np.linalg.norm(np.cross(self.gv[x_pos, -y_pos], self.gh[x_pos, -y_pos]))

    def get_normal(self):
        self.gradient_components_cal()
        self.cal_normal()
        return self.normal


def angle_between_normals(normals_a, normals_b):
    if normals_a.shape != normals_b.shape:
        raise ValueError("Two normals' arrays have different shape!")
    angle_list = []
    for i in range(normals_a.shape[0]):
        angle_list.append(np.dot(normals_a[i], normals_b[i]) /
                          (np.linalg.norm(normals_a[i]) * np.linalg.norm(normals_b[i])))
    angle_array = np.arccos(angle_list)
    return angle_array


def delete_edge_points(pcd, dim_x, dim_y):
    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    new_points, new_normals = [], []
    for i in range(normals.shape[0]):
        y_pos = i // dim_x + 1
        x_pos = i % dim_x
        if y_pos == 0 or y_pos == dim_y or x_pos == 0 or x_pos == (dim_x - 1):
            continue
        else:
            new_points.append(points[i])
            new_normals.append(normals[i])
    pcd.points = o3d.utility.Vector3dVector(np.array(new_points))
    pcd.normals = o3d.utility.Vector3dVector(np.array(new_normals))
    return pcd


if __name__ == "__main__":
    sphere, min_x, max_x = draw_cornea.draw_sphere(np.array([0.0, 0.0, -5]), 10, 50, 50, 10.0, 10.0)
    kernel_h, kernel_v = np.array([[0.5, 0, -0.5]]), np.array([[-0.5], [0], [0.5]])
    normal_cal = Marcos_normal(kernel_h, kernel_v, sphere, max_x - min_x, 50)
    normal = normal_cal.get_normal()
    sphere_noise = np.copy(sphere)
    sphere_noise[:, 2] = sphere_noise[:, 2] + np.random.normal(0, 0.1, sphere_noise.shape[0])
    normal_cal_noise = Marcos_normal(kernel_h, kernel_v, sphere_noise, max_x - min_x, 50)
    normal_noise = normal_cal_noise.get_normal()

    pcd_marcos, pcd_o3d = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_marcos_noise, pcd_o3d_noise = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_marcos.points, pcd_o3d.points = o3d.utility.Vector3dVector(sphere), o3d.utility.Vector3dVector(sphere)
    pcd_marcos_noise.points, pcd_o3d_noise.points = o3d.utility.Vector3dVector(sphere_noise),\
                                                    o3d.utility.Vector3dVector(sphere_noise)
    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=35),
                             fast_normal_computation=False)
    pcd_o3d_noise.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=35),
                                   fast_normal_computation=False)
    pcd_o3d.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    pcd_o3d_noise.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    pcd_o3d.normalize_normals()
    pcd_o3d_noise.normalize_normals()
    pcd_marcos_noise.normals = o3d.utility.Vector3dVector(normal_noise)
    pcd_marcos.normals = o3d.utility.Vector3dVector(normal)

    pcd_marcos = delete_edge_points(pcd_marcos, max_x-min_x, 50)
    pcd_marcos_noise = delete_edge_points(pcd_marcos_noise, max_x-min_x, 50)
    pcd_o3d = delete_edge_points(pcd_o3d, max_x-min_x, 50)
    pcd_o3d_noise = delete_edge_points(pcd_o3d_noise, max_x-min_x, 50)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd_marcos, mesh_frame], window_name="Marcos Normal without Noise",
                                      point_show_normal=True)
    o3d.visualization.draw_geometries([pcd_marcos_noise, mesh_frame], window_name="Marcos Normal with Noise",
                                      point_show_normal=True)
    o3d.visualization.draw_geometries([pcd_o3d, mesh_frame], window_name="Open3D Normal without Noise", point_show_normal=True)
    o3d.visualization.draw_geometries([pcd_o3d_noise, mesh_frame], window_name="Open3D Normal with Noise",
                                      point_show_normal=True)

    angle_array = angle_between_normals(np.asarray(pcd_marcos.normals), np.asarray(pcd_o3d.normals))
    print("The mean of two angle array is: {}".format(np.mean(angle_array)))
    angle_array_noise = angle_between_normals(np.asarray(pcd_marcos_noise.normals), np.asarray(pcd_o3d_noise.normals))
    print("The mean of two noisy angle array is: {} ".format(np.mean(angle_array_noise)))
    angle_array_marcos = angle_between_normals(np.asarray(pcd_marcos.normals), np.asarray(pcd_marcos_noise.normals))
    print("The mean of two Marcos angle array is: {}".format(np.mean(angle_array_marcos)))
    angle_array_o3d = angle_between_normals(np.asarray(pcd_o3d.normals), np.asarray(pcd_o3d_noise.normals))
    print("The mean of two Open3D angle array is: {}".format(np.mean(angle_array_o3d)))

    incidents = np.array([np.array([0.0, 0.0, -1.0]) for i in range(np.asarray(pcd_o3d.normals).shape[0])])
    refracts = ray_tracing.ray_tracing(np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.normals),
                                       incidents, 1.000, 1.376)
    refracts_noise = ray_tracing.ray_tracing(np.asarray(pcd_o3d_noise.points), np.asarray(pcd_o3d_noise.normals),
                                       incidents, 1.000, 1.376)
    pcd_o3d.normals = o3d.utility.Vector3dVector(refracts)
    pcd_o3d_noise.normals = o3d.utility.Vector3dVector(refracts_noise)
    o3d.visualization.draw_geometries([pcd_o3d, mesh_frame],
                                      window_name="Ray Tracing without noise (Open3D's normal)", point_show_normal=True)
    o3d.visualization.draw_geometries([pcd_o3d_noise, mesh_frame],
                                      window_name="Ray Tracing without noise (Open3D's normal)", point_show_normal=True)

