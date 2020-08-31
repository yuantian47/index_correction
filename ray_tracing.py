import numpy as np
import open3d as o3d
import normal_calculation
import draw_cornea


def ray_tracing(points, normals, incidents, n1, n2):
    if points.shape != normals.shape or points.shape != incidents.shape:
        raise ValueError("The shape of points, normals, or incidents are not same!")
    r = n1/n2
    refracts = []
    for i in range(points.shape[0]):
        c = -np.dot(normals[i], incidents[i])
        refract = r * incidents[i] + (r * c - np.sqrt(1 - np.power(r, 2)*(1 - np.power(c, 2)))) * normals[i]
        refracts.append(refract)
    return np.asarray(refracts)


if __name__ == "__main__":
    sphere, min_x, max_x = draw_cornea.draw_sphere(np.array([0.0, 0.0, -5]), 10, 50, 50, 10.0, 10.0)
    kernel_h, kernel_v = np.array([[0.5, 0, -0.5]]), np.array([[-0.5], [0], [0.5]])
    normal_cal = normal_calculation.Marcos_normal(kernel_h, kernel_v, sphere, max_x - min_x, 50)
    normal = normal_cal.get_normal()

    pcd_marcos = o3d.geometry.PointCloud()
    pcd_marcos.points = o3d.utility.Vector3dVector(sphere)
    pcd_marcos.normals = o3d.utility.Vector3dVector(normal)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd_marcos, mesh_frame], window_name="Marcos Normal", point_show_normal=True)

    incidents = np.array([np.array([0.0, 0.0, -1.0]) for i in range(normal.shape[0])])
    refracts = ray_tracing(sphere, normal, incidents, 1.000, 1.376)
    pcd_marcos.normals = o3d.utility.Vector3dVector(refracts)
    o3d.visualization.draw_geometries([pcd_marcos, mesh_frame], window_name="Ray Tracing", point_show_normal=True)

