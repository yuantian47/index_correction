import numpy as np
import open3d as o3d


def draw_sphere(center, radius, xdim, ydim, xlen, ylen):
    xd = float(xlen)/xdim
    yd = float(ylen)/ydim
    sphere_points = []
    min_x, max_x = int(-xdim/2), int(xdim/2)-1
    for y in range(int(-ydim/2), int(ydim/2)):
        if abs(y*yd) > radius:
            continue
        else:
            x_max_abs = min(np.sqrt(np.power(radius, 2) - np.power(y*yd, 2)), xlen)
            if x_max_abs < xd:
                continue
            else:
                min_x, max_x = max(int((-x_max_abs/xd)/2), min_x), min(int((x_max_abs/xd)/2), max_x)
                # for x in range(int((-x_max_abs/xd)/2), int((x_max_abs/xd)/2)):
                for x in range(min_x, max_x):
                    z = np.sqrt(np.power(radius, 2) - np.power(y*yd, 2) - np.power(x*xd, 2)) + center[1]
                    if z + center[2] >= 0:
                        sphere_points.append([center[0] + x*xd, center[1] + y*yd, z + center[2]])
    return np.asarray(sphere_points, np.float), min_x, max_x


if __name__ == "__main__":
    sphere_points, _, _ = draw_sphere(np.array([0.0, 0.0, -5]), 10.0, 50, 50, 10.0, 10.0)
    print("The shape of the sphere is:", sphere_points.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=10))
    o3d.visualization.draw_geometries([pcd, mesh_frame])
