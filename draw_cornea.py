import numpy as np
import open3d as o3d


def draw_sphere(center, radius, xdim, zdim, xlen, zlen):
    xd = float(xlen)/xdim
    zd = float(zlen)/zdim
    sphere_points = []
    for z in range(int(-zdim/2), int(zdim/2)):
        if abs(z*zd) > radius:
            continue
        elif abs(abs(z*zd) - radius) < 1e-3:
            sphere_points.append([center[0], 0, center[1] + z*zd])
            continue
        else:
            x_max_abs = np.sqrt(np.power(radius, 2) - np.power(z*zd, 2))
            if x_max_abs < xd:
                continue
            else:
                for x in range(int(-x_max_abs/xd), int(x_max_abs/xd)):
                    y = np.sqrt(np.power(radius, 2) - np.power(z*zd, 2) - np.power(x*xd, 2))
                    sphere_points.append([center[0] + x*xd, y, center[1] + z*zd])
    return np.asarray(sphere_points, np.float)


if __name__ == "__main__":
    sphere_points = draw_sphere(np.array([-0.15, 0.2]), 6.0, 500, 96, 12.0, 10.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], 'OCT Point Cloud')
