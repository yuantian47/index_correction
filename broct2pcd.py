import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm
import pandas as pd

from PyBROCT.io.reader import scans


class Broct2pcd:
    def __init__(self, broct_name, dim_arr, threshold):
        self.broct_name = broct_name
        self.dim_arr = dim_arr
        self.threshold = threshold
        for (index, data) in scans(self.broct_name):
            self.volume = data['volume']
            print("Changing the volume coordinate from left hand to right hand:")
            for i in tqdm(range(self.volume.shape[0])):
                self.volume[i] = self.volume[i][::-1, :]

    def trans_vol2mm(self):
        x_unit = self.dim_arr[0][0] / self.dim_arr[1][0]
        y_unit = self.dim_arr[0][1] / self.dim_arr[1][1]
        z_unit = self.dim_arr[0][2] / self.dim_arr[1][2]
        max_Ascan_index = np.argmax(self.volume, axis=1)
        volume_singlelay = []
        for i in tqdm(range(max_Ascan_index.shape[0])):
            for j in range(max_Ascan_index.shape[1]):
                if self.volume[i, max_Ascan_index[i][j], j] >= self.threshold:
                    volume_singlelay.append([i, max_Ascan_index[i][j], j])
        processed_volume = np.multiply(volume_singlelay, np.array([z_unit, y_unit, x_unit]))
        processed_volume = np.flip(processed_volume, 1)
        return processed_volume

    def gen_pcd(self, processed_volume):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(processed_volume)
        return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer the .broct file to point cloud in Open3d")
    parser.add_argument("broct_name", help="The location and filename of the .broct file.")
    parser.add_argument("dim_arr_name", help="The location and filename of the dimension file.")
    parser.add_argument("threshold", help="The point cloud threshold.", type=int)

    args = parser.parse_args()
    dim_arr = np.array(pd.read_csv(args.dim_arr_name, header=None))
    oct_data = Broct2pcd(args.broct_name, dim_arr, args.threshold)
    processed_volume = oct_data.trans_vol2mm()
    pcd = oct_data.gen_pcd(processed_volume)
    o3d.visualization.draw_geometries([pcd], 'OCT Point Cloud')
