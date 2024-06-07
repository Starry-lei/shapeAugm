import os
import sys
from pathlib import Path
from copy import deepcopy
import yaml
import numpy as np
import torch
# from chamferdist import ChamferDistance
import open3d as o3d
from utils_ssm.SSM import *
from glob import glob
from tqdm import tqdm
from utils_ssm.evaluation_utils import (
    get_correspondended_vertices,
    get_target_point_cloud,
    get_test_point_cloud,
    save_point_cloud
)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def load_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)


def saveCorrespondence(corr_path, data_root, corr_file_name):
    corres_verts = []
    print("data_root:", data_root)
    corr_files = sorted(glob(os.path.join(data_root, 'output', '*.txt')))
    assert len(corr_files) != 0, "No correspondence files found"
    ref_mat = load_points(corr_files[0])
    for vert in ref_mat:
        corres_verts.append([vert])
    for idx,  corr_file in enumerate(corr_files):
        if idx == 0:
            continue
        print("corr_file", corr_file)
        file_mat = load_points(corr_file)
        for idx,  each in enumerate(file_mat):
            corres_verts[idx].append(each)
    corres_verts = np.array(corres_verts)
    print("corres_verts shape:",corres_verts.shape)
    np.save(corr_path+"/"+corr_file_name, corres_verts)
    print("corres_verts saved")

if __name__ == "__main__":

    output_data_root = "./bottle_total"
    output_data_path = "./bottle_total/output"
    corr_path = "./bottle_ssm"
    exp_temp_data = "./bottle_samples"
    corr_file_name = "bottle2048.npy"
    corrVertsPath = os.path.join(corr_path, corr_file_name)
    if not os.path.exists(corrVertsPath):
        saveCorrespondence(corr_path, output_data_root, corr_file_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correspondence_path = corrVertsPath
    corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
    raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
    print("raw_data_matrix shape:", raw_data_matrix.shape)
    num_shapes = raw_data_matrix.shape[0]
    ssm = SSM(np.transpose(corresponded_train_verts, (1, 0)))

    # Number of components for 99% of total variance
    print("Number of components for 99% of total variance: ", ssm.reuired_components_number)
    # Generate similar samples from the SSM
    n_samples = 10
    ref_shape_id = 3
    if ref_shape_id == -1:
        # randomly select a shape as the reference shape
        ref_shape_id = np.random.randint(0, num_shapes)


    ref_shape = raw_data_matrix[ref_shape_id]
    ref_shape = ref_shape.reshape(1, -1, 3)
    # visualize the reference shape
    pc_ref = o3d.geometry.PointCloud()
    pc_ref_bk = o3d.geometry.PointCloud()
    pc_ref_bk.points = o3d.utility.Vector3dVector(ref_shape.squeeze())
    pc_ref.points = o3d.utility.Vector3dVector(ref_shape.squeeze() + [0.1, 0, 0])
    # o3d.visualization.draw_geometries([pc_ref])
    ref_shape_weight = ssm.get_theta(ref_shape.squeeze(), n_modes=ssm.reuired_components_number)
    # ref_shape_weight = ref_shape_weight.transpose(1, 0)
    print("ref_shape_weight shape:", ref_shape_weight.shape)

    # reconstruction = ssm.theta_to_shape_norm(ref_shape_weight, n_modes=ssm.reuired_components_number)
    # reconstruction = reconstruction.reshape(1, -1, 3)
    # # visualize the reconstruction
    # pc= o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(reconstruction.squeeze())
    # o3d.visualization.draw_geometries([pc_ref, pc])
    # samples_new = ssm.generate_random_samples(n_samples = n_samples,  n_modes=ssm.reuired_components_number)
    # exit(0)

    samples = ssm.generate_similar_samples(n_samples = n_samples, n_modes=ssm.reuired_components_number, ref_shape_weight=ref_shape_weight)
    print("samples shape", samples.shape)

    pc_points = []
    pc_points.append(pc_ref_bk)
    # Visualize the samples
    for i in range(n_samples):
        # you can save point cloud here
        save_point_cloud(samples[i].reshape(-1, 3), exp_temp_data+"/sample"+str(i)+".pcd")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(samples[i].reshape(-1, 3) + [0.1*(i+1),0, 0])
        pc_points.append(pc)
    o3d.visualization.draw_geometries(pc_points)



    # Check the mean shape
    # mean_chair_pointCloud = ssm.mean.reshape(1, -1, 3)
    # mean_chair_pointCloud = mean_chair_pointCloud.squeeze()
    # print("mean_bottle_pointCloud shape", mean_chair_pointCloud.shape)
    # print("mean_bottle_pointCloud len", len(mean_chair_pointCloud))
    # save_point_cloud(mean_chair_pointCloud, "mean_bottle_pointCloud.pcd")



