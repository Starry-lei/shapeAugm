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
    corr_files = sorted(glob(os.path.join(data_root, '*.particles')))
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



class PartSSMs:
    def __init__(self, partNamesPath, SSM_save_path):
        self.partNames = []
        self.partSSMs = {}
        self.load_part_names(partNamesPath)
        self.load_ssms(SSM_save_path)

    class PartSSM:
        def __init__(self, ssm):
            self.mean = ssm["mean"]
            self.modes_norm = ssm["modes_norm"]
            # self.modes_scaled = ssm["modes_scaled"]
            self.variances = ssm["variances"]
            self.variance_mode_number = ssm["variance_mode_number"]

        def get_variance_num_modes(self, num_modes: int) -> np.ndarray:
            """
            Get the variance of the first num_modes modes
            Args:
                num_modes:  number of modes to consider
            Returns:
                variance:   variance of the first num_modes modes
            """
            return self.variances[:num_modes]

        def get_theta(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
            """
            Project shape into the SSM to get a reconstruction
            Args:
                shape:      shape to reconstruct
                n_modes:    number of modes to use. If None, all relevant modes are used
            Returns:
                data_proj:  projected data as reconstruction
            """
            shape = shape.reshape(-1)
            data_proj = shape - self.mean
            if n_modes:
                evecs = self.modes_norm[:, :n_modes]
            else:
                evecs = self.modes_norm
            data_proj_re = data_proj.reshape(-1, 1)
            weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
            return weights

        def theta_to_shape_norm(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
            """
            Reconstruct shape from theta
            Args:
                weights:      weights of modes
            Returns:
                shape:      reconstructed shape
            """
            if n_modes:
                evecs = self.modes_norm[:, :n_modes]
            else:
                evecs = self.modes_norm
            # print("weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
            # print("evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)
            data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
            data_proj = data_proj.reshape(-1, 3)

            return data_proj

        def generate_similar_samples(self, n_samples: int = 1, n_modes=None, ref_shape_weight=None) -> np.ndarray:
            """
            Generate similar samples from the SSM.
            Args:
                n_samples:  number of samples to generate
                n_modes:    number of modes to use
            Returns:
                samples:    Generated random samples
            """

            variation_coeff = 0.1
            if n_modes:
                evecs = self.modes_norm[:, :n_modes]
            else:
                evecs = self.modes_norm

            added_noise = np.random.normal(0, variation_coeff,
                                           (ref_shape_weight.shape[0], ref_shape_weight.shape[1], n_samples))
            # print("added_noise shape:", added_noise) #added_noise shape: (6, 1, 10)
            print("added_noise shape:", added_noise.shape)  # added_noise shape: (6, 1)

            added_noise = (ref_shape_weight + np.transpose(added_noise, (2, 0, 1))).squeeze(2)

            print("added_noise shape:", added_noise.shape)  # added_noise shape: (10, 6)

            print("self.modes_scaled shape:", evecs.shape)  # self.modes_scaled shape: (6, 3)

            samples = self.mean + np.matmul(added_noise, evecs.transpose())

            # samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())

            return np.squeeze(samples)




    def load_part_names(self, partNamesPath):
        with open(partNamesPath, "r") as f:
            for line in f:
                self.partNames.append(line.strip())
        print("partNames:", self.partNames)

    def load_ssms(self, SSM_save_path):
        for idx, ssm_name in enumerate(self.partNames):
            # print("idx:", idx, "ssm_name:", ssm_name)
            try:
                SSM_path = os.path.join(SSM_save_path, ssm_name + "_ssm.npz")
                self.partSSMs[ssm_name] = self.PartSSM(np.load(SSM_path))
            except FileNotFoundError:
                print(f"File not found: {SSM_save_path}")
            except Exception as e:
                print(f"Error loading {SSM_save_path}: {e}")




# save all ssm parameters
SSM_save_path = "./SSMs"
# for idx, ssm_name in enumerate(partNames):
#     print("idx:", idx, "ssm_name:", ssm_name)
#     SSM_path = os.path.join(part_SSMs_path, ssm_name + ".npy")
#     corresponded_vertex, n_particles = get_correspondended_vertices(None, path=SSM_path)
#     raw_data_matrix = np.transpose(corresponded_vertex, (1, 0))
#     ssm = SSM(raw_data_matrix)  # =-=-=-=-=-=-=-=-= load SSM optimize loading-=-=-=-=-=-=-=-=-=-=-=-=-
#     # save the ssm parameters
#     ssm_name_save_path = os.path.join(SSM_save_path, ssm_name + "_ssm")
#     ssm.save_ssm(ssm_name_save_path)


if __name__ == "__main__":

    # env: conda activate strucNet
    num_mode = 100
    ssm_dict = {}
    obejctNames = []
    obejctNames_path = "./SSMs/obejctNames.txt"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    augmented_data = "./augmented_data"
    shape_ssms_path = "./shape_ssms"
    cate_shape_corr_path = "./categorical_shape_correspondence"
    for idx,  cate in enumerate(os.listdir(cate_shape_corr_path)):
        print("Building categorical shape SSMs:", cate)
        corr_file_name = cate + ".npy"
        if not os.path.exists(obejctNames_path):
            obejctNames.append(cate)
        corrVertsPath = os.path.join(shape_ssms_path, corr_file_name)
        if not os.path.exists(corrVertsPath):
            output_data_root = os.path.join(cate_shape_corr_path, cate)
            saveCorrespondence(shape_ssms_path, output_data_root, corr_file_name)
    print("Building done!")

    # save obejctNames as partNames.txt
    if not os.path.exists(obejctNames_path):
        with open("./SSMs/obejctNames.txt", "w") as f:
            for name in obejctNames:
                f.write(name+"\n")
        print("obejctNames.txt saved!")


    # save the ssm parameters, eigen decomposition costs a lot of time
    # selected_cate = ['remote_ssm','box', 'can','cup','cutlery','teapot']
    selected_cate = ['remote_ssm']
    for idx, corr in enumerate(os.listdir(shape_ssms_path)):
        corr_name = corr.split(".")[0]
        preComp_ssm_name_npz = os.path.join(SSM_save_path, corr_name + "_ssm"+".npz")
        if not os.path.exists(preComp_ssm_name_npz):
            print("Building SSMs:", corr_name)
            if corr_name in selected_cate:
                print("done SSMs:", corr_name)
                continue
            else:
                print("Not in selected categories:", corr_name)
                corrVertsPath = os.path.join(shape_ssms_path, corr)
                corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
                raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
                print("raw_data_matrix shape:", raw_data_matrix.shape)
                num_shapes = raw_data_matrix.shape[0]
                ssm = SSM(np.transpose(corresponded_train_verts, (1, 0)))
                ssm.save_ssm(os.path.join(SSM_save_path, corr_name + "_ssm"), num_mode)
        print(corr_name + " SSM PreComputation done!")

    # Augmentation and visualization

    # 1. # Load the precomputed SSMs

    PartSSMs = PartSSMs(obejctNames_path, SSM_save_path)
    print("Load SSMs for all objects. done!")


    selected_cate = "shoe"

    for idx, corr in enumerate(os.listdir(shape_ssms_path)):
        if corr != selected_cate+".npy":
            continue
        print("Augmenting shape SSMs:", corr)

        corr_name = corr.split(".")[0]
        corrVertsPath= os.path.join(shape_ssms_path, corr)
        corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
        raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
        # print("raw_data_matrix shape:", raw_data_matrix.shape)
        num_shapes = raw_data_matrix.shape[0]
        ssm = PartSSMs.partSSMs[selected_cate]
        # Number of components for 99% of total variance
        print("Number of components for 99% of total variance: ", len(ssm.variance_mode_number))
        n_samples = 10
        components_number = 15
        ref_shape_id = 5  # [0, num_all_shoes]

        if ref_shape_id == -1:
            # randomly select a shape as the reference shape
            ref_shape_id = np.random.randint(0, num_shapes)

        ref_shape = raw_data_matrix[ref_shape_id]
        ref_shape = ref_shape.reshape(1, -1, 3)
        # visualize the reference shape
        pc_ref = o3d.geometry.PointCloud()
        pc_ref_bk = o3d.geometry.PointCloud()
        pc_ref_bk.points = o3d.utility.Vector3dVector(ref_shape.squeeze())
        pc_ref.points = o3d.utility.Vector3dVector(ref_shape.squeeze() + [0.3, 0, 0])
        print("visualize the reference shape:please close the window to continue!")
        o3d.visualization.draw_geometries([pc_ref])
        ref_shape_weight = ssm.get_theta(ref_shape.squeeze(), n_modes=components_number)
        # ref_shape_weight = ref_shape_weight.transpose(1, 0)
        # print("ref_shape_weight shape:", ref_shape_weight.shape)

        samples = ssm.generate_similar_samples(n_samples = n_samples, n_modes=components_number, ref_shape_weight=ref_shape_weight)
        print("samples shape", samples.shape)

        pc_points = []
        # pc_points.append(pc_ref_bk)
        # Visualize the samples
        for i in range(n_samples):
            # you can save point cloud here
            pc_folder= augmented_data+"/"+corr_name
            if not os.path.exists(pc_folder):
                os.makedirs(pc_folder)
            file_name = augmented_data+"/"+corr_name+"/"+str(corr_name)+"_"+str(i)+".pcd"
            print("file_name:", file_name)
            save_point_cloud(samples[i].reshape(-1, 3), file_name)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(samples[i].reshape(-1, 3) + [0.5*(i+1),0, 0])
            pc_points.append(pc)
        o3d.visualization.draw_geometries(pc_points)

        if idx == 0:
            break








    #
    # output_data_root = "./bottle_total"
    # output_data_path = "./bottle_total/output"
    # corr_path = "./bottle_ssm"
    # exp_temp_data = "./bottle_samples"
    # corr_file_name = "bottle2048.npy"
    # corrVertsPath = os.path.join(corr_path, corr_file_name)
    # if not os.path.exists(corrVertsPath):
    #     saveCorrespondence(corr_path, output_data_root, corr_file_name)
    # # device = "cuda" if torch.cuda.is_available() else "cpu"
    # correspondence_path = corrVertsPath
    # corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
    # raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
    # print("raw_data_matrix shape:", raw_data_matrix.shape)
    # num_shapes = raw_data_matrix.shape[0]
    # ssm = SSM(np.transpose(corresponded_train_verts, (1, 0)))
    #
    # # Number of components for 99% of total variance
    # print("Number of components for 99% of total variance: ", ssm.reuired_components_number)
    # # Generate similar samples from the SSM
    # n_samples = 10
    #
    # ref_shape_id = 3
    # if ref_shape_id == -1:
    #     # randomly select a shape as the reference shape
    #     ref_shape_id = np.random.randint(0, num_shapes)
    #
    #
    # ref_shape = raw_data_matrix[ref_shape_id]
    # ref_shape = ref_shape.reshape(1, -1, 3)
    # # visualize the reference shape
    # pc_ref = o3d.geometry.PointCloud()
    # pc_ref_bk = o3d.geometry.PointCloud()
    # pc_ref_bk.points = o3d.utility.Vector3dVector(ref_shape.squeeze())
    # pc_ref.points = o3d.utility.Vector3dVector(ref_shape.squeeze() + [0.1, 0, 0])
    # # o3d.visualization.draw_geometries([pc_ref])
    # ref_shape_weight = ssm.get_theta(ref_shape.squeeze(), n_modes=ssm.reuired_components_number)
    # # ref_shape_weight = ref_shape_weight.transpose(1, 0)
    # print("ref_shape_weight shape:", ref_shape_weight.shape)
    #
    # # reconstruction = ssm.theta_to_shape_norm(ref_shape_weight, n_modes=ssm.reuired_components_number)
    # # reconstruction = reconstruction.reshape(1, -1, 3)
    # # # visualize the reconstruction
    # # pc= o3d.geometry.PointCloud()
    # # pc.points = o3d.utility.Vector3dVector(reconstruction.squeeze())
    # # o3d.visualization.draw_geometries([pc_ref, pc])
    # # samples_new = ssm.generate_random_samples(n_samples = n_samples,  n_modes=ssm.reuired_components_number)
    # # exit(0)
    #
    # samples = ssm.generate_similar_samples(n_samples = n_samples, n_modes=components_number, ref_shape_weight=ref_shape_weight)
    # print("samples shape", samples.shape)
    #
    # pc_points = []
    # pc_points.append(pc_ref_bk)
    # # Visualize the samples
    # for i in range(n_samples):
    #     # you can save point cloud here
    #     save_point_cloud(samples[i].reshape(-1, 3), exp_temp_data+"/sample"+str(i)+".pcd")
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(samples[i].reshape(-1, 3) + [0.1*(i+1),0, 0])
    #     pc_points.append(pc)
    # o3d.visualization.draw_geometries(pc_points)
    #
    #
    #
    # # Check the mean shape
    # # mean_chair_pointCloud = ssm.mean.reshape(1, -1, 3)
    # # mean_chair_pointCloud = mean_chair_pointCloud.squeeze()
    # # print("mean_bottle_pointCloud shape", mean_chair_pointCloud.shape)
    # # print("mean_bottle_pointCloud len", len(mean_chair_pointCloud))
    # # save_point_cloud(mean_chair_pointCloud, "mean_bottle_pointCloud.pcd")
    #


