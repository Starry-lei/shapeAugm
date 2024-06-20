import numpy as np
from sklearn.decomposition import PCA
from warnings import warn
from typing import Any, Tuple
import matplotlib.pyplot as plt
import h5py
def check_data_scale(dataset: np.ndarray) -> None:
    """Check that data is appropriate for PCA, meaning that each sample has 0
    mean and 1 std.

    Parameters
    ----------
    dataset : array_like
        N-dimension array of data to model, where each row on the first axis is one sample
        and each column on the second axis is a landmark value

    Returns
    -------
    None

    Raises
    ------
    Warning
        If mean of each sample in dataset not equal to 0
    Warning
        If standard deviation of each sample in dataset not equal to 1
    """
    if np.allclose(dataset.mean(axis=1), 0):
        pass
    else:
        warn("Dataset mean should be 0, " f"is equal to {dataset.mean(axis=1)}")

    if np.allclose(dataset.std(axis=1), 1):
        pass
    else:
        warn(
            "Dataset standard deviation should be 1, "
            f"is equal to {dataset.std(axis=1)}"
        )


class SSM:
    def __init__(self, correspondences: np.ndarray, pre_eigenval= None, preeigenvec=None, ssm_path= None ) -> None:
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:  Corresponded shapes
        """
        self.mean = np.mean(correspondences, 0)
        data_centered = correspondences - self.mean
        # data_centered = data_centered.transpose()
        cov_dual = np.matmul(data_centered.transpose(), data_centered) / (
            data_centered.shape[0] - 1
        )
       

        # ssm_name = ssm_path.split("/")[-1]

        evals, evectors = np.linalg.eigh(cov_dual)

        # if pre_eigenval is not None and preeigenvec is not None:
        #      # print("cov_dual shape", cov_dual.shape)
        #     evals, evectors = np.linalg.eigh(cov_dual)
        #     # save the eigenvalues and eigenvectors for later use   
        #     pre_eigenval = evals
        #     preeigenvec = evectors
        #     save_path = ssm_path

        #     np.save(save_path + ssm_name + "_pre_eigenval.npy", pre_eigenval)
        #     np.save(save_path + ssm_name + "_preeigenvec.npy", preeigenvec)
        # else:

        #     pre_eigenval= np.load(ssm_path + ssm_name + "_pre_eigenval.npy")
        #     preeigenvec = np.load(ssm_path + ssm_name + "_preeigenvec.npy")
        #     evals, evectors = pre_eigenval, preeigenvec



        # self.save_eigenvals = evals
        # self.save_eigenvecs = evectors



        evecs = evectors
        # Normalize the col-vectors
        evecs /= np.sqrt(np.sum(np.square(evecs), 0))

        # Sort
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]

        # Remove the last eigen pair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.variances[self.variances < 0] = 0
        self.modes_norm = evecs[:, :-1]


        variances = np.array(self.variances)

        self.eigenvals = variances

        # Calculate the total sum of variances
        total_variance = np.sum(variances)
        cumulative_variance = np.cumsum(variances)

        # Find the number of components for 99% of total variance
        required_mode_number = np.where(cumulative_variance >= 0.99 * total_variance)[0][0] + 1

        self.reuired_components_number = required_mode_number
        print("required_mode_number", required_mode_number)
        # self.modes_norm = self.modes_norm[:, :required_mode_number]
        self.variances_mode_number = self.variances[:required_mode_number]
        
        self.modes_scaled = np.multiply(self.modes_norm, np.sqrt(self.variances))


    def save_ssm(self, path: str, num_mode:int) -> None:
        """
        Save the SSM to a file
        Args: path
        """

        print("self.mean shape:", self.mean.shape)
        print("self.modes_norm shape:", self.modes_norm.shape)

        # print("self.variances_mode_number shape:", self.variances_mode_number.shape)


        np.savez(
            path,
            mean=self.mean,                  # mean shape
            modes_norm=self.modes_norm[:, :num_mode],      # eigenvectors
            variances=self.variances[:num_mode],        # eigenvalues
            variance_mode_number=self.variances_mode_number,
        )

    def save_ssm_compressed(self, path: str) -> None:
        """
        Save the SSM to a file
        Args: path
        """
        # np.savez(
        #     path,
        #     mean=self.mean,                  # mean shape
        #     modes_norm=self.modes_norm,      # eigenvectors
        #     modes_scaled=self.modes_scaled,
        #     variances=self.variances,        # eigenvalues
        #     variance_mode_number=self.variances_mode_number,
        # )
        np.savez_compressed(
            path,
            mean=self.mean,
            modes_norm=self.modes_norm,
            modes_scaled=self.modes_scaled,
            variances=self.variances,
            variance_mode_number=self.variances_mode_number,
            compression='lzma'  # Or 'bz2'
        )

    def save_ssm_h5(self, path: str) -> None:
        file_name = path + ".h5"
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('mean', data=self.mean)
            hf.create_dataset('modes_norm', data=self.modes_norm)
            hf.create_dataset('modes_scaled', data=self.modes_scaled)
            hf.create_dataset('variances', data=self.variances)
            hf.create_dataset('variance_mode_number', data=self.variances_mode_number)

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
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        # print("weights shape:", weights.shape)
        return weights


    def theta_to_shape(self, weights: np.ndarray,n_modes: int = None) -> np.ndarray:
        """
        Reconstruct shape from theta
        Args:
            weights:      weights of modes
        Returns:
            shape:      reconstructed shape
        """

        if n_modes:
            evecs = self.modes_scaled[:, :n_modes]
        else:
            evecs = self.modes_scaled
        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

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

        print("weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
        print("evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)

        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0) )
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

    def generate_random_samples(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
            print("self.modes_scaled", self.modes_scaled.shape)
        weights = np.random.standard_normal([n_samples, n_modes])
        print("weights shape", weights.shape)
        samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())
        return np.squeeze(samples)

    def generate_similar_samples(self, n_samples: int = 1, n_modes=None,  ref_shape_weight=None) -> np.ndarray:
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


        added_noise = np.random.normal(0, variation_coeff, (ref_shape_weight.shape[0], ref_shape_weight.shape[1], n_samples))
        # print("added_noise shape:", added_noise) #added_noise shape: (6, 1, 10)
        print("added_noise shape:", added_noise.shape) #added_noise shape: (6, 1)

        added_noise = (ref_shape_weight + np.transpose(added_noise, (2, 0, 1))).squeeze(2)

        print("added_noise shape:", added_noise.shape) #added_noise shape: (10, 6)

        print("self.modes_scaled shape:", evecs.shape) #self.modes_scaled shape: (6, 3)

        samples = self.mean + np.matmul(added_noise, evecs.transpose())



        # samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())

        return np.squeeze(samples)
    def get_reconstruction(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
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
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)

        print("weights shape:", weights.shape)
        print("weights:", weights)

        data_proj = self.mean + np.matmul(
            weights.transpose(1, 0), evecs.transpose(1, 0)
        )
        data_proj = data_proj.reshape(-1, 3)
        return data_proj

    def do_pca(
            self, dataset: np.ndarray, desired_variance: float = 0.9) -> Tuple[Any, int]:
        """Fit principal component analysis to given dataset.

        Parameters
        ----------
        dataset : array_like
            2D array of data to model, where each row on the first axis is one sample
            and each column on the second axis is e.g. shape or appearance for a landmark
        desired_variance : float
            Fraction of total variance to be described by the reduced-dimension model

        Returns
        -------
        pca : sklearn.decomposition._pca.PCA
            Object containing fitted PCA information e.g. components, explained variance
        required_mode_number : int
            Number of principal components needed to produce desired_variance

        Raises
        ------
        Warning
            If mean of each sample in dataset not equal to 0
        Warning
            If standard deviation of each sample in dataset not equal to 1
        """
        # self._check_data_scale(dataset)

        pca = PCA(svd_solver="auto")
        pca.fit(dataset)
        required_mode_number = np.where(
            np.cumsum(pca.explained_variance_ratio_) > desired_variance
        )[0][0]

        return pca, required_mode_number
