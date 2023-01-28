from .view import plot_conv
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt


class PCAView:
    """
    A class for PCA-based visualizations of convolutional layers.
    """

    def __init__(self, weight: Union[np.ndarray, torch.Tensor]) -> None:
        if type(weight) != np.ndarray:
            weight = weight.detach().cpu().numpy()

        self.kernel_size = (weight.shape[2], weight.shape[3])

        self.pca = PCA()
        self.pca.fit(weight.reshape(-1, np.prod(self.kernel_size)))

    def plot_conv(self, img_scale: float=1) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the PCA components of the convolutional weight.
        """
        return plot_conv(self.pca.components_.reshape(*self.kernel_size, *self.kernel_size), img_scale)
    
    def plot_variance_ratio(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the explained variance ratio of the PCA components.
        """
        fig = plt.figure()
        
        plt.bar(np.arange(len(self.pca.explained_variance_ratio_)), self.pca.explained_variance_ratio_)
        plt.xlabel("PCA component")
        plt.ylabel("Explained variance ratio")

        return fig, plt.gca()
