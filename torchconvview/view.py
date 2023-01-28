from typing import Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_conv_rgb(weight: Union[np.ndarray, torch.Tensor], img_scale: float=1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a convolutional weight with RGB filters, as found in the first layer of a CNN. 
    Note, that this function assumes that the second dimension of the weight tensor is 3.

    :param weight: The weight tensor of the convolutional layer.
    :param img_scale: The scale of the image. Allows to resize the resulting plot.    
    :return: A tuple of the figure and the axis.
    """
    if type(weight) != np.ndarray:
        weight = weight.detach().cpu().numpy()

    assert weight.shape[1] == 3, "The second dimension of the weight tensor must be 3."

    filters = []
    for w in weight:
        f = w.transpose(1, 2, 0)
        f = 0.5 + 0.5 * f / np.abs(weight).max()
        filters.append(f)

    fig = plt.figure(figsize=np.asarray(weight.shape[:2]) * img_scale)

    plt.imshow(np.hstack(filters))
    plt.xticks(np.arange(-0.5, (weight.shape[0] - 1) * weight.shape[2], weight.shape[2]), [])
    plt.yticks([])
    for tic in plt.gca().xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    plt.grid(which="major", color="black", linestyle="-", linewidth=1)

    return fig, plt.gca()


def plot_conv(weight: Union[np.ndarray, torch.Tensor], img_scale: float=1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot all kernels of a convolutional weight.

    :param weight: The weight tensor of the convolutional layer.
    :param img_scale: The scale of the image. Allows to resize the resulting plot.    
    :return: A tuple of the figure and the axis.  
    """
    if type(weight) != np.ndarray:
        weight = weight.detach().cpu().numpy()
    
    t = abs(weight).max()
    
    fig = plt.figure(figsize=np.asarray(weight.shape[:2]) * img_scale)

    plt.imshow(np.hstack(np.hstack(weight.transpose(1, 0, 2, 3))), vmin=-t, vmax=t, cmap="seismic")
    plt.xticks(np.arange(-0.5, (weight.shape[0]-1) * weight.shape[2], weight.shape[2]), [])
    plt.yticks(np.arange(-0.5, (weight.shape[1]-1) * weight.shape[2], weight.shape[2]), [])
    for tic in plt.gca().xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    plt.grid(which="major", color="black", linestyle="-", linewidth=1)

    return fig, plt.gca()
