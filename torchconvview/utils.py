import matplotlib.pyplot as plt
import numpy as np


def plot_grid(data, img_scale=1):
    t = abs(data).max()
    
    fig = plt.figure(figsize=(data.shape[1] * img_scale, data.shape[0] * img_scale))

    plt.imshow(np.hstack(np.hstack(data)), vmin=-t, vmax=t, cmap="seismic")
    plt.xticks(np.arange(-0.5, (data.shape[1] - 1) * data.shape[2], data.shape[2]), [])
    plt.yticks(np.arange(-0.5, (data.shape[0] - 1) * data.shape[2], data.shape[2]), [])

    ax = plt.gca()
    ax.set_frame_on(False)

    for tic in plt.gca().xaxis.get_major_ticks():
       tic.tick1line.set_visible(False)
       tic.tick2line.set_visible(False)
    plt.grid(which="major", color="black", linestyle="-", linewidth=1)

    return fig, ax