from torchconvview import plot_conv, plot_conv_rgb, PCAView
import torch


def test_plot_conv():
    conv = torch.nn.Conv2d(3, 32, 3)
    plot_conv(conv.weight)
    plot_conv_rgb(conv.weight.detach())


def test_plot_conv_rgb():
    conv = torch.nn.Conv2d(3, 32, 3)
    plot_conv_rgb(conv.weight)
    plot_conv_rgb(conv.weight.detach().numpy())


def test_plot_conv_rgb_assert():
    conv = torch.nn.Conv2d(6, 6, 3)
    try:
        plot_conv_rgb(conv.weight)
        assert False
    except AssertionError:
        pass

def test_PCAView():
    conv = torch.nn.Conv2d(3, 3, 3)
    pca_view = PCAView(conv.weight)
    pca_view.plot_conv()
    pca_view.plot_variance_ratio()

    pca_view = PCAView(conv.weight.detach().numpy())
    pca_view.plot_conv()
    pca_view.plot_variance_ratio()
