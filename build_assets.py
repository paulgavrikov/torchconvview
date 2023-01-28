import torchvision
from torchconvview import plot_conv, plot_conv_rgb, PCAView
import matplotlib.pyplot as plt

model = torchvision.models.resnet18(weights="default")
fig, _ = plot_conv_rgb(model.conv1.weight)
fig.savefig("docs/fig/output_plot_conv_rgb.png", dpi=100, bbox_inches="tight", pad_inches=0)
plt.close()
fig, _ = plot_conv(model.layer1[1].conv2.weight)
fig.savefig("docs/fig/output_plot_conv.png", dpi=100, bbox_inches="tight", pad_inches=0)
plt.close()
pca = PCAView(model.layer4[1].conv2.weight)
fig, _ = pca.plot_conv()
fig.savefig("docs/fig/output_pcaview_plot_conv.png", dpi=100, bbox_inches="tight", pad_inches=0)
plt.close()
fig, _ = pca.plot_variance_ratio()
fig.savefig("docs/fig/output_pcaview_plot_variance_ratio.png", dpi=100, bbox_inches="tight", pad_inches=0)
plt.close()