from torchconvview.featuremaps import LayerHook, _get_activations, plot_activations
import torchvision
import torch


def test_LayerHook_pull():
    hook = LayerHook()
    assert hook.storage is None
    hook.storage = 1
    assert hook.pull() == 1
    hook.storage = 2
    assert hook.pull() == 2


def test_LayerHook_register_hook():
    model = torchvision.models.resnet18(pretrained=False)
    
    hook = LayerHook()
    hook.register_hook(model.layer1[0].conv1)
    assert hook.hook_handle is not None
    assert hook.storage is None

    hook.hook_handle.remove()


def test_LayerHook_register_pull():
    model = torchvision.models.resnet18(pretrained=False)
    
    x = torch.randn(1, 3, 224, 224)

    hook = LayerHook()
    hook.register_hook(model.layer1[0].conv1, store_input=True)
    model(x)
    d = hook.pull()
    assert d is not None
    assert hook.hook_handle is None
    assert hook.storage is None

    del d

    hook.register_hook(model.layer1[0].conv1, store_input=False)
    model(x)
    d = hook.pull()
    assert d is not None
    assert hook.hook_handle is None
    assert hook.storage is None

    del d


def test__get_activations():
    model = torchvision.models.resnet18(pretrained=False)
    
    x = torch.randn(1, 3, 224, 224)

    activations = _get_activations(model, x, model.layer1[0].conv1, get_input=True)
    assert activations is not None
    del activations

    activations = _get_activations(model, x, model.layer1[0].conv1, get_input=False)
    assert activations is not None
    del activations


def test_plot_activations():
    model = torchvision.models.resnet18(pretrained=False)
    
    x = torch.randn(1, 3, 224, 224)

    plot_activations(model, x, model.layer1[0].conv1, get_input=True)

    plot_activations(model, x, model.layer1[0].conv1, post_proc_fn=lambda x: x[0] * 5, get_input=True)

    plot_activations(model, x, model.layer1[0].conv1, get_input=False)

    plot_activations(model, x, model.layer1[0].conv1, post_proc_fn=lambda x: x * 5, get_input=False)
