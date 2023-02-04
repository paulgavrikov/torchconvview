from .utils import plot_grid

class LayerHook():
    
    def __init__(self):
        self.storage = None
        self.hook_handle = None

    def pull(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        return self.storage

    def register_hook(self, module, store_input=True):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.storage = None
        def hook(model, inp, out):
            if store_input:
                self.storage = inp
            else:
                self.storage = out
        self.hook_handle = module.register_forward_hook(hook)


def _get_activations(model, x, layer):
    hook = LayerHook()
    hook.register_hook(layer, store_input=False)
    model(x)
    return hook.pull()


def plot_activations(model, x, layer, **kwargs):
    activations = _get_activations(model, x, layer).detach().numpy()
    plot_grid(activations, **kwargs)