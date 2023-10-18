from .utils import plot_grid


class LayerHook():
    
    def __init__(self):
        self.storage = None
        self.hook_handle = None

    def pull(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        data = self.storage
        self.storage = None
        return data

    def register_hook(self, module, store_input=True):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.storage = None
        def hook(_, inp, out):
            if store_input:
                self.storage = inp
            else:
                self.storage = out
        self.hook_handle = module.register_forward_hook(hook)


def _get_activations(model, x, layer, get_input=False):
    hook = LayerHook()
    hook.register_hook(module=layer, store_input=get_input)
    model(x)
    return hook.pull()


def plot_activations(model, x, layer, post_proc_fn=None, get_input=False, **kwargs):
    activations = _get_activations(model=model, x=x, layer=layer, get_input=get_input)
    if post_proc_fn is not None:
        activations = post_proc_fn(activations)
    elif get_input:
        activations = activations[0]  # inputs are stored as a tuple ...

    activations = activations.detach().cpu().numpy()
    plot_grid(activations, **kwargs)