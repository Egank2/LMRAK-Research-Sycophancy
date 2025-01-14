class PathPatching:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def add_hook(self, layer_name, hook_fn):
        # Locate the layer
        layer = dict([*self.model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

# Example Usage
path_patching = PathPatching(model_handler.model)

def example_hook_fn(module, inputs, outputs):
    print(f"Hooked Layer: {module}, Outputs Shape: {outputs.shape}")

path_patching.add_hook("transformer.h.0", example_hook_fn)

# Test Path Patching
test_text = "The quick brown fox jumps over the lazy dog."
_ = model_handler.forward_pass(test_text)
path_patching.remove_hooks()