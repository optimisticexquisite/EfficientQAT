import torch

def save_layer_weights(model, layer_names, save_path="./layer_weights.pth"):
    layer_weights = {}
    for name, param in model.named_parameters():
        # Check if the parameter is one of the specified layers
        if any(layer_name in name for layer_name in layer_names):
            layer_weights[name] = param.data.clone().cpu()  # Save as a CPU tensor for portability
    # Save all weights to a single file
    torch.save(layer_weights, save_path)
    print(f"Weights saved to {save_path}")

def attach_hooks_for_dataset(model, layer_names):
    """
    Attaches hooks to specified layers to capture input and output tensors.
    """
    # Dictionary to store inputs and outputs for each layer
    layer_io = {name: {"inputs": [], "outputs": []} for name in layer_names}

    # Hook function to record inputs and outputs for each layer
    def hook(module, input, output, layer_name):
        layer_io[layer_name]["inputs"].append(input[0].detach().cpu())
        layer_io[layer_name]["outputs"].append(output.detach().cpu())

    # Attach hooks to the specified layers
    handles = []
    for name, layer in model.named_modules():
        if any(layer_name in name for layer_name in layer_names):
            handle = layer.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n))
            handles.append(handle)

    return layer_io, handles

