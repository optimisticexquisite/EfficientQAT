import torch
layer_io = torch.load("layer_io_wikitext2.pth")
# print(layer_io["model.layers.28.self_attn.q_proj"]["inputs"][0].shape)
print(layer_io["model.layers.28.input_layernorm"]["outputs"][0].shape)
