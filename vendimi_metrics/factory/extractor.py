import torch

def build_easy_model(device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    model.eval()
    model.to(device)

    return model