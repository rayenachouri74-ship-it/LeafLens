"""LeafLens model: MobileNetV2 (ImageNet-pretrained when available)
with a small classification head.

Two modes, used for the baseline comparison in train.py:
  - frozen: backbone weights frozen, only the head trains.
  - finetuned: the whole network trains at a lower learning rate.
If pretrained weights cannot be downloaded, the same architecture
trains from scratch and train.py records that fact in history.json
-- it never silently pretends pretrained weights were used.
"""
import torch
import torch.nn as nn
from torchvision import models

from data import CLASSES


def build_model(num_classes=len(CLASSES), mode="frozen",
                pretrained=True):
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        except Exception:
            weights = None
    try:
        net = models.mobilenet_v2(weights=weights)
        used_pretrained = weights is not None
    except Exception as e:
        # E.g. no network for the weight download: fall back to scratch.
        print(f"WARNING: pretrained weights unavailable ({e}); "
              "training from scratch.")
        net = models.mobilenet_v2(weights=None)
        used_pretrained = False
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    if mode == "frozen":
        for name, param in net.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
    return net, used_pretrained


def grad_cam(model, image_tensor, target_class, device="cpu"):
    """Simple Grad-CAM on the last convolutional feature map.

    Returns a 2-D numpy array in [0, 1] at the input resolution, or
    None if anything about the hook setup fails (caller must then
    omit the heatmap rather than fake one).
    """
    try:
        import numpy as np
        model.eval()
        activations, gradients = [], []

        def fwd_hook(_m, _i, out):
            activations.append(out.detach())

        def bwd_hook(_m, gi, go):
            gradients.append(go[0].detach())

        last_conv = model.features[-1]
        h1 = last_conv.register_forward_hook(fwd_hook)
        h2 = last_conv.register_full_backward_hook(bwd_hook)
        try:
            x = image_tensor.unsqueeze(0).to(device)
            x.requires_grad_(True)
            logits = model(x)
            score = logits[0, target_class]
            model.zero_grad()
            score.backward()
        finally:
            h1.remove()
            h2.remove()
        act = activations[0][0]          # C x H x W
        grad = gradients[0][0]           # C x H x W
        weights = grad.mean(dim=(1, 2))  # C
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.cpu().numpy()
        # Upsample to input size with PIL (no cv2 dependency).
        from PIL import Image
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        cam_img = Image.fromarray((cam * 255).astype("uint8"))
        cam_img = cam_img.resize((w, h), Image.BILINEAR)
        return np.asarray(cam_img, dtype=float) / 255.0
    except Exception as e:
        print(f"Grad-CAM failed ({e}); heatmap omitted.")
        return None
