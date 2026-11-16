"""LeafLens prediction CLI.

Usage:
  python3 predict.py <image> [--checkpoint checkpoints/finetuned_best.pt]
  python3 predict.py <image> --lang fr

Output: top classes with probabilities, a triage wording, and a
disclaimer. If the top probability is below the uncertain threshold
(default 0.60), the output is 'uncertain' -- it does NOT force a class.
This is a triage aid, NOT a definitive diagnosis. Never prescribe a
pesticide dose. Check other leaves and ask an agronomist before
spraying anything.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from data import CLASSES, get_transforms
from model import build_model

TRIAGE = {
    "healthy": {
        "en": "Looks healthy in this photo. Keep monitoring other leaves.",
        "fr": "Semble sain sur cette photo. Continuez à surveiller les autres feuilles.",
    },
    "olive_peacock_spot": {
        "en": "Possible peacock spot (Spilocaea oleagina). Check other leaves "
              "for similar spots and ask an agronomist before spraying.",
        "fr": "Possible tache ocellée (Spilocaea oleagina). Vérifiez les autres "
              "feuilles et demandez conseil à un agronome avant tout traitement.",
    },
    "aculus_olearius": {
        "en": "Possible bud mite damage (Aculus olearius). Check buds and "
              "other leaves and ask an agronomist before spraying.",
        "fr": "Possibles dégâts d'acarien (Aculus olearius). Vérifiez les bourgeons "
              "et les autres feuilles et demandez conseil à un agronome "
              "avant tout traitement.",
    },
}

DISCLAIMER = {
    "en": (
        "DISCLAIMER: This is a triage aid, NOT a definitive diagnosis. "
        "It can be wrong, especially on photos unlike its training data. "
        "Check other leaves and ask an agronomist before spraying. "
        "No pesticide dose is given here."
    ),
    "fr": (
        "AVERTISSEMENT : Ceci est un outil de tri, PAS un diagnostic "
        "définitif. Il peut se tromper, surtout sur des photos "
        "différentes de ses données d'entraînement. Vérifiez les autres "
        "feuilles et demandez conseil à un agronome avant tout "
        "traitement. Aucune dose de pesticide n'est donnée ici."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint",
                    default="checkpoints/finetuned_best.pt")
    ap.add_argument("--mode", default="finetuned")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--lang", choices=["en", "fr"], default="en")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.is_file():
        print(f"ERROR: image not found: {path}", file=sys.stderr)
        return 2
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt} "
              "(run train.py first)", file=sys.stderr)
        return 2
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"ERROR: could not read image {path}: {e}",
              file=sys.stderr)
        return 2

    model, _ = build_model(mode=args.mode, pretrained=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    tf = get_transforms(args.image_size, augment=False)
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()),
                    key=lambda t: -t[1])
    print(f"Image: {path}")
    for cls, p in ranked:
        print(f"  {cls}: {p:.3f}")
    top_cls, top_p = ranked[0]
    if top_p < args.threshold:
        print(f"\nTriage: UNCERTAIN (top probability {top_p:.3f} "
              f"below {args.threshold}). Do not rely on this photo alone: "
              "photograph other leaves in good light and ask an agronomist.")
    else:
        print(f"\nTriage: {top_cls} "
              f"(confidence {top_p:.3f})\n  {TRIAGE[top_cls][args.lang]}")
    print(f"\n{DISCLAIMER[args.lang]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
