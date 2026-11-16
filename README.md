# LeafLens — offline crop-leaf health triage

**What it is:** Point a camera at a leaf, get a careful triage with a
confidence — **NOT a definitive diagnosis**. Three classes only:
healthy, olive peacock spot (*Spilocaea oleagina*), and olive bud
mite damage (*Aculus olearius*).

**What it is not:** a diagnosis, a spray prescription, or field
accuracy. It never gives a pesticide dose. If the result is uncertain,
or if it says a disease is possible, the advice is always the same:
check other leaves and ask an agronomist before spraying anything.

## How to run

```bash
cd ~/workspace/projects/leaflens
pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --break-system-packages -r requirements.txt

# 1. Get the data (already done here; see organize_data.py)
#    GitHub mirror of the Denizli olive set -> data/olive_extracted
#    -> canonical folders in data/olive/
python3 organize_data.py

# 2. Train the frozen-backbone baseline (12 epochs max, early stopping)
python3 -u train.py --data data/olive --epochs 12 --mode frozen

# 3. Fine-tune from the frozen checkpoint
python3 -u train.py --data data/olive --epochs 12 --mode finetuned \
    --init checkpoints/frozen_best.pt

# 4. Evaluate on the held-out test split (never used for training
#    or checkpoint selection)
python3 evaluate.py --checkpoint checkpoints/finetuned_best.pt \
    --mode finetuned --gradcam-out results/gradcam_example.png

# 5. Predict one photo
python3 predict.py <image.jpg>
python3 predict.py <image.jpg> --lang fr

# 6. Tests
python3 test_leaflens.py
```

All commands assume you are in this folder (split paths are relative
to it; the dataset also resolves them against the code location).

## Model

MobileNetV2, ImageNet-pretrained, with a small head (dropout 0.2 +
linear layer to 3 classes), image size 128, augmentation on train
only (horizontal flip, rotation up to 20°, brightness/contrast
jitter — no unrealistic distortion). Class-weighted cross-entropy
loss handles the imbalance (peacock spot is the largest class).
Two models are compared, not just the best one: a frozen backbone
(head only) and a fine-tuned model (all weights, low learning rate
1e-4) started from the frozen checkpoint. Best checkpoint is picked
by validation loss, with early stopping (patience 4).

## Results (summary — full numbers in RESULTS.md)

Held-out test set, n = 509, same for both models:
frozen backbone accuracy 0.8723 (macro-F1 0.8714);
**fine-tuned accuracy 0.9607 (macro-F1 0.9611)**.
Fine-tuned confusion: healthy 155/157, peacock 205/219 (10 called
healthy — the main, dangerous confusion), aculus 129/133.
Only 8/509 test images fell under the 0.60 uncertain rule.
These are Denizli lab-style photos, not field accuracy.

## Data

Real field photos from Denizli, Turkey (spring/summer), published
as the 3,400-image olive dataset and mirrored openly on GitHub at
https://github.com/sinanuguz/CNN_olive_dataset — see RESULTS.md for
the full provenance, the archive layout, and the train3/train4
labelling note. The archive's own train/test folders are NOT used as
the split here: all 3,400 images were pooled into canonical class
folders and re-split stratified 70/15/15 with seed 42, so the test
set reported here is a fresh held-out set, not the authors' original
test set, and results are not directly comparable to papers that
used the original split.

Kaggle copies of the same dataset exist (e.g. serhathoca/zeytin),
but no Kaggle credentials were available or used here; the GitHub
mirror was openly downloadable without auth, so Kaggle was not
attempted. Licence for the GitHub mirror was not stated in its
README; treat it as research use and credit the original Denizli
collection.

## The uncertain rule

If the top probability is below 0.60, `predict.py` outputs
**UNCERTAIN** instead of forcing a class. The threshold is a
judgement call, not a tuned optimum: 0.60 is high enough that a
barely-better-than-chance prediction (chance = 0.33 for 3 classes)
is never presented as an answer, and low enough that a clearly
favoured class is still reported. RESULTS.md shows how many test
images fall under it and how accurate the remaining confident
predictions are, so you can see the trade-off instead of taking
the threshold on faith.

## Honest limits (short version; full version in RESULTS.md)

- Lab-style photos: single leaves on plain backgrounds, one orchard
  region (Denizli), one season window. A phone photo in a real
  orchard — cluttered background, shade, different cultivar, a
  Tunisian orchard — is a domain shift this test set does not cover.
- Only 3 classes. Anything else (other diseases, nutrient problems,
  sunburn, insects) will be forced into one of these three or into
  UNCERTAIN; it cannot say "something else".
- Small, imbalanced classes, single seed, single split. Numbers in
  RESULTS.md are from this run only.
- Misdiagnosis risk is real: a wrong "healthy" could delay treatment,
  a wrong "disease" could cause an unnecessary spray. That is why
  the tool says triage, shows confidence, and defers to an agronomist.

## 10-minute explain-back outline (for Elia)

1. What are the 3 classes, and where did the photos come from?
   (Denizli, Turkey; 3,400 field photos; not Tunisian orchards.)
2. How was the data split, and why is the test set trustworthy?
   (Stratified 70/15/15, seed 42; test never trained on and never
   used to pick the checkpoint — validation picked it.)
3. What is MobileNetV2 transfer learning in one sentence?
   (A small network pre-trained on ImageNet, reused as a feature
   extractor, with a new 3-class head trained on olive leaves.)
4. What is the difference between the frozen and fine-tuned models,
   and which won, by how much? (See RESULTS.md — quote both.)
5. What does the confusion matrix say — which two classes get
   confused most, and why might that be?
6. What does the uncertain rule do, and at what threshold?
   (Below 0.60 top probability -> UNCERTAIN, check other leaves,
   ask an agronomist.)
7. What does this NOT prove? (Field accuracy, other cultivars /
   regions, other diseases, a diagnosis, a spray dose.)
8. Show it: run `predict.py` on a held-out test image and on a
   non-leaf/bad path, and explain both outputs.

## Files

- `organize_data.py` — archive -> canonical class folders, with
  exact-count checks.
- `data.py` — discovery, stratified split, class weights, transforms.
- `model.py` — MobileNetV2 + head, frozen/finetuned modes, Grad-CAM.
- `train.py` — training loop, early stopping, best checkpoint.
- `evaluate.py` — test metrics, confusion matrix, confidence bands,
  uncertain rule, Grad-CAM example.
- `predict.py` — CLI triage with disclaimer, EN/FR wording.
- `test_leaflens.py` — data counts, model shape, predict.py,
  uncertain handling, bad-path handling.
- `results/` — split.csv, training histories, evaluation JSON/CSV,
  training logs.
- `checkpoints/` — best checkpoints (frozen, finetuned).
- `demo.html` — simple offline demo page (see RESULTS.md for its
  example predictions; it runs nothing by itself, it documents
  how to use predict.py).
