# LeafLens — results from actual runs (2026-10-24)

Every number below comes from a log or JSON file in `results/`
produced by a command listed here. Nothing is estimated, rounded
beyond the decimals shown, or copied from a paper.

## Dataset — what was actually used

- **Source:** GitHub mirror of the Denizli olive-leaf dataset,
  https://github.com/sinanuguz/CNN_olive_dataset
  (cloned 2026-10-24 into `data/olive_github`; images shipped as
  5 `.rar` files, extracted with `unrar` to `data/olive_extracted`,
  105 MB, 3,400 JPEGs).
- **Collection (as described in the mirror README and the papers
  that used this set):** olive leaves collected in Denizli, Turkey,
  in spring/summer, labelled with agricultural-engineer help.
  Licence was **not stated** in the mirror README — research use,
  credit the original collection; do not assume a commercial licence.
- **Kaggle:** copies of the same 3,400-image set exist on Kaggle
  (e.g. `serhathoca/zeytin`), but no Kaggle credentials were
  available or used here, and Kaggle was not attempted — the GitHub
  mirror was openly downloadable without auth, so step 1 of the
  dataset plan was already satisfied by a GitHub-hosted copy.
  No PlantVillage fallback was needed, and no PlantVillage class
  is presented as an olive class anywhere here.

### Archive layout — and the train3/train4 labelling note

| Archive file | Contents in the archive | Assigned class |
|---|---|---|
| Train1.rar | `Healthy/` | healthy — 830 |
| Train2.rar | `aculus_olearius/` | aculus_olearius — 690 |
| test.rar | `test/Healthy`, `test/aculus_olearius`, `test/olive_peacock_spot` | healthy 220, aculus 200, peacock 260 |
| train3.rar + train4.rar | `train3/` (423), `train4/` (777) — **no class folder** | olive_peacock_spot — 1,200 |

train3/train4 are assigned to peacock spot by elimination + exact
arithmetic + a visual spot-check (a train3 leaf shows dark peacock
spots; a Healthy leaf does not):
they are the only class left once Healthy and Aculus are accounted
for, 423 + 777 = 1,200 exactly matches the published peacock
training count, and the final totals then match the published
dataset exactly. `organize_data.py` fails unless those exact totals
are reproduced.

A second trap was found and fixed during organising: train and test
folders reuse filenames (e.g. `A163.jpg` in both), so plain
symlinks overwrote each other and silently lost 100 healthy images
(first count: 3,300, not 3,400). The fix prefixes every link with
its source folder (`test_Healthy__A163.jpg`, etc.), after which the
count is exact.

### Final counts and split

| Class | Total | Train | Val | Test |
|---|---|---|---|---|
| healthy | 1,050 | 735 | 158 | 157 |
| olive_peacock_spot | 1,460 | 1,022 | 219 | 219 |
| aculus_olearius | 890 | 623 | 134 | 133 |
| **Total** | **3,400** | **2,380** | **511** | **509** |

- Split: stratified 70/15/15, seed 42, written to `results/split.csv`.
- **The archive's own train/test folders were NOT used as the
  split.** All 3,400 images were pooled and re-split, so the test
  set here is a fresh held-out set (untouched by training and by
  checkpoint selection), but results are NOT directly comparable
  to papers that used the original split.
- Class imbalance: peacock spot is 42.9% of the data, aculus
  26.2%, healthy 30.9%. Handled with class-weighted cross-entropy
  (train weights: healthy 1.079, peacock 0.776, aculus 1.273) plus
  the stratified split. No oversampling was used.

## Training

- Hardware/software: CPU only, 2 threads, PyTorch 2.14.1+cpu,
  torchvision 0.29.1+cpu, scikit-learn 1.9.1.
- Model: MobileNetV2, ImageNet-pretrained (weights downloaded from
  pytorch.org, file `mobilenet_v2-7ebf99e0.pth`), head = dropout
  0.2 + linear to 3 classes. Image size 128. Augmentation on train
  only: horizontal flip, rotation ≤20°, brightness/contrast jitter.
  AdamW, cosine schedule, batch 32.
- Checkpoint selection: best by **validation loss**, early
  stopping patience 4. Test was never used for selection.

### Frozen backbone (head only, LR 1e-3) — `results/history_frozen.json`

12 epochs, 1,477.5 s (~24.6 min). Best val loss 0.3079 at epoch 8
(val acc 0.8865 at that epoch). Full history:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0.7192 | 0.7122 | 0.4697 | 0.8552 |
| 2 | 0.4951 | 0.8235 | 0.4306 | 0.8571 |
| 3 | 0.4388 | 0.8315 | 0.3626 | 0.8845 |
| 4 | 0.4048 | 0.8487 | 0.3428 | 0.8787 |
| 5 | 0.3998 | 0.8487 | 0.3495 | 0.8748 |
| 6 | 0.3808 | 0.8563 | 0.3239 | 0.8845 |
| 7 | 0.3615 | 0.8651 | 0.3559 | 0.8689 |
| 8 | 0.3705 | 0.8584 | **0.3079** | **0.8865** |
| 9 | 0.3705 | 0.8592 | 0.3238 | 0.8787 |
| 10 | 0.3527 | 0.8647 | 0.3203 | 0.8787 |
| 11 | 0.3515 | 0.8634 | 0.3231 | 0.8787 |
| 12 | 0.3580 | 0.8651 | 0.3244 | 0.8826 |

### Fine-tuned (all weights, LR 1e-4, from frozen best) — `results/history_finetuned.json`

12 epochs, 2,589.7 s (~43.2 min). Best val loss 0.0772 at epoch 11
(val acc 0.9765 at that epoch). Full history:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0.3021 | 0.8824 | 0.2229 | 0.9276 |
| 2 | 0.2253 | 0.9147 | 0.1626 | 0.9432 |
| 3 | 0.1751 | 0.9294 | 0.1312 | 0.9472 |
| 4 | 0.1717 | 0.9353 | 0.1217 | 0.9609 |
| 5 | 0.1382 | 0.9462 | 0.1094 | 0.9628 |
| 6 | 0.1154 | 0.9555 | 0.1127 | 0.9667 |
| 7 | 0.1051 | 0.9601 | 0.1005 | 0.9706 |
| 8 | 0.1102 | 0.9592 | 0.0886 | 0.9706 |
| 9 | 0.0930 | 0.9681 | 0.0914 | 0.9746 |
| 10 | 0.0919 | 0.9672 | 0.0802 | 0.9804 |
| 11 | 0.0887 | 0.9647 | **0.0772** | **0.9765** |
| 12 | 0.0728 | 0.9668 | 0.0930 | 0.9726 |

### Interruption, honestly recorded

The first fine-tuning attempt was killed after epoch 1 by a VM
replacement (~20:36 UTC): the process disappeared, the system
PyTorch install was wiped, and only home-folder files survived
(checkpoints, data, logs). Its epoch-1 result (val loss 0.2327,
val acc 0.9198) is preserved as
`checkpoints/finetuned_attempt1_epoch1.pt` and
`results/train_finetuned_attempt1.log`, and is **not** the model
evaluated below. Torch was reinstalled (same version 2.14.1+cpu)
and fine-tuning was restarted cleanly from `frozen_best.pt`; the
history above is the restarted run only. The restarted epoch 1
(val loss 0.2229, val acc 0.9276) differs slightly from attempt 1
because training augmentation is random — that variation is normal
and is not hidden.

### Model size and inference

- Checkpoint size: 9.2 MB each (`frozen_best.pt`,
  `finetuned_best.pt`).
- Mean inference on the test set (fine-tuned model, CPU, batch 32):
  **36.72 ms per image** (from `evaluation_finetuned.json`).
  Single-image CLI latency will be higher because it loads the
  model each time; that was not measured and is not claimed.

## Evaluation — held-out test set, n = 509

Run: `python3 evaluate.py --checkpoint checkpoints/<model>_best.pt`
— see `results/evaluation_<mode>.json` and
`results/confusion_<mode>.csv`.

### Headline comparison (both models, same test set)

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Frozen backbone | 0.8723 | 0.8714 |
| **Fine-tuned** | **0.9607** | **0.9611** |

Fine-tuning gains +0.0884 accuracy over the frozen baseline. Both
numbers are reported; the fine-tuned number alone would hide how
much of the result came from pretraining.

### Fine-tuned, per class

| True class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.9281 | 0.9873 | 0.9568 | 157 |
| olive_peacock_spot | 0.9856 | 0.9361 | 0.9602 | 219 |
| aculus_olearius | 0.9627 | 0.9699 | 0.9663 | 133 |

Confusion matrix (rows = true, columns = predicted):

|  | healthy | peacock | aculus |
|---|---|---|---|
| healthy | 155 | 1 | 1 |
| olive_peacock_spot | 10 | 205 | 4 |
| aculus_olearius | 2 | 2 | 129 |

The main confusion is peacock spot → healthy (10 of 219 peacock
leaves called healthy). That is the dangerous direction: a missed
disease, not a false alarm. It is also why the tool must say
"triage, check other leaves" rather than "healthy, done".

### Frozen, per class (for comparison)

| True class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.8122 | 0.9363 | 0.8698 | 157 |
| olive_peacock_spot | 0.9570 | 0.8128 | 0.8790 | 219 |
| aculus_olearius | 0.8380 | 0.8947 | 0.8655 | 133 |

Frozen confusion: healthy 147/157, peacock 178/219 (25 called
healthy, 16 called aculus), aculus 119/133. Fine-tuning mainly
fixed peacock recall (0.8128 → 0.9361).

### Confidence and the uncertain rule (fine-tuned)

Threshold: top probability < 0.60 → UNCERTAIN (judgement call, see
README; chance level is 0.33 for 3 classes).

| Confidence band | n | Accuracy |
|---|---|---|
| 0.90–1.00 | 446 | 0.9888 |
| 0.75–0.90 | 37 | 0.8649 |
| 0.60–0.75 | 18 | 0.5556 |
| < 0.60 | 8 | 0.7500 |

- Uncertain cases: **8 / 509 (1.6%)**.
- Accuracy on confident cases only: **0.9641** (vs 0.9607 overall).
  The uncertain rule removes very few cases here because the model
  is usually very confident — that is a property of this test set,
  not proof that it will be uncertain when it should be on new
  photos. Probabilities here are near 1.00 on most correct
  predictions (see demo below); calibration (whether 0.90 really
  means 90% right) was **not** measured, so confidence must not be
  read as a calibrated probability.

Frozen model, for comparison: 69 / 509 (13.6%) uncertain, confident
accuracy 0.9250 vs 0.8723 overall — the frozen model is less
confident and benefits more from the uncertain rule.

### Grad-CAM

`evaluate.py --gradcam-out results/gradcam_example.png` ran and
saved a real blended image (128×128): first disease-class test
image, true = aculus_olearius, predicted = aculus_olearius,
confidence 0.999. The overlay is **faint at 128 px input
resolution** and should be treated as illustrative, not as proof
of exactly which leaf region the model used. It is included
because it runs, not because it is a strong explanation.

### Demo predictions (fine-tuned model, held-out test images)

| Held-out image (true class) | Top prediction | Confidence |
|---|---|---|
| `aculus_olearius__104.jpg` (aculus) | aculus_olearius | 0.999 |
| `Healthy__A667.JPG` (healthy) | healthy | 1.000 |
| `test_olive_peacock_spot__DSC_0092.JPG` (peacock) | olive_peacock_spot | 1.000 |

French wording verified: the same healthy image with `--lang fr`
gives "Semble sain sur cette photo…" plus a French disclaimer.
(The disclaimer was originally English-only and was fixed after
the first demo run; the fix was re-tested.)

## Tests

`python3 test_leaflens.py` → **ALL TESTS PASSED** (re-run after the
French-disclaimer fix):
- dataset totals: 3,400 (1,050 / 890 / 1,460)
- test split held out, n = 509, no overlap with train
- model output shape (4, 3)
- `predict.py` on a held-out image: exit 0, triage + disclaimer
- uncertain handling at threshold 1.01: UNCERTAIN
- bad image path: exit 2, clean ERROR, no traceback

## What this does NOT prove

- **Field accuracy.** Test photos are single leaves on plain
  backgrounds from one Denizli orchard collection, spring/summer.
  A phone photo in a real orchard — cluttered background, shade,
  wind, a different phone camera — is untested.
- **Cultivar / region shift.** Nothing here tests Tunisian olive
  varieties, other Turkish cultivars, or other climates. A model
  that scores 0.9607 on Denizli photos can score much lower
  elsewhere; that gap was not measured.
- **Other conditions.** Only 3 classes exist. Other diseases,
  nutrient deficiencies, sunburn, or insect damage will be forced
  into one of these three classes or into UNCERTAIN. It cannot
  say "something else", and it cannot diagnose anything outside
  its training labels.
- **A diagnosis or a treatment.** This is triage. A wrong
  "healthy" (10 peacock leaves in the test set) could delay
  treatment; a wrong "disease" could cause an unnecessary spray.
  No pesticide, dose, or spray timing is given, by design.
- **Calibration.** Confidence values are model softmax outputs,
  not calibrated probabilities. The 0.60 uncertain threshold is a
  judgement call, not a tuned optimum.
- **Generalisability from one split.** Single seed (42), single
  stratified split, no cross-validation. A different split would
  give slightly different numbers; no error bars are claimed.
- **Mastery / GitHub-readiness.** Built, trained, evaluated, and
  tested only. It is **NOT mastered by Elia and NOT GitHub-ready**
  until he can explain it out loud using the outline in README.md,
  including the peacock→healthy confusion and the limits above.
