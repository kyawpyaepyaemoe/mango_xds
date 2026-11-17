# Explainable Mango Leaf Disease Classification

**Does Grad-CAM tell a field officer the truth about a ResNet-50?**

Group project for **AST02.903 Explainable Data Science**, Asian Institute of Technology (Fall 2026). Instructor: Dr Sein Minn.

| Name | Student ID |
|---|---|
| Moe Pyae Pyae Kyaw | st127333 |
| Win Myint @ Htet Wai Yan Htut | st125326 |
| Nyo Lae Hnin | st127225 |
| Swel Myat Sandi Min Thein | st127401 |
| Kunatip Eiampan | st126907 |

---

## TL;DR

- A ResNet-50 classifies mango leaf photos into 8 classes with **100% test accuracy** on clean photos, after removing duplicate leaves from the test set. An interpretable logistic regression on 27 readable features reaches **96.5%**.
- Clean accuracy hides real risks:
  - The model partly reads the **photo background**: the background alone predicts the class with 70.8% accuracy.
  - Under simulated farmer-photo conditions it **misses 13.2% of high-priority diseases**.
  - Its accept flag still **accepts 72% of leaves from other plants**.
- **Grad-CAM is only partly trustworthy:**
  - It depends on the model, it is stable, and it finds a shortcut we planted (65% hit rate against 3.5% for an edge map).
  - On deletion tests it is **no more faithful than a plain edge detector**, and it does **not reveal the background shortcut** the model really uses.
- **Human study (8 people, 192 decisions):** people caught **21 of 48** wrong labels with the heat map and **23 of 48** without it. The heat map gave **no clear help**, and made people more confident in wrong labels they accepted.
- **Advice to the officer:** read the heat map as *"where to look first"*, never as the reason for a label. Trust the accept / retake flag more than the map.

## 1. The problem

A provincial agricultural **field officer** receives mango-leaf photos from farmers and decides **which farms to visit first this week**. For each photo the system shows:

1. a **disease label** (8 classes)
2. a **priority tier**: High / Moderate / Low / None
3. a **Grad-CAM heat map**
4. an **accept / retake** flag

| Mistake | Consequence |
|---|---|
| Wrong "Low" or "None" on a diseased leaf | An infected farm waits and the disease spreads |
| Wrong "High" on a healthy leaf | One visit is wasted |

The claim we test: *Grad-CAM heat maps are a trustworthy account of what this ResNet-50 uses to make its prediction.*

## 2. Data

**MangoLeafBD**: 4,000 photos (256×256 px), 500 per class, from four orchards in Bangladesh. The notebook downloads it from Kaggle (`warcoder/mango-leaf-disease-dataset`). The photos are **not** stored in this repository.

| Tier | Classes | Basis |
|---|---|---|
| High | Anthracnose, Bacterial Canker, Die Back, Powdery Mildew | Yield loss up to 100% (anthracnose), 20–25% (powdery mildew); severe yearly losses; whole-tree defoliation (die back) |
| Moderate | Gall Midge | Up to 60% of leaves damaged at peak, without killing tissue |
| Low | Sooty Mould, Cutting Weevil | Sooty mould does not infect tissue; the weevil is a minor pest |
| None | Healthy | |

Sources for the tiers are in the report. The assignment to tiers is our reading of them, not an agronomist's ranking.

Known data problems:

- Only about **1,800 distinct leaves**; the rest are zoomed or rotated copies.
- Each class was often photographed on the **same background**.
- There are **no lesion annotations**.

## 3. Repository contents

```
Mango_Leaf_Disease_Classification_v3.ipynb   main analysis (§1–§20), run on a Colab GPU, outputs included
requirements.txt                             pinned package versions
README.md                                    this file
```

When the notebook runs it writes its outputs to `MyDrive/XDS/mango_outputs_v2/` on Google Drive:

- `tables/`: 25 CSV tables
- `figs/`: figures
- `ckpt/`: model checkpoints (named with a hash of the training split)
- `annotation_kit/`: 40 images for hand-drawn lesion masks
- `human_study/`: forms, reference sheet and answer key

## 4. How to run

**Colab (recommended)**

1. Open the notebook in Colab.
2. Runtime → Change runtime type → **T4 GPU** → Run all.
3. The first cell installs packages and mounts Google Drive. Approve every permission in the pop-up.
4. A full run takes about 45–60 minutes on a T4, mostly training 3 seeds plus the planted-shortcut model. Saved checkpoints are reused on later runs.

**Local**

```bash
pip install -r requirements.txt
papermill Mango_Leaf_Disease_Classification_v3.ipynb out.ipynb
```

The config cell is tagged `parameters`, so papermill can override it.

**Options (environment variables)**

| Variable | Effect |
|---|---|
| `MANGO_QUICK=1` | Fast CPU smoke test on a subsample |
| `MANGO_ROOT` | Use a local copy of the dataset instead of downloading it |
| `MANGO_OOD_DIR` | Your own folder of non-mango leaf photos |
| `MANGO_INFER_DIR` | A folder of your own photos to run through the officer output (§20) |

**Seeds:** split 42; training seeds 0, 1, 2; every other random draw has a fixed seed listed in the notebook's seed registry.

## 5. Method

| § | Step | What it does |
|---|---|---|
| 1 | Data inventory | Leaf mask (Otsu), dark-pixel lesion proxy, background colour by class |
| 2 | **Leak-free split** | Rotation/flip-invariant ImageNet embeddings + filename timestamps → leaf groups; `StratifiedGroupKFold`; threshold sensitivity check |
| 3 | Interpretable baseline | Logistic regression on 27 named features; group-bootstrap sign stability |
| 4 | ResNet-50 | Fine-tuned from ImageNet, AdamW, one-cycle schedule, 12 epochs, 3 seeds |
| 5 | Triage evaluation | Missed-high rate, false alarms, cross-tier errors; sensitivity to the tier table |
| 6 | Grad-CAM | `layer4[-1]`, always compared with 4 **null maps**: random blobs, centre, image edges, leaf mask |
| 7 | Plausibility | Lesion enrichment, pointing game, off-leaf heat |
| 8 | **Faithfulness** | Deletion/insertion curves (2 baselines, paired bootstrap), cascading randomisation, background reliance, **planted-shortcut** test |
| 9 | Stability | Repeat runs, 3 seeds, small input changes; normalised against a control |
| 10 | Shortcuts and field shift | Background-only classifier, background swaps, simulated farmer photos |
| 11 | Non-mango inputs | Temperature-scaled confidence + kNN feature distance; accept flag |
| 12 | Actionability | Smallest set of photo fixes (background, lighting, focus, compression) that makes a photo usable |
| 13 | Officer output | Label, tier, confidence, retake advice, heat map captioned "where to look first" |
| 14 | Ethics | Licences, background proxy check, who could be harmed |
| 15 | Human study | Error-detection study with and without Grad-CAM |
| 17 | **Verdict table** | Built automatically from the computed numbers and fixed pass rules |

## 6. Results

| Claim | Evidence | Verdict |
|---|---|---|
| Split removes near-duplicates | 1,849 leaf groups; the first split leaked 64% of test; a random split inflates accuracy by ~5 points | Done |
| Complex model is worth it | 100% vs 96.5% clean; 97.2% vs 79.9% with a grey background | Yes, for robustness |
| Grad-CAM beats the edge map on deletion | −0.029 [−0.039, −0.019] | **No** |
| Grad-CAM beats the edge map on insertion | +0.068 [+0.058, +0.079] | Yes |
| Grad-CAM depends on the model | Correlation 0.11 after randomising the last layer | Yes |
| Grad-CAM finds a planted shortcut | Model flips 100% of stamped leaves; peak on the square 65% vs 3.5% for edges | Yes |
| Grad-CAM shows background reliance | Spearman 0.23 (edge map 0.11; pass rule 0.3) | **No** |
| Explanation stable across seeds | Normalised Spearman 0.84 | Yes |
| Explanation stable under small changes | Normalised Spearman at least 0.92 | Yes |
| Model does not rely on the background | 91.5% on other-class backgrounds vs 99.6% on same-class | **No** |
| High-priority diseases not missed in the field | 13.2% missed under combined conditions (0.0% clean) | **No** |
| Accept flag rejects other plants, keeps field photos | 72% of non-mango and 81% of field mango photos accepted | **No** |
| Retake advice is actionable | 42% usable before any fix, 86% with one fix, 100% with all four | Yes, on simulated photos |
| Heat map helps people catch errors | 21/48 with vs 23/48 without; 2 of 8 people improved | **No clear help** |

### Human study

- **Design:** 8 participants × 24 hard photos = 192 decisions. The model's label was wrong on 12 and right on 12. Participants chose *Accept* or *Flag for expert review* and rated their confidence from 1 to 5. Two counterbalanced forms showed the heat map on opposite halves.
- **Success rule, fixed in advance:** errors caught rise by ≥10 points, false flags rise by less, and ≥5 of 8 people improve.

| Measure | With heat map | Without | Change |
|---|---|---|---|
| Wrong labels flagged | 21/48 | 23/48 | −4.2 pts |
| Right labels flagged (false flags) | 11/48 | 19/48 | −16.7 pts |
| Correct decisions | 58/96 | 52/96 | +6.2 pts |

With 8 people this is a pilot; we report counts and bootstrap ranges and claim no significance.

## 7. Limitations

- One dataset from four orchards in one country, one leaf per photo.
- Field conditions, background swaps and bad photos are **simulated**; no real farmer photos.
- The lesion proxy (dark pixels) has not been checked against hand-drawn masks.
- Non-mango rejection was tested on one dataset (PlantVillage).
- The human study is a pilot with 8 classmates. Its photos come from a separately trained copy of the same model (same recipe and seed, earlier run), which also scores 100% on clean test photos.

## 8. Ethics and data

- **MangoLeafBD**: Mendeley Data, DOI [10.17632/hxsnvwty3r.1](https://data.mendeley.com/datasets/hxsnvwty3r/1), licence **CC BY-NC 3.0** (non-commercial use with attribution).
- **PlantVillage**: original release licensed **CC0 1.0**; used for evaluation only and not redistributed here.
- The images contain no people and no personal data. Human-study participants are identified only as P1–P8.
- **Who could be harmed:** a false "Low" or "None" delays a visit to an infected farm. Farmers with older phones or poor light get more retake results; the officer should not rank those farms lower for that reason.
- The system is **decision support only**. The officer makes the final call.

## 9. Model card

| | |
|---|---|
| Model | ResNet-50 (23.5M parameters), fine-tuned from ImageNet; best checkpoint by validation macro-F1; 3 seeds |
| Intended use | Decision support for ranking farm visits from leaf photos; not an autonomous diagnosis |
| Training data | MangoLeafBD, 2,871 photos after a leak-free split by leaf |
| Evaluation | 100% on 566 clean test photos; 88.7% accuracy and 13.2% missed-high under combined field conditions |
| Not validated for | Real farmer photos, other countries or cultivars, non-mango plants |
| Explanation | Grad-CAM on `layer4[-1]`: "where to look first", not "why" |
| Confidence | Temperature-scaled (T = 0.589); a ranking score, not a probability, for real field photos |

## References

- Ahmed, S. I. et al. (2023). MangoLeafBD: a comprehensive image dataset to classify diseased and healthy mango leaves. *Data in Brief*, 47, 108941.
- Selvaraju, R. R. et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV*.
- Petsiuk, V., Das, A., Saenko, K. (2018). RISE: randomized input sampling for explanation of black-box models. *BMVC*.
- Adebayo, J. et al. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Adebayo, J. et al. (2022). Post hoc explanations may be ineffective for detecting unknown spurious correlation. *ICLR*.
- Hase, P., Bansal, M. (2020). Evaluating explainable AI: which algorithmic explanations help users predict model behavior? *ACL*.
- Sun, Y. et al. (2022). Out-of-distribution detection with deep nearest neighbors. *ICML*.
- Hughes, D. P., Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
