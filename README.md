* [ ] 

# ATML PA1 — Beyond IID

Four tasks on distribution shift: inductive biases under controlled image interventions (Task 1),
unsupervised domain adaptation on PACS (Task 2), domain generalization to an unseen PACS domain
(Task 3), and open-set recognition on CIFAR-10 / CIFAR-100 (Task 4).

Every reported number traces to a CSV or JSON under `taskN/modal_run/outputs/`. The executed notebook
for each task is committed alongside its results, so each figure and table can be traced back to the
cell that produced it.

---

## Layout

```
pa1/
  task1/  task1.ipynb   run_task1_modal.py   modal_run/{task1_executed.ipynb, outputs/}
  task2/  task2.ipynb   run_task2_modal.py   modal_run/{task2_executed.ipynb, outputs/}
  task3/  task3.ipynb   run_task3_modal.py   modal_run/{task3_executed.ipynb, outputs/}
  task4/  task4.ipynb   run_task4_modal.py   modal_run/{task4_executed.ipynb, outputs/}
  shared/               PACS split indices + the Task 2 ERM checkpoint history reused by Task 3
  report/               NeurIPS source, generated tables, staged figures
  report_master.md      implementation record: every free choice and every deviation, with reasoning
  requirements.txt      environment specification
```

Datasets, checkpoints and cached logits are **not** committed (see `.gitignore`); they are regenerated
by the notebooks.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The recorded runs used **torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, Python 3.11** on an NVIDIA RTX
PRO 6000 (Blackwell, compute capability 12.0). Blackwell requires the `cu128` wheels — the default
PyPI builds will not launch kernels on `sm_120`. Tasks 2 and 3 print this attestation into their
executed notebooks.

The notebooks run unchanged on CPU, but far more slowly.

---

## Reproducing each task

Each notebook is self-contained: it downloads its own data into `./data`, writes every result to
`./outputs/<task>/`, and needs no arguments. Run them either locally or on Modal.

### Locally

```bash
cd task1 && jupyter nbconvert --to notebook --execute task1.ipynb --output executed.ipynb
```

…and the same for `task2`, `task3`, `task4`.

### On Modal (how the reported runs were produced)

```bash
pip install modal && modal setup
modal run --detach task1/run_task1_modal.py     # ~8 min
modal run --detach task2/run_task2_modal.py     # ~13 min
modal run --detach task3/run_task3_modal.py     # ~7 min
modal run --detach task4/run_task4_modal.py     # ~26 min
```

Each runner bakes its notebook into the image (`copy=True`) so a detached run does not depend on the
local machine staying up, executes it with papermill, and commits results to a Modal Volume in a
`finally` block so a failed run still preserves partial output. Results are fetched with
`modal volume get`.

The runners expose two parameter sets: `SPEC_PARAMETERS` (empty — the notebook's own values, which are
the ones the manual fixes) and `SCALED_PARAMETERS`. **Every reported run used `SPEC_PARAMETERS`.**

### Ordering constraint

**Task 2 must run before Task 3.** Task 3 reuses Task 2's source-only ERM checkpoint as its baseline
rather than retraining it (`run_configs.json` records `retrained: false`), and both read the same PACS
split file, `shared/pacs_splits_seed6304.json`. Tasks 1 and 4 are independent and can run in any order.

---

## Protocol notes

A single seed, **6304**, fixes every split, initialisation, loader order and sampling decision. cuDNN
determinism flags are not set, so runs are not bitwise reproducible; see `report_master.md` §2 for a
measured scale of that nondeterminism.

Target-domain and unknown-class data are held behind an explicit lock in each notebook: PACS Sketch
labels are withheld until after every Task 2/3 checkpoint is selected, and CIFAR-100 is not loaded
until after Task 4's thresholds are fixed. Checkpoint selection uses source-validation metrics only.

---

## Attribution

Materially reused external code and assets:

- **AdaIN weights** (Task 1) — the pretrained normalised-VGG encoder and decoder released with
  [`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN), the reference
  reimplementation of Huang & Belongie (2017), obtained from a HuggingFace mirror. The architectures
  are transcribed to match those checkpoints layer-for-layer; the decoder checkpoint stores its weights
  under a `net.` prefix, which is stripped on load.
- **PROSER** (Task 4) — loss construction and detection score follow Zhou et al. (2021) and the
  authors' implementation at [`LAMDA-CL/CVPR21-Proser`](https://github.com/zhoudw-zdw/CVPR21-Proser).
- **Datasets** — PACS via the `flwrlabs/pacs` mirror; CIFAR-10/100 via the `uoft-cs` mirrors (pinned by
  dataset commit); STL-10 via torchvision.
- **Report template** — official NeurIPS 2026 style files from `neurips.cc`.

No other external code is materially reused. The cue-conflict generation and rejection rule, the MMD
implementation, the gradient-reversal layer, the CDAN conditioning, the SAM step, the PROSER training
loop and all evaluation code were written for this assignment.

Deviations from the assignment's prescribed protocol — gradient clipping, the unbiased MMD estimator,
L2 normalisation of the discriminator input, and the data-source change for CIFAR — are each recorded
with their reasoning in `report_master.md` §9.
