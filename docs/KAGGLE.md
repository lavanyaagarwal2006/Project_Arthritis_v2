# Training on a free Kaggle GPU

A Mac is fine for tests, baselines and the web app. Deep-model cross-validation (5 folds x up to 40
epochs over ~10k crops) belongs on a GPU: on a Kaggle T4 a DenseNet121 CV run takes roughly 1-3 hours
instead of days. Kaggle gives ~30 free GPU hours a week. Sessions run up to 12 h.

## One-time setup
1. kaggle.com: sign in, verify your phone number (needed for GPU and internet access).
2. Create a Hugging Face **Read** token (huggingface.co/settings/tokens). Your account already has
   RAM-H1200 access.
3. New Notebook. In the right panel: **Session options → Accelerator: GPU T4 x2**, **Internet: on**.
   **Add-ons → Secrets → Add secret**: name `HF_TOKEN`, value = the token.
   The code must be pushed to GitHub first (the notebook clones it).

## Notebook cells (one per block)
```python
!git clone https://github.com/lavanyaagarwal2006/project_arthritis_v2.git /kaggle/working/pa
%cd /kaggle/working/pa
!pip install -q timm
```
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from huggingface_hub import snapshot_download
snapshot_download("TokyoTechMagicYang/RAM-H1200-v1", repo_type="dataset", local_dir="data/RAM-H1200-v1",
                  allow_patterns=["Metadata.xlsx", "SvdH_Scoring/*"])   # skips the unused segmentation set
```
```python
# folds.csv and joints_manifest.csv come from the repo, so folds are identical to the Mac runs.
# Only rebuild the manifest if it is missing; never re-make splits.
!test -f splits/joints_manifest.csv || PYTHONPATH=src python -m arthritis.manifest
!PYTHONPATH=src python scripts/speedtest.py --exp densenet121
```
```python
EXP = "densenet121_v2"          # or densenet121, densenet121_native, densenet121_v2_mt
!PYTHONPATH=src python -m arthritis.train --exp {EXP} --run-name {EXP}_cv
!PYTHONPATH=src python -m arthritis.evaluate results/{EXP}_cv
!cd results && zip -qr /kaggle/working/{EXP}_results.zip {EXP}_cv summary.csv
```

For a long run, click **Save Version → Save & Run All**. It runs in the background, and you can close
the tab. When it finishes, download `<EXP>_results.zip` from the version's **Output** tab.

## Bring results home
```bash
cd ~/projects/project-arthritis
unzip -o ~/Downloads/densenet121_v2_results.zip -d results/
git add results/ && git commit -m "exp: densenet121_v2 5-fold CV (Kaggle T4)" && git push
```
Results carry the git SHA of the code that produced them, so make sure the notebook cloned the commit
you meant to run (the run log prints it).

Never upload the dataset itself to a public Kaggle dataset: the licence is non-commercial share-alike,
and the gate exists for a reason. Downloading inside the notebook with your own token keeps that intact.
