PY ?= python
export PYTHONPATH := src
EXP ?= densenet121
SCHEME ?= cv
DATA ?= data/RAM-H1200-v1
RUN ?=

.PHONY: inspect manifest manifest-jsn crop-stats splits train evaluate smoke smoke2 test ladder reproduce

inspect:            ## print the dataset layout and label counts (safe before the plan is frozen)
	$(PY) scripts/inspect_dataset.py $(DATA) --out results/data_inspection.txt

manifest:           ## one row per joint ROI, joined to patient/center metadata
	$(PY) -m arthritis.manifest

manifest-jsn:       ## joint-space-narrowing crops (auxiliary task for densenet121_v2_mt)
	$(PY) -m arthritis.manifest --score-type JSN

crop-stats:         ## crop sizes per joint + pixel-scale check across centers -> sets train.canvas
	$(PY) scripts/crop_stats.py

splits:             ## patient-grouped folds; the official test split is sealed
	$(PY) -m arthritis.splits

train:              ## make train EXP=densenet121 SCHEME=cv|loco
	$(PY) -m arthritis.train --exp $(EXP) --scheme $(SCHEME)

evaluate:           ## make evaluate RUN=results/<run_dir>
	$(PY) -m arthritis.evaluate $(RUN)

test:               ## unit tests (no GPU, no data)
	$(PY) -m pytest -q tests

smoke:              ## whole pipeline on synthetic data, including one 2-epoch torch run
	$(PY) scripts/make_synthetic.py data/synthetic
	$(PY) -m arthritis.manifest --exp configs/smoke.yaml
	$(PY) -m arthritis.splits --exp configs/smoke.yaml
	$(PY) -m arthritis.train --exp configs/smoke.yaml --smoke --run-name smoke_scratch
	$(PY) -m arthritis.evaluate results/smoke_scratch --n-boot 200

smoke2:             ## v2 path on synthetic data: native canvas, ordinal head, joint embedding, JSN multi-task
	$(PY) scripts/make_synthetic.py data/synthetic
	$(PY) -m arthritis.manifest --exp configs/smoke2.yaml
	$(PY) -m arthritis.manifest --exp configs/smoke2.yaml --score-type JSN
	$(PY) -m arthritis.splits --exp configs/smoke2.yaml
	$(PY) -m arthritis.train --exp configs/smoke2.yaml --smoke --run-name smoke_v2
	$(PY) -m arthritis.evaluate results/smoke_v2 --n-boot 200

ladder:             ## every rung, internal CV
	for e in majority joint_prior pixel_logreg scratch_cnn resnet50 densenet121 convnext_tiny; do \
	  $(PY) -m arthritis.train --exp $$e --scheme cv --run-name $$e\_cv && \
	  $(PY) -m arthritis.evaluate results/$$e\_cv; done

reproduce: manifest splits ladder   ## regenerates results/summary.csv from scratch
