PY ?= python
export PYTHONPATH := src
EXP ?= densenet121
SCHEME ?= cv
DATA ?= data/RAM-H1200-v1
RUN ?=

.PHONY: inspect manifest splits train evaluate smoke test ladder reproduce

inspect:            ## print the dataset layout and label counts (safe before the plan is frozen)
	$(PY) scripts/inspect_dataset.py $(DATA) --out results/data_inspection.txt

manifest:           ## one row per joint ROI, joined to patient/center metadata
	$(PY) -m arthritis.manifest

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
	$(PY) -m arthritis.train --exp configs/smoke.yaml --smoke --run-name smoke_v1cnn
	$(PY) -m arthritis.evaluate results/smoke_v1cnn --n-boot 200

ladder:             ## every rung, internal CV
	for e in majority pixel_logreg v1_cnn resnet50 densenet121 convnext_tiny; do \
	  $(PY) -m arthritis.train --exp $$e --scheme cv --run-name $$e\_cv && \
	  $(PY) -m arthritis.evaluate results/$$e\_cv; done

reproduce: manifest splits ladder   ## regenerates results/summary.csv from scratch
