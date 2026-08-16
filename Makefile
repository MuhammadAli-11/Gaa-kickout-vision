# gaa-kickout-vision — automating the McColgan et al. (2026) kickout coding scheme
VIDEO_ID ?= lgf25_r3
PY       := PYTHONPATH=src python

.PHONY: help setup test demo prepare detect track shots fit-pitch features teams \
        code eval-coding tier1 events eval-events report overlay check clean-outputs

help:
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:        ## install python deps
	pip install -r requirements.txt

test:         ## known-answer statistics tests
	$(PY) -m pytest tests/ -q

demo:         ## evaluation half on synthetic data — no footage needed
	$(PY) tools/make_synthetic.py
	@RID=$$($(PY) -c "import sys;sys.path.insert(0,'src');from lib.config import load_config,run_id;print(run_id(load_config()))"); \
	mkdir -p outputs/$$RID && cp outputs/synthetic/*.parquet outputs/synthetic/*.json outputs/synthetic/*.csv outputs/$$RID/ 2>/dev/null || true
	$(PY) src/s08_code_kickouts.py    --video-id synthetic_demo --from-gt
	$(PY) src/s11_evaluate_coding.py  --video-id synthetic_demo

plan:         ## plan working windows from your kickout timestamp list
	$(PY) tools/plan_segments.py --kickouts data/gt/kickouts_source.txt \
		--video-id $(VIDEO_ID) --window-minutes 12 --max-windows 2

prepare:      ## s01 standardise footage
	$(PY) src/s01_prepare_footage.py --video-id $(VIDEO_ID)
detect:       ## s02 YOLO detection
	$(PY) src/s02_detect.py --video-id $(VIDEO_ID)
track:        ## s03 ByteTrack
	$(PY) src/s03_track.py --video-id $(VIDEO_ID)
features-img: ## s05 pass 1 — image space only (needed before registration)
	$(PY) src/s05_features.py --video-id $(VIDEO_ID) --image-only
shots:        ## s04 list camera shots to register
	$(PY) src/s04_register_pitch.py --video-id $(VIDEO_ID) --list-shots
fit-pitch:    ## s04 fit + validate homographies (after clicking landmarks)
	$(PY) src/s04_register_pitch.py --video-id $(VIDEO_ID) --fit
features:     ## s05 pass 2 — pitch space in metres
	$(PY) src/s05_features.py --video-id $(VIDEO_ID)
teams:        ## s06 assign players to teams
	$(PY) src/s06_assign_teams.py --video-id $(VIDEO_ID)
code:         ## s08 code kickouts at MANUAL timestamps (Tier 1)
	$(PY) src/s08_code_kickouts.py --video-id $(VIDEO_ID) --from-gt
eval-coding:  ## s11 per-variable agreement + ceiling + coverage
	$(PY) src/s11_evaluate_coding.py --video-id $(VIDEO_ID) --source gt_timestamps

tier1: prepare detect track features-img shots  ## Tier 1 up to the manual landmark step
	@echo ""
	@echo "  ---> Now click pitch landmarks on the shots containing kickouts:"
	@echo "       python src/s04_register_pitch.py --video-id $(VIDEO_ID) --annotate --shot N"
	@echo "  ---> Then:  make tier1-rest VIDEO_ID=$(VIDEO_ID)"

tier1-rest: fit-pitch features teams code eval-coding  ## Tier 1 after landmarks
	$(PY) src/s09_evaluate_detection.py --video-id $(VIDEO_ID) || true
	$(PY) src/s12_failure_audit.py --video-id $(VIDEO_ID) || true
	$(PY) src/s14_report.py --video-id $(VIDEO_ID)

events:       ## s07 automatic temporal kickout detection (Tier 2)
	$(PY) src/s07_detect_kickouts.py --video-id $(VIDEO_ID)
eval-events:  ## s10 temporal localisation metrics (Tier 2)
	$(PY) src/s10_evaluate_events.py --video-id $(VIDEO_ID)

overlay:      ## s13 demo video
	$(PY) src/s13_render_overlay.py --video-id $(VIDEO_ID)
report:       ## s14 figures and tables
	$(PY) src/s14_report.py --video-id $(VIDEO_ID)

check:        ## refuse to push footage: run before every commit
	@git status --porcelain -uall | grep -E '\.(mp4|mkv|jpg|png|parquet|pt)$$' && \
		{ echo "REFUSING: media or weights staged"; exit 1; } || echo "clean: no media staged"

clean-outputs:
	rm -rf outputs/run_* outputs/synthetic
