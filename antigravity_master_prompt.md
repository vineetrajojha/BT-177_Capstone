# MASTER PROMPT FOR ANTIGRAVITY — BEVFusionLite: finish the research + ship a hosted website

Paste everything below the line into Antigravity (Agent Manager, Planning mode). If it is too long for one message, save this file as `docs/MASTER_PROMPT.md` in the repo and tell the agent: "Read docs/MASTER_PROMPT.md fully and follow it."

---------------------------------------------------------------------

## 0. ROLE AND GOAL

You are the sole engineer finishing **BEVFusionLite** (repo: https://github.com/vineetrajojha/BT-177_Capstone), a lightweight camera-LiDAR BEV fusion 3D object detector built as an OpenPCDet plugin, evaluated on KITTI (primary) and nuScenes-mini (secondary).

Two deliverables, in this order of priority:
1. **A working, honest research pipeline**: trained heavy and light models, benchmark numbers, ablations, reports (the 12-prompt plan in section 5).
2. **A deployed public website** that presents the project end to end: live demo, results dashboard, ablations, nuScenes-mini, architecture, report downloads, reproduction guide (section 6 and 7).

Use existing pretrained models and open-source repos wherever they reduce risk, but label every borrowed piece clearly (section 3).

## 1. PERMANENT RULES (never violate)

1. Work only in the repo root. Keep layout: `configs/ bevfusionlite/ scripts/ tests/ tools/ results/ reports/ demo/ docs/ third_party/ patches/ data/ (gitignored)`. Add `web/` (frontend) and `api/` (backend) for the website.
2. Reproducibility: fixed seeds, config saved next to each checkpoint, git commit hash logged, `pip freeze` in `results/env.txt`.
3. **Never fabricate** numbers, citations, or results. Every number shown anywhere (report, website, README) must come from a file under `results/` produced by a script. Every citation must be fetched and verified (title, authors, year, arXiv/DOI). If you cannot verify, write UNVERIFIED and stop to ask me.
4. Every task ends with: (a) files created or changed, (b) exact reproduce commands, (c) acceptance checklist with PASS/FAIL and evidence.
5. pytest for every new module: shape tests, gradient-flow test, determinism test where relevant.
6. Do not silently change a target, split, metric, or hyperparameter. If a spec looks wrong, stop and ask.
7. Long training runs go in the background, log to `results/<run>/train.log`, checkpoint every epoch, support `--resume`.
8. If compute is limited, reduce epochs/batch size and record it in `results/<run>/DEVIATIONS.md`. Never reduce the evaluation protocol.
9. Evaluation: KITTI AP R40 (3D and BEV), Car/Pedestrian/Cyclist, easy/moderate/hard, val split. Mean and std over 3 seeds where seeds are required.
10. Pretrained weights, third-party code and datasets: record name, URL, commit/version, license, and exactly how each was used in `docs/THIRD_PARTY.md`. Never present a pretrained model's result as our own.
11. If a decision is genuinely mine to make (budget, accounts, downloads needing login, anything irreversible), stop and ask. Otherwise decide, state the decision at the top of your work, and continue.

## 2. FIRST ACTION: AUDIT THE REPO (do this before writing code)

The repo already contains: `bevfusionlite/`, `data/kitti/`, `docs/`, `patches/`, `reports/`, `results/calibration/`, `scripts/`, `tests/`, `HANDOVER.md`, `Makefile`, `README.md`. The README states the architecture: PillarVFE -> PointPillarScatter -> BaseBEVBackbone (LiDAR), ResNet-18 -> FPN -> LSS view transform (camera), channel concat -> 2-layer conv fusion, AnchorHeadSingle (Car, Pedestrian, Cyclist).

Do this:
1. Read `HANDOVER.md`, `README.md`, `Makefile`, `docs/`, `reports/`, and every file under `scripts/`, `tests/`, `bevfusionlite/`.
2. Run `pytest` and `python scripts/check_env.py` (if present). Report what passes and fails.
3. Write `docs/STATUS_AUDIT.md`: for each of the plan prompts P1 to P12 (section 5) say DONE / PARTIAL / NOT STARTED with evidence (file paths, test output). Flag anything that deviates from the spec (for example the README says "LSS view transform" while the spec requires a **LiDAR-depth-guided** transform with sparse depth map, 48 bins, aux depth loss).
4. Check whether KITTI data exists in `data/kitti/` and whether file counts match (7481 train+val, 7518 test; split 3712/3769). If not, tell me exactly what to download and stop for that part only.
5. Show me the audit and the plan for the remaining work, then continue with the first unfinished prompt. Do not redo finished work that passes its acceptance checks.

## 3. REUSE MAP (use these instead of writing from scratch; verify each URL, version and license before use)

| Need | Reuse | How to use it |
|---|---|---|
| Detection framework | OpenPCDet (open-mmlab) | Already the base. Pin commit in `docs/PINNED_VERSIONS.md`; fixes only via `patches/`. |
| LiDAR-only reference | OpenPCDet model zoo: official KITTI PointPillars checkpoint and its README numbers | Use as the **published reference** and as a sanity check for the pipeline. Still train our own seeds 0,1,2 per the plan, because the study needs same-code, same-protocol runs. Do not report the zoo checkpoint as our result. |
| Image backbones | torchvision ImageNet weights: ResNet-18, ResNet-50, MobileNetV3-Large | Image branch and ablation rows. |
| View transform reference | Lift-Splat-Shoot, BEVDepth, MIT BEVFusion (`mit-han-lab/bevfusion`), BEVFusion-PKU (`ADLab-AutoDrive/BEVFusion`), mmdet3d BEVFusion | Read for geometry and depth-guided lifting ideas. Our `bev_transform.py` stays pure PyTorch with no custom CUDA. Do not copy code without recording license and attribution. |
| Faster training (allowed, must be disclosed) | Initialise the LiDAR branch from the official OpenPCDet PointPillars weights | Only if compute forces it. This changes the protocol, so it needs my approval, goes in `DEVIATIONS.md`, and must be applied equally to heavy and light and to all ablation rows. |
| nuScenes | `nuscenes-devkit`, OpenPCDet nuScenes dataset code, CenterHead | Follow plan P9. ImageNet weights only for the image backbone. |
| Demo 3D viewer | Streamlit + Rerun SDK (per plan P11), or Plotly/three.js | Used for the Streamlit demo. The public website uses its own viewer (section 6). |
| Efficiency | fvcore, torch profiler | `tools/benchmark.py` per plan P5. |

Pretrained detectors from other repos (for example MIT BEVFusion nuScenes weights) may be shown on the website **only** in a clearly separated "Reference models (not ours)" section, with source and license, and never mixed into our tables.

## 4. COMPUTE STRATEGY

Official latency/memory numbers must be measured on an NVIDIA **T4** (otherwise `official: false`). Do this:
- Ask me what GPU I have access to (local GPU, Kaggle, Colab, cloud credits). Until I answer, assume a free T4 session with time limits.
- Make every training script resumable and robust to session death (`--resume`, per-epoch checkpoints, checkpoints synced to a persistent location such as Google Drive or a Hugging Face Hub private repo).
- Give me a table of the estimated GPU hours per prompt, mark them as estimates, and propose which prompts to shrink (fewer epochs, seed 0 only for variants) with the DEVIATIONS.md entries pre-written. I decide.
- Provide a ready-to-run notebook (`tools/colab_train.ipynb`) and a Kaggle script that clone the repo, install env, mount data, and launch any run by name.

## 5. RESEARCH PIPELINE (the original 12-prompt plan; keep all acceptance gates)

Execute only the prompts not already DONE per your audit. Each prompt has hard gates; do not proceed if a gate fails.

- **P1 Environment**: conda env `bevlite` (Python 3.10, PyTorch 2.1+ matching CUDA), OpenPCDet cloned to `third_party/OpenPCDet` and pinned, compile fixes only via `patches/`, registry in `bevfusionlite/__init__.py`, `scripts/check_env.py`, Makefile targets env/test/lint. Gate: check_env runs on GPU, pytest passes.
- **P2 KITTI pipeline + calibration**: `scripts/prepare_kitti.py`, `bevfusionlite/datasets/kitti_fusion.py` (points cropped to camera FOV and range x[0,69.12] y[-39.68,39.68] z[-3,1]; image 384x1248 ImageNet-normalised; P2/R0_rect/Tr_velo_to_cam; 4x4 lidar-to-image; GT in LiDAR frame), consistent augmentation (BEV flip disabled while the image branch is on, documented), `scripts/verify_calibration.py` on 20 seeded frames, `tests/test_calibration.py` (3 px tolerance). Gates: train 3712 / val 3769; calibration check >= 0.90; 20 overlays saved and visually checked; collate and val epoch run.
- **P3 PointPillars reference**: `configs/lidar_only_pointpillars.yaml` (voxel 0.16x0.16x4, 32 pts/pillar, grid 432x496, 80 epochs, Adam one-cycle lr 0.003, bs 4, AMP), seeds 0,1,2, `results/aggregate.py` -> `results/pointpillars_kitti_val.csv` with published reference side by side. Gate: seed 0 Car 3D AP R40 moderate >= 74 (published about 77). Do not continue with a broken reference.
- **P4 Literature review + novelty**: 25 to 40 **verified** papers (BEV fusion, lifting, LiDAR encoders, lightweight/efficient detection, datasets/metrics), `reports/literature_review.md`, `reports/novelty_statement.md` (max 250 words, name >= 5 closest works, claim only what ablations support; do not claim LiDAR-depth-guided lifting is new unless the review shows no prior work), `reports/references.bib`. Run in parallel with training. I will spot-check 5 citations.
- **P5 `tools/benchmark.py`**: params (total and per module), GFLOPs (fvcore, batch 1, real val sample, list exclusions), latency (50 warm-up, 300 timed, synchronized; model-only and end-to-end), peak memory, checkpoint MB, JSON to `results/<run>/efficiency.json` with GPU name, torch version, precision, batch size, `official` flag.
- **P6 Model code + heavy baseline**: image branch (ResNet-50 heavy / ResNet-18 light, FPN to stride 8, C_img 128/64), LiDAR branch (PillarVFE + scatter; heavy 0.16 m 64 ch, light 0.32 m 32 ch), **LiDAR-depth-guided image-to-BEV transform** (sparse depth map min-pooled to feature stride, depth head with depth+mask channels, D=48 bins over 1 to 69 m, softmax, frustum outer product, geometry to LiDAR frame, scatter-add into BEV, N cameras supported, aux depth CE loss weight 0.1), fusion (concat -> conv-BN-ReLU stack; modes camera_only / lidar_only), BEV backbone + AnchorHeadSingle, `BEVFusionLiteDetector` registered in OpenPCDet, all configs. Tests per stage, synthetic-point BEV-cell test, gradient-to-backbone test. Gates: all tests pass; heavy overfits a 16-frame subset in 300 iterations; heavy Car 3D moderate not more than 2 points below PointPillars.
- **P7 Light model**: `docs/ARCHITECTURE_LIGHT.md`, light seeds 0,1,2, camera_only and lidar_only variants, fp32 and fp16 benchmarks, `results/main_comparison.csv`, `results/targets_check.md` (MET / NOT MET, never edit the targets): Car 3D within 3 AP of heavy; params and FLOPs at least 50% lower; peak memory at least 40% lower; FPS at least 2x heavy and at least 20 FPS on T4; fusion beats LiDAR-only especially on Pedestrian and Cyclist.
- **P8 Ablations (3 seeds each)**: image backbone (R18 ref, MobileNetV3-L, R50), voxel (0.16, 0.32 ref, 0.40), fusion (concat+2 convs ref, concat+1x1, channel-attention), depth guidance (ON ref, OFF), depth bins (24, 48 ref, 64), precision fp32 vs fp16. Outputs: `results/ablation_table.csv`, md/LaTeX table, AP-vs-latency figure with error bars, written analysis citing only table values and flagging differences within one std.
- **P9 nuScenes-mini**: devkit v1.0-mini (stop and ask me if the download needs my login), `bevfusionlite/datasets/nuscenes_fusion.py` (6 cameras, 10 sweeps, 5 features, per-camera matrices), pillar 0.2/0.4 m configs, CenterHead (or anchor head, state which), train heavy and light x 3 seeds from ImageNet weights only, official mAP/NDS, `results/nuscenes_mini.csv`, `reports/nuscenes_notes.md` with the small-data limitation statement.
- **P10 KITTI test-server package**: median-seed model (rule stated in report), 7518 test predictions in KITTI label format with LiDAR->camera conversion, `scripts/check_submission.py`, val dry run matching OpenPCDet eval within 0.1 AP, `results/kitti_test_submission.zip`, `reports/submission_instructions.md`. I upload it myself and paste the returned numbers.
- **P11 Streamlit/Rerun demo**: kept as the internal/offline demo and as a fallback (`demo/`, `scripts/precompute_demo.py` caching predictions for 50 fixed val frames, AppTest tests).
- **P12 Final report + audit**: `reports/final_report.md` (+PDF if pandoc is available), `scripts/make_report_tables.py` (no hand-typed numbers), `docs/REPRODUCE.md`, `docs/slides_outline.md` (12 slides), `scripts/audit.py`, README with a 10-line quick start.

Targets may not all be met. Report honest partial results. Never move a target.

## 6. THE WEBSITE (the main new deliverable)

### 6.1 Architecture (decided; do not ask unless blocked)

The website must work **without a GPU at serve time** and be free or very cheap to host.

- **Frontend**: Next.js (App Router) + TypeScript + Tailwind, in `web/`. Deploy on **Vercel**. Dark, clean, research-grade look. Mobile responsive. Lighthouse performance >= 90 and accessibility >= 90 on the home page.
- **Backend / data**: static JSON + images generated by scripts from `results/` and the cached predictions, committed to `web/public/data/` (or served from a small FastAPI service in `api/`). No number may be typed by hand into the frontend; it must read generated JSON.
- **Live inference (optional, tier 2)**: a Hugging Face Space (Docker, FastAPI or Gradio) running the light model on CPU for a **single uploaded KITTI-format sample** (image + velodyne .bin + calib .txt), with a hard file-size limit, rate limit, and timeouts. The frontend calls it if available and falls back to the cached demo frames if the Space is asleep or fails. Put model weights on the Hugging Face Hub (model repo, with a model card stating the dataset, metrics from `results/`, limitations, and license).
- **Cached demo (tier 1, always works)**: `scripts/precompute_demo.py` runs heavy and light on 50 fixed val frames and writes per-frame JSON (boxes, classes, scores, GT boxes) plus the camera image with projected boxes, BEV render and point cloud (downsampled).

### 6.2 Pages and features (all required)

1. **Home**: one-sentence pitch, 3 headline numbers read from `results/main_comparison.csv` (params reduction, FPS gain, Car AP gap), animated architecture pipeline, links to GitHub, report PDF, demo.
2. **Live Demo**: frame selector (slider + id input), model switch (light / heavy / side by side), score threshold slider, GT toggle, class filter. Panels: camera image with projected 3D boxes, interactive 3D point cloud with predicted/GT boxes (three.js or Plotly 3D), BEV view with boxes colour-coded by class, metrics card (latency, FPS, peak memory, params read from `results/<run>/efficiency.json`, labelled "measured on <GPU name from the JSON>, batch 1, <precision>"). "Upload your own sample" box if tier 2 is live.
3. **Results dashboard**: main comparison table (sortable), per-class AP bars with error bars (mean +/- std over seeds), AP-vs-latency scatter, params-vs-AP, targets check shown as MET / NOT MET cards with measured values, fusion-vs-LiDAR-only comparison, fp32 vs fp16.
4. **Ablations**: interactive table + AP-vs-latency figure from `results/ablation_table.csv`, the written analysis, and visible "within one std" flags.
5. **nuScenes-mini**: table, 6-camera sample gallery with projected boxes, the limitation statement verbatim.
6. **Architecture**: interactive diagram of the light model (hover for channels/strides/params/FLOPs per module from `docs/ARCHITECTURE_LIGHT.md` data), explanation of the LiDAR-depth-guided view transform with a small visual (depth bins -> frustum -> BEV).
7. **Literature and novelty**: the verified table from `reports/literature_review.md`, novelty statement, bibliography with links.
8. **Reproduce**: ordered commands from `docs/REPRODUCE.md`, environment, seeds, hardware, DEVIATIONS summary.
9. **Reference models (not ours)**: clearly separated section listing pretrained baselines used (PointPillars zoo checkpoint, ImageNet backbones), with source, license, and the exact role.
10. **About / Limitations / Credits**: honest limitations (small dataset, front camera only on KITTI, calibration sensitivity, mini-only nuScenes, unmet targets), authors, supervisor/institution, **KITTI and nuScenes attribution and non-commercial licence notice** (do not publish raw dataset files beyond the small demo subset those licences allow; check the licence text and say what you included).
11. **Downloads**: final report PDF, slides outline, results CSV/JSON bundle, model weights link (HF Hub), submission instructions.
12. SEO + sharing: titles, meta descriptions, OpenGraph image, sitemap, robots.txt, favicon.

### 6.3 Quality gates for the website

- `npm run build` passes with zero type errors; ESLint clean.
- Playwright smoke tests: every page loads, demo slider changes the frame, model switch works, tables render, no console errors.
- A script `scripts/export_web_data.py` regenerates all `web/public/data/*.json` from `results/`; a test fails if any displayed number lacks a source file.
- Every metric on the site shows its source file path in a tooltip or footnote ("source: results/main_comparison.csv").
- Cold load of the demo page under 3 s on a normal connection with cached frames (lazy-load images and point clouds; downsample clouds).
- Rate limiting, input validation and size limits on any upload endpoint; no secrets in the repo; `.env.example` provided.

## 7. DEPLOYMENT (deliver a live URL)

1. Push the work to GitHub on a branch `website`, with clear commits per feature (regular commits count for my project evaluation).
2. Deploy `web/` to Vercel (use the Vercel CLI or tell me the exact dashboard clicks; stop and ask me only for login/authorization).
3. If tier 2 is built, deploy `api/` as a Hugging Face Space or on Render (free tier), set CORS to the Vercel domain, and add a health endpoint and a cold-start message in the UI.
4. Provide a custom-domain guide (my domain is vineetraj.tech; propose a subdomain such as `bevfusionlite.vineetraj.tech`) with the exact DNS records.
5. Add GitHub Actions: lint + tests on push; optional deploy hook.
6. Final deliverable message: live URL, GitHub URL, HF model/Space URLs, and a checklist PASS/FAIL for every gate in sections 5 and 6, with evidence. List every NOT MET target honestly.

## 8. WORKING STYLE

- Start with the audit (section 2) and show a short plan. Then work in this order, running training in the background while you build other things:
  1. Audit -> fix gaps in P1/P2 (gate: calibration).
  2. Start P3 PointPillars training (background). Meanwhile do P4 literature and P5 benchmark tool.
  3. P6 model code and tests -> heavy training; meanwhile scaffold `web/` with mock data **clearly marked as MOCK** (remove all mock data before the final deployment; add a test that fails if any MOCK flag remains in production data).
  4. P7, P8, P9, P10 as compute allows.
  5. `export_web_data.py`, cached demo frames, finish the website, deploy.
  6. P12 final report and audit.
- Keep a `docs/PROGRESS.md` with a dated log, the current gate status, and what is blocked on me.
- When you need me, ask one specific question with a recommended default.
- Do not invent results to make the website look complete. If a result does not exist yet, the page shows "Not yet measured" with the planned run name.

## 9. THINGS ONLY I CAN DO (ask me at the right moment)

- Download KITTI (needs an account) and nuScenes-mini (needs an account) if the agent cannot.
- Provide or approve GPU access and spending.
- Spot-check 5 citations from the literature review.
- Upload the KITTI test-server zip and paste back the returned results.
- Log in to Vercel / Hugging Face / DNS provider when asked.
