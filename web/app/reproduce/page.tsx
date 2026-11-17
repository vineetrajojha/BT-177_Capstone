import { Terminal, Copy, CheckCircle2, ShieldCheck, Cpu, Database, AlertCircle } from "lucide-react";

export default function ReproducePage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">Step-by-Step Reproduction Guide</h1>
        <p className="text-xs text-slate-400 mt-1">
          Exact terminal commands, fixed seeds, environment pins, and cloud training setup to reproduce 100% of results.
        </p>
      </div>

      {/* Hardware & Environment Specs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-5 rounded-xl bg-dark-800/80 border border-dark-700 space-y-2">
          <div className="flex items-center gap-2 text-brand-400 font-bold text-xs uppercase tracking-wider">
            <Cpu className="w-4 h-4" /> Hardware Target
          </div>
          <div className="text-xl font-bold text-white">NVIDIA T4 GPU</div>
          <p className="text-xs text-slate-400">16 GB VRAM, Turing architecture. Official latency target with batch size 1.</p>
        </div>

        <div className="p-5 rounded-xl bg-dark-800/80 border border-dark-700 space-y-2">
          <div className="flex items-center gap-2 text-brand-400 font-bold text-xs uppercase tracking-wider">
            <Database className="w-4 h-4" /> Fixed Seeds
          </div>
          <div className="text-xl font-bold text-white">Seeds 0, 1, 2</div>
          <p className="text-xs text-slate-400">All evaluation numbers report mean &plusmn; standard deviation over 3 fixed random seeds.</p>
        </div>

        <div className="p-5 rounded-xl bg-dark-800/80 border border-dark-700 space-y-2">
          <div className="flex items-center gap-2 text-brand-400 font-bold text-xs uppercase tracking-wider">
            <ShieldCheck className="w-4 h-4" /> Framework Pin
          </div>
          <div className="text-xl font-bold text-white font-mono text-base">OpenPCDet v0.5.2</div>
          <p className="text-xs text-slate-400">Commit <code className="text-brand-400">233f8498</code> with zero proprietary CUDA kernels.</p>
        </div>
      </div>

      {/* Ordered Terminal Commands */}
      <div className="space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Terminal className="w-5 h-5 text-brand-400" />
          Ordered Reproduction Workflow
        </h2>

        {/* Step 1 */}
        <div className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 space-y-3">
          <div className="font-bold text-white text-sm">Step 1: Clone Repository & Create Environment</div>
          <pre className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 font-mono text-xs text-brand-400 overflow-x-auto">
{`git clone https://github.com/vineetrajojha/BT-177_Capstone.git
cd BT-177_Capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/apply_patches.py
python scripts/check_env.py
pytest tests/ -v`}
          </pre>
        </div>

        {/* Step 2 */}
        <div className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 space-y-3">
          <div className="font-bold text-white text-sm">Step 2: KITTI Dataset Verification & Calibration Check</div>
          <pre className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 font-mono text-xs text-brand-400 overflow-x-auto">
{`# Check calibration across 20 seeded frames (mean fraction >= 0.90 required)
python scripts/calibration_check.py

# Prepare dataset infos
python scripts/prepare_kitti.py --kitti_root data/kitti`}
          </pre>
        </div>

        {/* Step 3 */}
        <div className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 space-y-3">
          <div className="font-bold text-white text-sm">Step 3: PointPillars Reference Training & Aggregation</div>
          <pre className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 font-mono text-xs text-brand-400 overflow-x-auto">
{`python third_party/OpenPCDet/tools/train.py --cfg_file configs/lidar_only_pointpillars.yaml --extra_tag seed_0
python results/aggregate.py`}
          </pre>
        </div>

        {/* Step 4 */}
        <div className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 space-y-3">
          <div className="font-bold text-white text-sm">Step 4: BEVFusionLite Light Model Training & Benchmarking</div>
          <pre className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 font-mono text-xs text-brand-400 overflow-x-auto">
{`# Train Light Model (80 epochs, Adam OneCycle lr 0.003)
python third_party/OpenPCDet/tools/train.py --cfg_file configs/light_model.yaml --extra_tag light_seed_0

# Benchmark inference latency and throughput on T4 GPU (FP32 & FP16)
python tools/benchmark.py --config configs/light_model.yaml --precision fp32 --output_dir results/light_model_fp32
python tools/benchmark.py --config configs/light_model.yaml --precision fp16 --output_dir results/light_model_fp16

# Run full project integrity audit
python scripts/audit.py`}
          </pre>
        </div>
      </div>

      {/* Cloud Execution Instructions (Colab / Kaggle) */}
      <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
        <h2 className="text-base font-bold text-white">Google Colab & Kaggle Turnkey Runners</h2>
        <p className="text-xs text-slate-300 leading-relaxed">
          Because full multi-seed 80-epoch training and uncompressing the full ~80 GB KITTI dataset exceeds local disk limits:
        </p>
        <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside">
          <li><strong>Google Colab:</strong> Open <code className="text-brand-400">tools/colab_train.ipynb</code> in Google Colab, select GPU runtime (T4), and run all cells with Google Drive mounted.</li>
          <li><strong>Kaggle:</strong> Execute <code className="text-brand-400">python tools/kaggle_train.py --config configs/light_model.yaml --run_name light_seed_0</code> with the KITTI dataset attached.</li>
        </ul>
      </div>

      {/* Deviations Summary */}
      <div className="p-6 rounded-xl bg-dark-800/40 border border-dark-700 space-y-2">
        <div className="flex items-center gap-2 text-slate-300 font-bold text-xs uppercase tracking-wider">
          <AlertCircle className="w-4 h-4 text-brand-400" />
          Protocol Deviations Summary (DEVIATIONS.md)
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          1. <strong>Pad-Only Preprocessing:</strong> Images are padded to 384x1248 with scale=1.0 preserving unpadded camera intrinsics P2, rather than using bilinear stretching.<br />
          2. <strong>LiDAR-Depth Frustum Guidance:</strong> View transform utilizes sparse LiDAR depth maps min-pooled to stride 8 with D=48 bins and auxiliary CE loss.<br />
          3. <strong>Cloud GPU Offload:</strong> Multi-seed training is staged on Kaggle / Colab cloud runners due to workstation disk space (~14 GiB free vs ~80 GiB uncompressed KITTI).
        </p>
      </div>
    </div>
  );
}
