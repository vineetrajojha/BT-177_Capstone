import { Terminal, ShieldCheck, Cpu, Database, AlertCircle } from "lucide-react";

export default function ReproducePage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Step-by-Step Reproduction Guide</h1>
        <p className="text-xs text-zinc-400 mt-1">
          Exact terminal commands, fixed seeds, environment pins, and cloud training setup to reproduce 100% of results.
        </p>
      </div>

      {/* Hardware & Environment Specs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1.5">
          <div className="flex items-center gap-2 text-zinc-400 font-semibold text-[11px] uppercase tracking-wider">
            <Cpu className="w-3.5 h-3.5 text-zinc-400" /> Hardware Target
          </div>
          <div className="text-lg font-bold text-white">NVIDIA T4 GPU</div>
          <p className="text-xs text-zinc-400 leading-relaxed">16 GB VRAM, Turing architecture. Official latency target with batch size 1.</p>
        </div>

        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1.5">
          <div className="flex items-center gap-2 text-zinc-400 font-semibold text-[11px] uppercase tracking-wider">
            <Database className="w-3.5 h-3.5 text-zinc-400" /> Fixed Seeds
          </div>
          <div className="text-lg font-bold text-white">Seeds 0, 1, 2</div>
          <p className="text-xs text-zinc-400 leading-relaxed">All evaluation numbers report mean &plusmn; standard deviation over 3 fixed random seeds.</p>
        </div>

        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1.5">
          <div className="flex items-center gap-2 text-zinc-400 font-semibold text-[11px] uppercase tracking-wider">
            <ShieldCheck className="w-3.5 h-3.5 text-zinc-400" /> Framework Pin
          </div>
          <div className="text-lg font-bold text-white font-mono text-sm">OpenPCDet v0.5.2</div>
          <p className="text-xs text-zinc-400 leading-relaxed">Commit <code className="text-zinc-200">233f8498</code> with zero proprietary CUDA kernels.</p>
        </div>
      </div>

      {/* Ordered Terminal Commands */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Terminal className="w-4 h-4 text-zinc-400" />
          Ordered Reproduction Workflow
        </h2>

        {/* Step 1 */}
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
          <div className="font-medium text-white text-xs">Step 1: Clone Repository &amp; Create Environment</div>
          <pre className="p-3 rounded bg-black border border-zinc-800/80 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
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
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
          <div className="font-medium text-white text-xs">Step 2: KITTI Dataset Verification &amp; Calibration Check</div>
          <pre className="p-3 rounded bg-black border border-zinc-800/80 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
{`# Check calibration across 20 seeded frames (mean fraction >= 0.90 required)
python scripts/calibration_check.py

# Prepare dataset infos
python scripts/prepare_kitti.py --kitti_root data/kitti`}
          </pre>
        </div>

        {/* Step 3 */}
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
          <div className="font-medium text-white text-xs">Step 3: PointPillars Reference Training &amp; Aggregation</div>
          <pre className="p-3 rounded bg-black border border-zinc-800/80 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
{`python third_party/OpenPCDet/tools/train.py --cfg_file configs/lidar_only_pointpillars.yaml --extra_tag seed_0
python results/aggregate.py`}
          </pre>
        </div>

        {/* Step 4 */}
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
          <div className="font-medium text-white text-xs">Step 4: BEVFusionLite Light Model Training &amp; Benchmarking</div>
          <pre className="p-3 rounded bg-black border border-zinc-800/80 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
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
      <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
        <h2 className="text-xs font-semibold text-white uppercase tracking-wider">Google Colab &amp; Kaggle Turnkey Runners</h2>
        <p className="text-xs text-zinc-400 leading-relaxed">
          Because full multi-seed 80-epoch training and uncompressing the full ~80 GB KITTI dataset exceeds local disk limits:
        </p>
        <ul className="text-xs text-zinc-400 space-y-1.5 list-disc list-inside">
          <li><strong>Google Colab:</strong> Open <code className="text-zinc-200">tools/colab_train.ipynb</code> in Google Colab, select GPU runtime (T4), and run all cells with Google Drive mounted.</li>
          <li><strong>Kaggle:</strong> Execute <code className="text-zinc-200">python tools/kaggle_train.py --config configs/light_model.yaml --run_name light_seed_0</code> with the KITTI dataset attached.</li>
        </ul>
      </div>

      {/* Deviations Summary */}
      <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
        <div className="flex items-center gap-2 text-zinc-300 font-medium text-xs uppercase tracking-wider">
          <AlertCircle className="w-3.5 h-3.5 text-zinc-400" />
          <span>Protocol Deviations Summary (DEVIATIONS.md)</span>
        </div>
        <p className="text-xs text-zinc-400 leading-relaxed">
          1. <strong>Pad-Only Preprocessing:</strong> Images are padded to 384x1248 with scale=1.0 preserving unpadded camera intrinsics P2, rather than using bilinear stretching.<br />
          2. <strong>LiDAR-Depth Frustum Guidance:</strong> View transform utilizes sparse LiDAR depth maps min-pooled to stride 8 with D=48 bins and auxiliary CE loss.<br />
          3. <strong>Cloud GPU Offload:</strong> Multi-seed training is staged on Kaggle / Colab cloud runners due to workstation disk space (~14 GiB free vs ~80 GiB uncompressed KITTI).
        </p>
      </div>
    </div>
  );
}
