import Link from "next/link";
import { ArrowRight, Play, Github, CheckCircle2, ShieldCheck, Zap, Layers, Sparkles, Cpu } from "lucide-react";
import summaryData from "../public/data/summary.json";
import targetsData from "../public/data/targets_check.json";

export default function HomePage() {
  const headlines = summaryData.headline_metrics;

  return (
    <div className="space-y-20 pb-16">
      {/* Hero Section */}
      <section className="relative pt-20 pb-16 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 inline-block" />
            <span>Open-Source Research Plugin for OpenPCDet</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-black text-white tracking-tight leading-tight">
            Lightweight Camera-LiDAR <br />
            <span className="text-zinc-200">
              BEV Fusion 3D Object Detection
            </span>
          </h1>

          <p className="max-w-3xl mx-auto text-base sm:text-lg text-zinc-400 leading-relaxed font-normal">
            BEVFusionLite pairs a symmetric 0.32m pillarized LiDAR encoder with a compact ResNet-18 camera stream and a LiDAR-depth-guided frustum view transformer, sustaining <strong className="text-white font-medium">31.2 FPS on an NVIDIA T4 GPU</strong> with zero proprietary CUDA kernels.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3 pt-3">
            <Link
              href="/demo"
              className="flex items-center gap-2 px-5 py-2.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs shadow-sm transition-colors"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Explore Interactive 3D Demo</span>
            </Link>

            <Link
              href="/results"
              className="flex items-center gap-2 px-5 py-2.5 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 font-medium text-xs transition-colors"
            >
              <span>View Benchmark Results</span>
              <ArrowRight className="w-3.5 h-3.5 text-zinc-400" />
            </Link>

            <a
              href="https://github.com/vineetrajojha/BT-177_Capstone"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 px-5 py-2.5 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white font-medium text-xs transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              <span>GitHub</span>
            </a>
          </div>

          <div className="text-xs text-zinc-500 pt-2 flex items-center justify-center gap-2">
            <span>Source verified from</span>
            <code className="text-zinc-300 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-[11px] font-mono">
              {summaryData.source_file}
            </code>
          </div>
        </div>
      </section>

      {/* Headline Metric Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {headlines.map((item, idx) => (
            <div
              key={idx}
              className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 hover:border-zinc-700 transition-colors space-y-2 relative group"
            >
              <div className="text-[11px] uppercase font-semibold text-zinc-400 tracking-wider">
                {item.label}
              </div>
              <div className="text-3xl font-black text-white group-hover:text-blue-400 transition-colors">
                {item.value}
              </div>
              <div className="text-xs text-zinc-400">
                {item.detail}
              </div>
              <div className="pt-2 border-t border-zinc-900 text-[10px] text-zinc-500 font-mono">
                source: {item.source_file}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Interactive Architecture Flow Diagram */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="text-center space-y-1">
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Pure-PyTorch Dual-Stream BEV Pipeline
          </h2>
          <p className="text-xs text-zinc-400 max-w-2xl mx-auto">
            From raw multimodal sensor inputs to 3D bounding box predictions without non-standard CUDA extensions.
          </p>
        </div>

        <div className="p-6 rounded-xl bg-zinc-950 border border-zinc-800 grid grid-cols-1 lg:grid-cols-4 gap-4 relative">
          {/* Step 1: Encoders */}
          <div className="space-y-3 p-4 rounded-lg bg-zinc-900/60 border border-zinc-800">
            <div className="flex items-center gap-2 text-white font-semibold text-xs">
              <Zap className="w-3.5 h-3.5 text-blue-400" />
              <span>1. Modality Encoders</span>
            </div>
            <div className="space-y-2 text-xs text-zinc-300">
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">LiDAR Branch:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">32-dim PillarVFE on 0.32m grid (216x248)</p>
              </div>
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">Camera Branch:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">ResNet-18 + FPN to stride 8 (64 ch)</p>
              </div>
            </div>
          </div>

          {/* Step 2: Depth-Guided Lifting */}
          <div className="space-y-3 p-4 rounded-lg bg-zinc-900/60 border border-zinc-800">
            <div className="flex items-center gap-2 text-white font-semibold text-xs">
              <Layers className="w-3.5 h-3.5 text-blue-400" />
              <span>2. Depth-Guided Lifting</span>
            </div>
            <div className="space-y-2 text-xs text-zinc-300">
              <p className="leading-relaxed text-[11px] text-zinc-400">
                Projects sparse LiDAR onto image plane; min-pools to stride 8.
              </p>
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">D=48 Depth Bins:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">[1.0m to 69.0m] with 0.1 aux CE loss</p>
              </div>
              <p className="text-[11px] text-blue-400 font-medium">
                Eliminates ray feature smearing
              </p>
            </div>
          </div>

          {/* Step 3: BEV Fusion */}
          <div className="space-y-3 p-4 rounded-lg bg-zinc-900/60 border border-zinc-800">
            <div className="flex items-center gap-2 text-white font-semibold text-xs">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              <span>3. ConvFuser Module</span>
            </div>
            <div className="space-y-2 text-xs text-zinc-300">
              <p className="leading-relaxed text-[11px] text-zinc-400">
                Symmetric channel concatenation in shared Bird&apos;s-Eye-View.
              </p>
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">2x Conv-BN-ReLU:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">(64 cam + 32 LiDAR = 96) -&gt; 64 ch</p>
              </div>
              <p className="text-[11px] text-zinc-400">
                Pure PyTorch scatter-add
              </p>
            </div>
          </div>

          {/* Step 4: Head & Prediction */}
          <div className="space-y-3 p-4 rounded-lg bg-zinc-900/60 border border-zinc-800">
            <div className="flex items-center gap-2 text-white font-semibold text-xs">
              <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
              <span>4. Head & Output</span>
            </div>
            <div className="space-y-2 text-xs text-zinc-300">
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">BaseBEVBackbone:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">2-stage downsample & upsample</p>
              </div>
              <div className="p-2.5 rounded bg-zinc-950 border border-zinc-800">
                <span className="font-semibold text-white">AnchorHeadSingle:</span>
                <p className="text-zinc-400 text-[11px] mt-0.5">Car, Pedestrian, Cyclist 3D boxes</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Target Criteria Status Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Verified Acceptance Targets</h2>
            <p className="text-xs text-zinc-400">Strictly audited against predefined criteria without modifying thresholds.</p>
          </div>
          <Link href="/results" className="text-xs text-blue-400 hover:text-blue-300 font-medium flex items-center gap-1">
            <span>Full Results</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {targetsData.targets.map((t, idx) => (
            <div key={idx} className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 flex flex-col justify-between">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">{t.name}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    {t.status}
                  </span>
                </div>
                <p className="text-[11px] text-zinc-400 leading-relaxed">{t.threshold}</p>
              </div>
              <div className="mt-3 pt-2.5 border-t border-zinc-800/80 flex items-center justify-between text-xs">
                <span className="text-zinc-500">Measured:</span>
                <span className="font-semibold text-zinc-200">{t.measured}</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
