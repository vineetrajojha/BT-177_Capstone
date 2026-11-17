import Link from "next/link";
import { ArrowRight, Play, Github, FileText, CheckCircle2, ShieldCheck, Zap, Layers, Sparkles, Cpu } from "lucide-react";
import summaryData from "../public/data/summary.json";
import targetsData from "../public/data/targets_check.json";

export default function HomePage() {
  const headlines = summaryData.headline_metrics;

  return (
    <div className="space-y-20 pb-16">
      {/* Hero Section */}
      <section className="relative pt-20 pb-16 overflow-hidden">
        {/* Glow backdrop */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-brand-500/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-dark-800 border border-brand-500/30 text-brand-400 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Open-Source Research Plugin for OpenPCDet</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-black text-white tracking-tight leading-tight">
            Lightweight Camera-LiDAR <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400">
              BEV Fusion 3D Object Detection
            </span>
          </h1>

          <p className="max-w-3xl mx-auto text-lg text-slate-300 leading-relaxed font-normal">
            BEVFusionLite pairs a symmetric 0.32m pillarized LiDAR encoder with a compact ResNet-18 camera stream and a LiDAR-depth-guided frustum view transformer, sustaining <strong className="text-white font-semibold">31.2 FPS on an NVIDIA T4 GPU</strong> with zero proprietary CUDA kernels.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <Link
              href="/demo"
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-brand-500 hover:bg-brand-600 text-dark-900 font-bold text-sm shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Explore Interactive 3D Demo</span>
            </Link>

            <Link
              href="/results"
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-dark-800 hover:bg-dark-700 border border-dark-600 text-white font-medium text-sm transition-all"
            >
              <span>View Benchmark Results</span>
              <ArrowRight className="w-4 h-4 text-slate-400" />
            </Link>

            <a
              href="https://github.com/vineetrajojha/BT-177_Capstone"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-dark-800/80 hover:bg-dark-700 border border-dark-600 text-slate-300 font-medium text-sm transition-all"
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
          </div>

          <div className="text-xs text-slate-500 pt-2 flex items-center justify-center gap-2">
            <span>Source verified from</span>
            <code className="text-brand-400 bg-dark-800 px-2 py-0.5 rounded border border-dark-700">
              {summaryData.source_file}
            </code>
          </div>
        </div>
      </section>

      {/* Headline Metric Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {headlines.map((item, idx) => (
            <div
              key={idx}
              className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 hover:border-brand-500/40 transition-colors space-y-2 relative group"
            >
              <div className="text-xs uppercase font-bold text-slate-400 tracking-wider">
                {item.label}
              </div>
              <div className="text-3xl sm:text-4xl font-black text-white group-hover:text-brand-400 transition-colors">
                {item.value}
              </div>
              <div className="text-xs text-slate-400">
                {item.detail}
              </div>
              <div className="pt-2 text-[10px] text-slate-500 font-mono">
                source: {item.source_file}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Interactive Architecture Flow Diagram */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            Pure-PyTorch Dual-Stream BEV Pipeline
          </h2>
          <p className="text-sm text-slate-400 max-w-2xl mx-auto">
            From raw multimodal sensor inputs to 3D bounding box predictions without non-standard CUDA extensions.
          </p>
        </div>

        <div className="p-8 rounded-2xl bg-dark-800/40 border border-dark-700 grid grid-cols-1 lg:grid-cols-4 gap-6 relative">
          {/* Step 1: Encoders */}
          <div className="space-y-4 p-5 rounded-xl bg-dark-900/80 border border-dark-700/80">
            <div className="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <Zap className="w-4 h-4" />
              <span>1. Modality Encoders</span>
            </div>
            <div className="space-y-3 text-xs text-slate-300">
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">LiDAR Branch:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">32-dim PillarVFE on 0.32m grid (216x248)</p>
              </div>
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">Camera Branch:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">ResNet-18 + FPN to stride 8 (64 ch)</p>
              </div>
            </div>
          </div>

          {/* Step 2: Depth-Guided Lifting */}
          <div className="space-y-4 p-5 rounded-xl bg-dark-900/80 border border-brand-500/30">
            <div className="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <Layers className="w-4 h-4" />
              <span>2. Depth-Guided Lifting</span>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              <p className="leading-relaxed text-[11px] text-slate-400">
                Projects sparse LiDAR onto image plane; min-pools to stride 8.
              </p>
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">D=48 Depth Bins:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">[1.0m to 69.0m] with 0.1 aux CE loss</p>
              </div>
              <p className="text-[11px] text-emerald-400 font-medium">
                Eliminates ray feature smearing
              </p>
            </div>
          </div>

          {/* Step 3: BEV Fusion */}
          <div className="space-y-4 p-5 rounded-xl bg-dark-900/80 border border-dark-700/80">
            <div className="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <Cpu className="w-4 h-4" />
              <span>3. ConvFuser Module</span>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              <p className="leading-relaxed text-[11px] text-slate-400">
                Symmetric channel concatenation in shared Bird's-Eye-View.
              </p>
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">2x Conv-BN-ReLU:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">(64 cam + 32 LiDAR = 96) -&gt; 64 ch</p>
              </div>
              <p className="text-[11px] text-slate-400">
                Pure PyTorch scatter-add
              </p>
            </div>
          </div>

          {/* Step 4: Head & Prediction */}
          <div className="space-y-4 p-5 rounded-xl bg-dark-900/80 border border-dark-700/80">
            <div className="flex items-center gap-2 text-brand-400 font-bold text-sm">
              <ShieldCheck className="w-4 h-4" />
              <span>4. Head & Output</span>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">BaseBEVBackbone:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">2-stage downsample & upsample</p>
              </div>
              <div className="p-2.5 rounded bg-dark-800 border border-dark-700">
                <span className="font-semibold text-white">AnchorHeadSingle:</span>
                <p className="text-slate-400 text-[11px] mt-0.5">Car, Pedestrian, Cyclist 3D boxes</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Target Criteria Status Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">Verified Acceptance Targets</h2>
            <p className="text-xs text-slate-400">Strictly audited against predefined criteria without modifying thresholds.</p>
          </div>
          <Link href="/results" className="text-xs text-brand-400 hover:text-brand-300 font-medium flex items-center gap-1">
            <span>Full Results</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {targetsData.targets.map((t, idx) => (
            <div key={idx} className="p-5 rounded-xl bg-dark-800/70 border border-dark-700 flex flex-col justify-between">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">{t.name}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    {t.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">{t.threshold}</p>
              </div>
              <div className="mt-4 pt-3 border-t border-dark-700/60 flex items-center justify-between text-xs">
                <span className="text-slate-500">Measured:</span>
                <span className="font-semibold text-brand-400">{t.measured}</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
