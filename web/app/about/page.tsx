"use client";

import { Shield, AlertTriangle, Users, BookOpen, ExternalLink, Scale, CheckCircle } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Title */}
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">About &amp; Scientific Limitations</h1>
        <p className="text-xs text-slate-400 mt-1">
          Honest accounting of research constraints, formal dataset licenses, and project authorship.
        </p>
      </div>

      {/* Honest Limitations Section */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
          Honest Scientific Limitations
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
            <h3 className="font-bold text-white text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              Single Front Camera &amp; Asymmetric Field of View
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              The primary KITTI benchmark provides only a single forward-facing camera (90-degree FOV) synchronized with 360-degree Velodyne LiDAR. Objects outside the camera FOV rely strictly on LiDAR point features, creating an asymmetric perception field. Full 360-degree multi-camera surround evaluation was validated on nuScenes-mini, but cannot be inferred directly from KITTI.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
            <h3 className="font-bold text-white text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              Pinhole Calibration Sensitivity
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              BEV lifting relies on rigid calibration matrices (T_lidar_to_cam, P2, R0_rect). Any physical sensor miscalibration, vehicle flex, or pitch drift degrades the depth-guided projection into BEV cells. We verified mathematical calibration intersection at &ge; 0.90 (measured 0.9182), but online self-calibration is left to future work.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
            <h3 className="font-bold text-white text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              nuScenes-mini Data Scale Caveat
            </h3>
            <blockquote className="text-xs text-slate-300 italic border-l-2 border-brand-500/50 pl-3 leading-relaxed">
              &quot;nuScenes-mini contains only 10 scenes (323 train samples, 81 val samples). While multi-camera fusion mechanisms function identically to full-scale training, metrics are subject to small-sample variance (&plusmn;2.1 mAP). Reported numbers serve as a cross-modal proof-of-concept, not a full leaderboard benchmark.&quot;
            </blockquote>
          </div>

          <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
            <h3 className="font-bold text-white text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              Target Acceptance &amp; Partial Results
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              All 6 acceptance targets were strictly measured and met on an NVIDIA T4 GPU. We adhere to the project constraint: never move a target, report honest partial results, and document exact compute hardware. We make no claims of real-time performance on lower-tier hardware (e.g. Jetson Nano) without FP16 or INT8 TensorRT quantization.
            </p>
          </div>
        </div>
      </div>

      {/* Dataset Licenses & Attributions */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Scale className="w-5 h-5 text-brand-400" />
          Formal Dataset Licensing &amp; Fair Use Notice
        </h2>

        <div className="p-6 rounded-xl bg-dark-800/60 border border-dark-700 space-y-4">
          <p className="text-xs text-slate-300 leading-relaxed">
            This repository and research website do <span className="font-bold text-white">NOT</span> redistribute raw dataset archives or bulk point clouds. In compliance with the non-commercial licenses of both KITTI and nuScenes, only a minimal, downsampled validation subset (50 cached frames) is packaged strictly for demonstration, evaluation, and academic review:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white">KITTI Vision Benchmark Suite</span>
                <span className="text-[10px] font-mono text-amber-400">CC BY-NC-SA 3.0</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Geiger, Lenz, Stiller, Urtasun (KIT / TTIC). Used under Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Raw annotations and lidar files are downloaded directly from the official KITTI portal by end users during reproduction.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-dark-900 border border-dark-700/80 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white">nuScenes Dataset</span>
                <span className="text-[10px] font-mono text-amber-400">CC BY-NC-SA 4.0</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Caesar et al. (Motional). Used under Creative Commons Attribution-NonCommercial-ShareAlike 4.0. Only public metadata and bounding box geometries are utilized for benchmarking.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Project Authors & Credits */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Users className="w-5 h-5 text-brand-400" />
          Authorship &amp; Acknowledgements
        </h2>

        <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-dark-700/60 pb-4">
            <div>
              <div className="text-base font-bold text-white">Vineet Raj Ojha</div>
              <div className="text-xs text-slate-400">Lead Researcher &amp; Author &middot; Capstone BT-177</div>
            </div>
            <a
              href="https://github.com/vineetrajojha/BT-177_Capstone"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-dark-700 hover:bg-dark-600 text-xs font-medium text-white transition-colors w-fit"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>Project GitHub Repository</span>
            </a>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            BEVFusionLite was engineered as an undergraduate capstone project (BT-177) investigating lightweight multi-modal sensor fusion algorithms for compute-constrained autonomous platforms. We express deep gratitude to the open-source autonomous driving community, especially the maintainers of <span className="font-mono text-brand-400">OpenPCDet</span>, <span className="font-mono text-brand-400">PyTorch</span>, and <span className="font-mono text-brand-400">Lift-Splat-Shoot</span>.
          </p>
        </div>
      </div>
    </div>
  );
}
