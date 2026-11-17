"use client";

import { useState } from "react";
import { Filter, Layers, Info, CheckCircle2, AlertCircle } from "lucide-react";
import ablationData from "../../public/data/ablation_table.json";

export default function AblationsPage() {
  const [selectedAxis, setSelectedAxis] = useState<string>("All");

  const axes = ["All", ...Array.from(new Set(ablationData.data.map((r: any) => r.axis)))];

  const filtered = selectedAxis === "All"
    ? ablationData.data
    : ablationData.data.filter((r: any) => r.axis === selectedAxis);

  const refCar = 76.10;
  const refCarStd = 0.28;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">6-Axis Architectural Ablations</h1>
        <p className="text-xs text-slate-400 mt-1">
          Evaluating isolated components across 3 seeds on KITTI val (AP R40). Hardware: NVIDIA T4. Source: <code className="text-brand-400 font-mono">{ablationData.source_file}</code>
        </p>
      </div>

      {/* Axis Filter Pills */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-400 flex items-center gap-1.5 mr-2">
          <Filter className="w-3.5 h-3.5" /> Filter Axis:
        </span>
        {axes.map((ax) => (
          <button
            key={ax}
            onClick={() => setSelectedAxis(ax)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${
              selectedAxis === ax
                ? "bg-brand-500 text-dark-900 shadow-sm"
                : "bg-dark-800 text-slate-300 hover:bg-dark-700 border border-dark-700"
            }`}
          >
            {ax}
          </button>
        ))}
      </div>

      {/* Main Ablations Table */}
      <div className="overflow-x-auto rounded-xl border border-dark-700 bg-dark-800/40">
        <table className="w-full text-left text-xs">
          <thead className="bg-dark-800/90 text-slate-400 border-b border-dark-700 uppercase font-mono text-[10px] tracking-wider">
            <tr>
              <th className="py-3 px-4 font-semibold">Axis</th>
              <th className="py-3 px-4 font-semibold">Variant</th>
              <th className="py-3 px-3 font-semibold">Params (M)</th>
              <th className="py-3 px-3 font-semibold">Latency (ms)</th>
              <th className="py-3 px-3 font-semibold">FPS (T4)</th>
              <th className="py-3 px-3 font-semibold">Car 3D (Mod)</th>
              <th className="py-3 px-3 font-semibold">Ped 3D (Mod)</th>
              <th className="py-3 px-3 font-semibold">Cyc 3D (Mod)</th>
              <th className="py-3 px-4 font-semibold">Within 1 Std of Ref?</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-700/60 font-mono text-slate-300">
            {filtered.map((r: any, idx: number) => {
              const isRef = r.variant.includes("(Ref)");
              const diff = Math.abs(r.car_3d_mod - refCar);
              const withinStd = isRef ? "REF" : (diff <= (r.car_3d_std + refCarStd) ? "YES" : "NO");

              return (
                <tr
                  key={idx}
                  className={`hover:bg-dark-700/30 transition-colors ${
                    isRef ? "bg-brand-500/10 font-bold text-white" : ""
                  }`}
                >
                  <td className="py-3 px-4 font-sans font-medium text-slate-400">{r.axis}</td>
                  <td className="py-3 px-4 font-sans font-semibold text-white flex items-center gap-1.5">
                    {r.variant}
                    {isRef && <span className="text-[10px] text-brand-400 bg-brand-500/20 px-1.5 py-0.5 rounded border border-brand-500/30">Reference</span>}
                  </td>
                  <td className="py-3 px-3">{r.params_M}</td>
                  <td className="py-3 px-3">{r.latency_ms}</td>
                  <td className="py-3 px-3 text-brand-400">{r.fps}</td>
                  <td className="py-3 px-3 text-white font-bold">{r.car_3d_mod.toFixed(2)} ± {r.car_3d_std.toFixed(2)}</td>
                  <td className="py-3 px-3">{r.ped_3d_mod.toFixed(2)} ± {r.ped_3d_std.toFixed(2)}</td>
                  <td className="py-3 px-3">{r.cyc_3d_mod.toFixed(2)} ± {r.cyc_3d_std.toFixed(2)}</td>
                  <td className="py-3 px-4 font-sans">
                    {isRef ? (
                      <span className="text-slate-400 font-semibold text-[11px]">—</span>
                    ) : withinStd === "YES" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30">
                        <CheckCircle2 className="w-3 h-3" /> Within 1 Std
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                        <AlertCircle className="w-3 h-3" /> Stat. Significant
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* AP vs Latency Figure & Written Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Figure Card */}
        <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">AP vs Latency Pareto Frontier</h2>
            <span className="text-[11px] text-slate-400">NVIDIA T4 Profile</span>
          </div>
          <div className="rounded-lg overflow-hidden border border-dark-700 bg-dark-900 p-2">
            <img
              src="/data/ap_vs_latency.png"
              alt="AP vs Latency Pareto Frontier"
              className="w-full h-auto rounded"
            />
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            The vertical dashed line denotes the 20 FPS real-time automotive boundary (50 ms). BEVFusionLite Light (32.1 ms) and FP16 (22.3 ms) reside comfortably inside the real-time operational window.
          </p>
        </div>

        {/* Written Scientific Analysis */}
        <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-5 text-xs text-slate-300">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Empirical Ablation Analysis</h2>

          <div className="space-y-2">
            <div className="font-semibold text-brand-400">1. Depth Guidance is the Primary Accuracy Lever</div>
            <p className="leading-relaxed text-slate-400">
              Disabling LiDAR depth guidance degrades Car 3D moderate AP from <strong className="text-white">76.10 down to 72.80 (-3.30 AP)</strong>, and Pedestrian from <strong className="text-white">55.10 to 49.30 (-5.80 AP)</strong>. This exceeds 3 standard deviations, confirming that unguided LSS disperses visual representations in 3D space.
            </p>
          </div>

          <div className="space-y-2">
            <div className="font-semibold text-brand-400">2. Discretization Bins: D=48 Saturates Performance</div>
            <p className="leading-relaxed text-slate-400">
              Scaling depth bins from 24 to 48 yields a +1.50 Car AP improvement. Increasing bins further to 64 yields a marginal delta of +0.15 AP (strictly within 1 std: 0.28) while incurring +5.5 ms latency.
            </p>
          </div>

          <div className="space-y-2">
            <div className="font-semibold text-brand-400">3. Mixed Precision Acceleration</div>
            <p className="leading-relaxed text-slate-400">
              Switching from FP32 to FP16 mixed precision accelerates edge throughput from <strong className="text-white">31.2 FPS to 44.8 FPS (+43.6% gain)</strong> with virtually identical detection precision (76.10 vs 76.05 AP).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
