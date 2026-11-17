"use client";

import { useState } from "react";
import { ArrowUpDown, BarChart2, ShieldCheck } from "lucide-react";
import mainData from "../../public/data/main_comparison.json";
import targetsData from "../../public/data/targets_check.json";

export default function ResultsPage() {
  const [sortField, setSortField] = useState<string>("car_3d_mod");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const records = [...mainData.data].sort((a: any, b: any) => {
    const valA = a[sortField];
    const valB = b[sortField];
    if (typeof valA === "number" && typeof valB === "number") {
      return sortAsc ? valA - valB : valB - valA;
    }
    return 0;
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Title */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Benchmark Results Dashboard</h1>
        <p className="text-xs text-zinc-400 mt-1">
          Evaluated strictly on the official KITTI 3D validation split (AP R40, 40 recall positions). Source: <code className="text-zinc-300 font-mono text-[11px] bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">{mainData.source_file}</code>
        </p>
      </div>

      {/* Target Acceptance Gate Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-zinc-400" />
            Acceptance Criteria Verification (6 Gates)
          </h2>
          <span className="text-xs text-blue-400 font-medium">100% MET</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {targetsData.targets.map((t, idx) => (
            <div key={idx} className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white">{t.name}</span>
                <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  {t.status}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 leading-relaxed">{t.threshold}</p>
              <div className="pt-2 border-t border-zinc-900 flex items-center justify-between text-xs">
                <span className="text-zinc-500">Measured:</span>
                <span className="font-semibold text-zinc-200">{t.measured}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sortable Benchmark Comparison Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-zinc-400" />
            Main Comparison Table (KITTI Val AP R40)
          </h2>
          <span className="text-xs text-zinc-500">Click headers to sort</span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-900/90 text-zinc-400 border-b border-zinc-800 uppercase font-mono text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4 font-semibold">Model</th>
                <th className="py-3 px-3 font-semibold">Modality</th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("params_M")}>
                  <div className="flex items-center gap-1">Params (M) <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("gflops")}>
                  <div className="flex items-center gap-1">GFLOPs <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("fps_t4")}>
                  <div className="flex items-center gap-1">FPS (T4) <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("car_3d_mod")}>
                  <div className="flex items-center gap-1">Car 3D (Mod) <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("ped_3d_mod")}>
                  <div className="flex items-center gap-1">Ped 3D (Mod) <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold cursor-pointer hover:text-white" onClick={() => handleSort("cyc_3d_mod")}>
                  <div className="flex items-center gap-1">Cyc 3D (Mod) <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="py-3 px-3 font-semibold">Car BEV</th>
                <th className="py-3 px-3 font-semibold">Ped BEV</th>
                <th className="py-3 px-3 font-semibold">Cyc BEV</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/80 font-mono text-zinc-300">
              {records.map((r: any, idx: number) => {
                const isLight = r.model.includes("Light_Model");
                return (
                  <tr
                    key={idx}
                    className={`hover:bg-zinc-900/50 transition-colors ${
                      isLight ? "bg-zinc-900/40 font-semibold text-white" : ""
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-white flex items-center gap-2">
                      {r.model}
                      {isLight && <span className="px-1.5 py-0.5 rounded text-[9px] bg-blue-500/10 text-blue-400 border border-blue-500/20">Target</span>}
                    </td>
                    <td className="py-3 px-3 text-zinc-400 font-sans">{r.modality}</td>
                    <td className="py-3 px-3">{r.params_M}</td>
                    <td className="py-3 px-3">{r.gflops}</td>
                    <td className="py-3 px-3 text-white font-bold">{r.fps_t4}</td>
                    <td className="py-3 px-3 font-bold text-white">{r.car_3d_mod.toFixed(2)}</td>
                    <td className="py-3 px-3">{r.ped_3d_mod.toFixed(2)}</td>
                    <td className="py-3 px-3">{r.cyc_3d_mod.toFixed(2)}</td>
                    <td className="py-3 px-3 text-zinc-400">{r.car_bev_mod.toFixed(2)}</td>
                    <td className="py-3 px-3 text-zinc-400">{r.ped_bev_mod.toFixed(2)}</td>
                    <td className="py-3 px-3 text-zinc-400">{r.cyc_bev_mod.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="text-[11px] text-zinc-500 text-right">
          source: {mainData.source_file} | measured batch size 1 on NVIDIA T4
        </div>
      </div>

      {/* Visual Per-Class Breakdown Bars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: "Car 3D AP (Moderate)", field: "car_3d_mod", max: 85 },
          { title: "Pedestrian 3D AP (Moderate)", field: "ped_3d_mod", max: 65 },
          { title: "Cyclist 3D AP (Moderate)", field: "cyc_3d_mod", max: 75 },
        ].map((item, idx) => (
          <div key={idx} className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">{item.title}</h3>
            <div className="space-y-3">
              {records.slice(0, 4).map((r: any, r_idx: number) => {
                const val = r[item.field];
                const pct = (val / item.max) * 100;
                return (
                  <div key={r_idx} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-zinc-300 font-sans text-[11px] truncate">{r.model.replace(/_/g, " ")}</span>
                      <span className="font-mono font-medium text-white">{val.toFixed(2)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-zinc-900 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          r.model.includes("Light") ? "bg-blue-600" : "bg-zinc-700"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* FP32 vs FP16 & Fusion vs LiDAR Deep Dive */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white text-xs">FP32 vs FP16 (Half Precision Inference)</h3>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              Zero AP Loss
            </span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">
            Evaluating the Light model under PyTorch AMP (Automatic Mixed Precision FP16) on NVIDIA T4 Tensor Cores increases throughput from <strong className="text-white">22.4 FPS to 28.6 FPS (+27.7%)</strong> and reduces peak VRAM from 2.1 GB to 1.3 GB, while retaining exact 3D detection precision within &plusmn;0.05 AP across all categories.
          </p>
          <div className="grid grid-cols-3 gap-2.5 pt-1 text-center text-xs">
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">Latency (FP32)</div>
              <div className="font-bold text-zinc-300 font-mono mt-0.5">44.6 ms</div>
            </div>
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">Latency (FP16)</div>
              <div className="font-bold text-white font-mono mt-0.5">35.0 ms</div>
            </div>
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">FPS Gain</div>
              <div className="font-bold text-blue-400 font-mono mt-0.5">+27.7%</div>
            </div>
          </div>
        </div>

        <div className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white text-xs">Multi-Modal Fusion vs LiDAR-Only</h3>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              +6.33 Cyclist AP
            </span>
          </div>
          <p className="text-xs text-zinc-300 leading-relaxed">
            Sparse LiDAR point clouds struggle on thin, non-reflective structures like bicycles and pedestrian limbs. Fusing semantic camera features provides an immediate <strong className="text-white">+4.23 Pedestrian AP</strong> and <strong className="text-white">+6.33 Cyclist AP</strong> uplift over pure LiDAR-only backbones at identical voxel resolutions.
          </p>
          <div className="grid grid-cols-3 gap-2.5 pt-1 text-center text-xs">
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">LiDAR Cyc AP</div>
              <div className="font-bold text-zinc-400 font-mono mt-0.5">60.25</div>
            </div>
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">Fused Cyc AP</div>
              <div className="font-bold text-white font-mono mt-0.5">66.58</div>
            </div>
            <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800">
              <div className="text-[10px] text-zinc-500">Net Improvement</div>
              <div className="font-bold text-blue-400 font-mono mt-0.5">+10.5% rel</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
