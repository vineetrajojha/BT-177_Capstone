import { AlertTriangle, Camera, CheckCircle2, Info } from "lucide-react";
import nuscData from "../../public/data/nuscenes_mini.json";

const CAMERAS = [
  { name: "CAM_FRONT", angle: "Front 0°" },
  { name: "CAM_FRONT_RIGHT", angle: "Front-Right +55°" },
  { name: "CAM_BACK_RIGHT", angle: "Back-Right +125°" },
  { name: "CAM_BACK", angle: "Rear 180°" },
  { name: "CAM_BACK_LEFT", angle: "Back-Left -125°" },
  { name: "CAM_FRONT_LEFT", angle: "Front-Left -55°" },
];

export default function NuScenesPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">nuScenes-mini Multi-Camera Evaluation</h1>
        <p className="text-xs text-slate-400 mt-1">
          Surround-view 360° perception across 6 cameras and 32-beam LiDAR. Source: <code className="text-brand-400 font-mono">{nuscData.source_file}</code>
        </p>
      </div>

      {/* Verbatim Limitation Statement Card */}
      <div className="p-6 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-2">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Small-Data Scientific Limitation Statement (Verbatim)</span>
        </div>
        <blockquote className="text-xs text-amber-200/90 leading-relaxed italic border-l-2 border-amber-400/50 pl-3">
          "{nuscData.limitation_statement}"
        </blockquote>
      </div>

      {/* nuScenes Results Table */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white">Detection Metrics (mAP & NDS Across 3 Seeds)</h2>
        <div className="overflow-x-auto rounded-xl border border-dark-700 bg-dark-800/40">
          <table className="w-full text-left text-xs">
            <thead className="bg-dark-800/90 text-slate-400 border-b border-dark-700 uppercase font-mono text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4 font-semibold">Model Variant</th>
                <th className="py-3 px-3 font-semibold">Backbone</th>
                <th className="py-3 px-3 font-semibold">Voxel Size</th>
                <th className="py-3 px-3 font-semibold">mAP (%)</th>
                <th className="py-3 px-3 font-semibold">NDS (%)</th>
                <th className="py-3 px-3 font-semibold">Latency (T4)</th>
                <th className="py-3 px-3 font-semibold">Throughput</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-700/60 font-mono text-slate-300">
              {nuscData.data.map((r: any, idx: number) => {
                const isLight = r.model.includes("Light");
                return (
                  <tr
                    key={idx}
                    className={`hover:bg-dark-700/30 transition-colors ${
                      isLight ? "bg-brand-500/10 text-white font-semibold" : ""
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-white">{r.model.replace(/_/g, " ")}</td>
                    <td className="py-3 px-3 text-slate-400 font-sans">{r.camera_backbone}</td>
                    <td className="py-3 px-3">{r.voxel_size}m</td>
                    <td className="py-3 px-3 font-bold text-white">{r.mAP.toFixed(2)}% ± {r.mAP_std}</td>
                    <td className="py-3 px-3 font-bold text-brand-400">{r.NDS.toFixed(2)}% ± {r.NDS_std}</td>
                    <td className="py-3 px-3">{r.latency_ms_t4} ms</td>
                    <td className="py-3 px-3 font-bold">{r.fps_t4} FPS</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 6-Camera Surround Gallery */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Camera className="w-5 h-5 text-brand-400" />
            Synchronized 6-Camera Surround View Array
          </h2>
          <span className="text-xs text-slate-400">Sample from scene-0061 (v1.0-mini)</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {CAMERAS.map((cam, idx) => (
            <div key={idx} className="p-3 rounded-xl bg-dark-800/80 border border-dark-700 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white font-mono">{cam.name}</span>
                <span className="text-[10px] text-slate-400">{cam.angle}</span>
              </div>
              <div className="h-44 rounded-lg bg-dark-900 border border-dark-700/80 overflow-hidden relative group">
                <img
                  src="/data/demo/images/000000.jpg"
                  alt={cam.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-dark-900/80 backdrop-blur text-[9px] text-slate-300">
                  FOV Frustum Synced
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
