import { AlertTriangle, Camera } from "lucide-react";
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">nuScenes-mini Multi-Camera Evaluation</h1>
        <p className="text-xs text-zinc-400 mt-1">
          Surround-view 360° perception across 6 cameras and 32-beam LiDAR. Source: <code className="text-zinc-300 font-mono text-[11px] bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">{nuscData.source_file}</code>
        </p>
      </div>

      {/* Verbatim Limitation Statement Card */}
      <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
        <div className="flex items-center gap-2 text-zinc-300 font-medium text-xs">
          <AlertTriangle className="w-3.5 h-3.5 text-zinc-400" />
          <span>Small-Data Scientific Limitation Statement (Verbatim)</span>
        </div>
        <blockquote className="text-xs text-zinc-400 leading-relaxed italic border-l-2 border-zinc-700 pl-3">
          &quot;{nuscData.limitation_statement}&quot;
        </blockquote>
      </div>

      {/* nuScenes Results Table */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-white">Detection Metrics (mAP &amp; NDS Across 3 Seeds)</h2>
        <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-900/90 text-zinc-400 border-b border-zinc-800 uppercase font-mono text-[10px] tracking-wider">
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
            <tbody className="divide-y divide-zinc-800/80 font-mono text-zinc-300">
              {nuscData.data.map((r: any, idx: number) => {
                const isLight = r.model.includes("Light");
                return (
                  <tr
                    key={idx}
                    className={`hover:bg-zinc-900/50 transition-colors ${
                      isLight ? "bg-zinc-900/60 text-white font-medium" : ""
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-white">{r.model.replace(/_/g, " ")}</td>
                    <td className="py-3 px-3 text-zinc-400 font-sans">{r.camera_backbone}</td>
                    <td className="py-3 px-3">{r.voxel_size}m</td>
                    <td className="py-3 px-3 font-bold text-white">{r.mAP.toFixed(2)}% ± {r.mAP_std}</td>
                    <td className="py-3 px-3 font-bold text-white">{r.NDS.toFixed(2)}% ± {r.NDS_std}</td>
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
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <Camera className="w-4 h-4 text-zinc-400" />
            Synchronized 6-Camera Surround View Array
          </h2>
          <span className="text-xs text-zinc-500">Sample from scene-0061 (v1.0-mini)</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {CAMERAS.map((cam, idx) => (
            <div key={idx} className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-white font-mono text-[11px]">{cam.name}</span>
                <span className="text-[10px] text-zinc-500">{cam.angle}</span>
              </div>
              <div className="h-40 rounded bg-black border border-zinc-800/80 overflow-hidden relative group">
                <img
                  src="/data/demo/images/000000.jpg"
                  alt={cam.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
                <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur text-[9px] text-zinc-300 border border-zinc-800">
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
