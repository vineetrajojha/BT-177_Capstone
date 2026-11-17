"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Play, Pause, ChevronLeft, ChevronRight, Layers, Eye, Cpu, Zap, Activity, Info, Upload } from "lucide-react";
import demoManifest from "../../public/data/demo/manifest.json";

// Dynamic import Three.js viewer with SSR disabled
const ThreePointCloudViewer = dynamic(
  () => import("@/components/ThreePointCloudViewer"),
  { ssr: false }
);

export default function DemoPage() {
  const [frameIndex, setFrameIndex] = useState(0);
  const [modelType, setModelType] = useState<"light" | "heavy" | "side_by_side">("light");
  const [scoreThreshold, setScoreThreshold] = useState(0.40);
  const [showGT, setShowGT] = useState(true);
  const [selectedClasses, setSelectedClasses] = useState<string[]>(["Car", "Pedestrian", "Cyclist"]);
  const [frameData, setFrameData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);

  const currentFrameId = demoManifest.frames[frameIndex]?.frame_id || "000000";

  // Load frame JSON
  useEffect(() => {
    setLoading(true);
    fetch(`/data/demo/frames/${currentFrameId}.json`)
      .then((res) => res.json())
      .then((data) => {
        setFrameData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading frame data:", err);
        setLoading(false);
      });
  }, [currentFrameId]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % demoManifest.frames.length);
    }, 1500);
    return () => clearInterval(interval);
  }, [isPlaying]);

  const toggleClass = (cls: string) => {
    setSelectedClasses((prev) =>
      prev.includes(cls) ? prev.filter((c) => c !== cls) : [...prev, cls]
    );
  };

  const activePredictions = frameData
    ? (modelType === "heavy" ? frameData.predictions.heavy : frameData.predictions.light)
    : [];

  const efficiency = modelType === "heavy" ? demoManifest.models.heavy : demoManifest.models.light;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-dark-700/80 pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">Interactive 3D Multi-Modal Demo</h1>
          <p className="text-xs text-slate-400 mt-1">
            Precomputed validation frames with synchronized 3D point cloud, camera projections, and real-time filtering.
          </p>
        </div>

        {/* Model Switcher */}
        <div className="inline-flex rounded-lg bg-dark-800 p-1 border border-dark-700">
          <button
            onClick={() => setModelType("light")}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              modelType === "light"
                ? "bg-brand-500 text-dark-900 shadow-sm"
                : "text-slate-300 hover:text-white"
            }`}
          >
            Light Model (ResNet-18)
          </button>
          <button
            onClick={() => setModelType("heavy")}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              modelType === "heavy"
                ? "bg-brand-500 text-dark-900 shadow-sm"
                : "text-slate-300 hover:text-white"
            }`}
          >
            Heavy Baseline (ResNet-50)
          </button>
        </div>
      </div>

      {/* Control Bar */}
      <div className="p-4 rounded-xl bg-dark-800/80 border border-dark-700 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
        {/* Frame Navigator */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Frame: <strong className="text-white font-mono">{currentFrameId}</strong></span>
            <span>({frameIndex + 1} / {demoManifest.total_frames})</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 rounded bg-dark-700 hover:bg-dark-600 text-slate-200"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
            </button>
            <button
              onClick={() => setFrameIndex((prev) => (prev > 0 ? prev - 1 : demoManifest.total_frames - 1))}
              className="p-1.5 rounded bg-dark-700 hover:bg-dark-600 text-slate-200"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <input
              type="range"
              min={0}
              max={demoManifest.total_frames - 1}
              value={frameIndex}
              onChange={(e) => setFrameIndex(parseInt(e.target.value))}
              className="w-full accent-brand-500 cursor-pointer"
            />
            <button
              onClick={() => setFrameIndex((prev) => (prev + 1) % demoManifest.total_frames)}
              className="p-1.5 rounded bg-dark-700 hover:bg-dark-600 text-slate-200"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Score Threshold Slider */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Confidence Threshold</span>
            <span className="text-white font-mono">{scoreThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.10}
            max={0.90}
            step={0.05}
            value={scoreThreshold}
            onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
            className="w-full accent-brand-500 cursor-pointer"
          />
        </div>

        {/* Class Filter Toggles */}
        <div className="space-y-1">
          <div className="text-xs text-slate-400">Class Filters</div>
          <div className="flex items-center gap-1.5">
            {[
              { name: "Car", color: "bg-emerald-500" },
              { name: "Pedestrian", color: "bg-blue-500" },
              { name: "Cyclist", color: "bg-orange-500" },
            ].map((c) => {
              const active = selectedClasses.includes(c.name);
              return (
                <button
                  key={c.name}
                  onClick={() => toggleClass(c.name)}
                  className={`px-2.5 py-1 rounded text-[11px] font-semibold border flex items-center gap-1.5 transition-all ${
                    active
                      ? "bg-dark-700 text-white border-dark-600"
                      : "bg-dark-900/60 text-slate-500 border-dark-800 line-through"
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full ${c.color}`} />
                  {c.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* GT Toggle */}
        <div className="flex items-center justify-end">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
            <input
              type="checkbox"
              checked={showGT}
              onChange={(e) => setShowGT(e.target.checked)}
              className="rounded bg-dark-700 border-dark-600 text-brand-500 focus:ring-0 cursor-pointer"
            />
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block" />
              Show Ground Truth (GT)
            </span>
          </label>
        </div>
      </div>

      {/* Main Panels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 1: Interactive 3D Point Cloud */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-brand-400" />
              Interactive 3D LiDAR Point Cloud (WebGL / Three.js)
            </span>
            <span className="text-slate-500 font-mono">
              {frameData ? `${frameData.num_points.toLocaleString()} points` : "Loading..."}
            </span>
          </div>

          {frameData ? (
            <ThreePointCloudViewer
              points={frameData.point_cloud}
              predictedBoxes={activePredictions}
              gtBoxes={frameData.gt_boxes}
              showGT={showGT}
              scoreThreshold={scoreThreshold}
              classFilter={selectedClasses}
            />
          ) : (
            <div className="w-full h-[450px] rounded-xl bg-dark-800 animate-pulse flex items-center justify-center text-slate-500 text-xs">
              Loading point cloud...
            </div>
          )}
        </div>

        {/* Panel 2: Camera 2D Image with Projected 3D Wireframes */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-brand-400" />
              Camera Projection Overlay (Frontal FOV)
            </span>
            <span className="text-slate-500 font-mono">Pinhole Rectified P2</span>
          </div>

          <div className="relative w-full h-[450px] rounded-xl overflow-hidden border border-dark-700 bg-dark-900 flex items-center justify-center">
            {frameData ? (
              <img
                src={showGT ? frameData.overlay_url : frameData.image_url}
                alt={`Frame ${currentFrameId}`}
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-slate-500 text-xs">Loading camera image...</div>
            )}

            <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-dark-900/80 backdrop-blur border border-dark-700 text-[10px] text-slate-300">
              {modelType === "heavy" ? "Heavy (ResNet-50)" : "Light (ResNet-18)"}
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Card */}
      <div className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 grid grid-cols-2 sm:grid-cols-4 gap-6">
        <div>
          <div className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">Inference Latency</div>
          <div className="text-2xl font-black text-white mt-1">{efficiency.latency_ms} ms</div>
          <div className="text-[10px] text-slate-500 mt-0.5">NVIDIA T4, batch size 1</div>
        </div>

        <div>
          <div className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">Edge Throughput</div>
          <div className="text-2xl font-black text-brand-400 mt-1">{efficiency.fps} FPS</div>
          <div className="text-[10px] text-slate-500 mt-0.5">&gt; 20 FPS real-time pass</div>
        </div>

        <div>
          <div className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">Parameter Footprint</div>
          <div className="text-2xl font-black text-white mt-1">{efficiency.params_M} M</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Camera + LiDAR + Head</div>
        </div>

        <div>
          <div className="text-[11px] uppercase font-bold text-slate-400 tracking-wider">LiDAR Grid Granularity</div>
          <div className="text-2xl font-black text-white mt-1">{efficiency.voxel_size}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Voxel pillar dimension</div>
        </div>
      </div>
    </div>
  );
}
