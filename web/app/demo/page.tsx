"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Play, Pause, ChevronLeft, ChevronRight, Layers, Eye, Activity } from "lucide-react";
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Interactive 3D Multi-Modal Demo</h1>
          <p className="text-xs text-zinc-400 mt-1">
            Precomputed validation frames with synchronized 3D point cloud, camera projections, and real-time filtering.
          </p>
        </div>

        {/* Model Switcher */}
        <div className="inline-flex rounded-md bg-zinc-900 p-1 border border-zinc-800">
          <button
            onClick={() => setModelType("light")}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
              modelType === "light"
                ? "bg-blue-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Light Model (ResNet-18)
          </button>
          <button
            onClick={() => setModelType("heavy")}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
              modelType === "heavy"
                ? "bg-blue-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Heavy Baseline (ResNet-50)
          </button>
        </div>
      </div>

      {/* Control Bar */}
      <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
        {/* Frame Navigator */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-400">
            <span>Frame: <strong className="text-white font-mono">{currentFrameId}</strong></span>
            <span>({frameIndex + 1} / {demoManifest.total_frames})</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            </button>
            <button
              onClick={() => setFrameIndex((prev) => (prev > 0 ? prev - 1 : demoManifest.total_frames - 1))}
              className="p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <input
              type="range"
              min={0}
              max={demoManifest.total_frames - 1}
              value={frameIndex}
              onChange={(e) => setFrameIndex(parseInt(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer"
            />
            <button
              onClick={() => setFrameIndex((prev) => (prev + 1) % demoManifest.total_frames)}
              className="p-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Score Threshold Slider */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-400">
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
            className="w-full accent-blue-600 cursor-pointer"
          />
        </div>

        {/* Class Filter Toggles */}
        <div className="space-y-1">
          <div className="text-xs text-zinc-400">Class Filters</div>
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
                  className={`px-2 py-0.5 rounded text-[11px] font-medium border flex items-center gap-1.5 transition-all ${
                    active
                      ? "bg-zinc-900 text-white border-zinc-700"
                      : "bg-zinc-950 text-zinc-600 border-zinc-900 line-through"
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
        <div className="flex items-center md:justify-end">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-zinc-300">
            <input
              type="checkbox"
              checked={showGT}
              onChange={(e) => setShowGT(e.target.checked)}
              className="rounded bg-zinc-900 border-zinc-700 text-blue-600 focus:ring-0 cursor-pointer"
            />
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-yellow-400 inline-block" />
              Show Ground Truth (GT)
            </span>
          </label>
        </div>
      </div>

      {/* Main Panels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Panel 1: Interactive 3D Point Cloud */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-300">
            <span className="flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-zinc-400" />
              Interactive 3D LiDAR Point Cloud (WebGL / Three.js)
            </span>
            <span className="text-zinc-500 font-mono text-[11px]">
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
            <div className="w-full h-[450px] rounded-lg bg-zinc-950 border border-zinc-800 animate-pulse flex items-center justify-center text-zinc-500 text-xs">
              Loading point cloud...
            </div>
          )}
        </div>

        {/* Panel 2: Camera 2D Image with Projected 3D Wireframes */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-300">
            <span className="flex items-center gap-2">
              <Eye className="w-3.5 h-3.5 text-zinc-400" />
              Camera Projection Overlay (Frontal FOV)
            </span>
            <span className="text-zinc-500 font-mono text-[11px]">Pinhole Rectified P2</span>
          </div>

          <div className="relative w-full h-[450px] rounded-lg overflow-hidden border border-zinc-800 bg-zinc-950 flex items-center justify-center">
            {frameData ? (
              <img
                src={showGT ? frameData.overlay_url : frameData.image_url}
                alt={`Frame ${currentFrameId}`}
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-zinc-500 text-xs">Loading camera image...</div>
            )}

            <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/80 backdrop-blur border border-zinc-800 text-[10px] text-zinc-300">
              {modelType === "heavy" ? "Heavy (ResNet-50)" : "Light (ResNet-18)"}
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Card */}
      <div className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div>
          <div className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider">Inference Latency</div>
          <div className="text-xl font-bold text-white mt-0.5">{efficiency.latency_ms} ms</div>
          <div className="text-[10px] text-zinc-500">NVIDIA T4, batch size 1</div>
        </div>

        <div>
          <div className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider">Edge Throughput</div>
          <div className="text-xl font-bold text-white mt-0.5">{efficiency.fps} FPS</div>
          <div className="text-[10px] text-blue-400 font-medium">&gt; 20 FPS real-time pass</div>
        </div>

        <div>
          <div className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider">Parameter Footprint</div>
          <div className="text-xl font-bold text-white mt-0.5">{efficiency.params_M} M</div>
          <div className="text-[10px] text-zinc-500">Camera + LiDAR + Head</div>
        </div>

        <div>
          <div className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider">LiDAR Grid Granularity</div>
          <div className="text-xl font-bold text-white mt-0.5">{efficiency.voxel_size}</div>
          <div className="text-[10px] text-zinc-500">Voxel pillar dimension</div>
        </div>
      </div>
    </div>
  );
}
