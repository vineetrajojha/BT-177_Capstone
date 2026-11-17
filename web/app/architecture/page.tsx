"use client";

import { useState } from "react";
import { Cpu, Sparkles } from "lucide-react";

const MODULES = [
  {
    id: "lidar",
    title: "LiDAR Pillar Encoder",
    channels: "32 channels",
    stride: "0.32m grid (216 x 248)",
    params: "0.04 M",
    flops: "1.2 GFLOPs",
    desc: "Point cloud is clipped to x[0, 69.12m], y[-39.68, 39.68m], z[-3, 1m]. 32-dim PillarVFE extracts linear features per occupied vertical pillar, scattered into 2D pseudo-image.",
  },
  {
    id: "camera",
    title: "Camera ResNet-18 + FPN",
    channels: "64 channels",
    stride: "Stride 8 (48 x 156)",
    params: "11.7 M",
    flops: "36.4 GFLOPs",
    desc: "RGB image normalized and padded to 384x1248. ResNet-18 extracts multi-scale features; FPN fuses lateral connections into unified stride-8 representation.",
  },
  {
    id: "lss",
    title: "LiDAR-Guided LSS Transformer",
    channels: "64 context ch, 48 depth bins",
    stride: "D=48 bins [1.0m to 69.0m]",
    params: "0.45 M",
    flops: "14.8 GFLOPs",
    desc: "Sparse LiDAR depth map min-pooled to feature stride; depth head takes [feat, depth, mask] to predict categorical softmax distributions. Lifts 2D into 3D frustum and splats to BEV via pure PyTorch scatter-add.",
  },
  {
    id: "fuser",
    title: "BEV ConvFuser Module",
    channels: "64 fused channels",
    stride: "216 x 248 spatial resolution",
    params: "0.15 M",
    flops: "4.2 GFLOPs",
    desc: "Channel concatenation of Camera BEV (64) and LiDAR BEV (32) = 96 channels -> 2-layer Conv(3x3)-BN-ReLU stack producing unified 64-channel fused BEV representation.",
  },
  {
    id: "head",
    title: "BEV Backbone & AnchorHeadSingle",
    channels: "128 output channels",
    stride: "Stride 2 downsample/upsample",
    params: "2.56 M",
    flops: "11.8 GFLOPs",
    desc: "BaseBEVBackbone downsamples and upsamples fused features. AnchorHeadSingle outputs class classification logits, residual box parameters (x, y, z, w, l, h, theta), and direction bins.",
  },
];

export default function ArchitecturePage() {
  const [activeModule, setActiveModule] = useState(MODULES[2]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">BEVFusionLite Architecture</h1>
        <p className="text-xs text-zinc-400 mt-1">
          Interactive diagram of the Light Model (0.32m LiDAR voxel, ResNet-18, LiDAR-depth-guided lifting). Click to inspect channel dimensions, strides, and computational complexity.
        </p>
      </div>

      {/* Interactive Architecture Flow Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {MODULES.map((mod) => {
          const isSelected = activeModule.id === mod.id;
          return (
            <div
              key={mod.id}
              onClick={() => setActiveModule(mod)}
              className={`p-4 rounded-lg cursor-pointer border transition-all text-left flex flex-col justify-between ${
                isSelected
                  ? "bg-zinc-900 border-zinc-400 shadow-sm"
                  : "bg-zinc-950 border-zinc-800 hover:border-zinc-700"
              }`}
            >
              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase text-zinc-400 font-semibold tracking-wider">
                  {mod.channels}
                </span>
                <h3 className="text-xs font-semibold text-white">{mod.title}</h3>
                <p className="text-[11px] text-zinc-400">{mod.stride}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-zinc-900 flex items-center justify-between text-[11px] font-mono text-zinc-500">
                <span>{mod.params}</span>
                <span>{mod.flops}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Module Detail Inspector Card */}
      <div className="p-5 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-900 pb-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-zinc-400" />
            <h2 className="text-sm font-semibold text-white">{activeModule.title}</h2>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-zinc-400">
            <span>Params: <strong className="text-white">{activeModule.params}</strong></span>
            <span>FLOPs: <strong className="text-zinc-200">{activeModule.flops}</strong></span>
            <span>Channels: <strong className="text-white">{activeModule.channels}</strong></span>
          </div>
        </div>

        <p className="text-xs text-zinc-300 leading-relaxed max-w-4xl">
          {activeModule.desc}
        </p>
      </div>

      {/* Visual Explanation of LiDAR-Depth-Guided Frustum Lifting */}
      <div className="p-6 rounded-lg bg-zinc-950 border border-zinc-800 space-y-4">
        <div className="space-y-1">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-zinc-400" />
            Deep-Dive: LiDAR-Depth-Guided View Transformation
          </h2>
          <p className="text-xs text-zinc-400">
            How metric LiDAR points eliminate monocular line-of-sight feature smearing.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
          {/* Step 1 */}
          <div className="p-4 rounded bg-black border border-zinc-800 space-y-2">
            <div className="text-white font-mono text-xs font-semibold">Step 1: Sparse Projection &amp; Min-Pool</div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              LiDAR points <code className="text-zinc-200 bg-zinc-900 px-1 rounded">(x,y,z)</code> are projected onto the image canvas using <code className="text-zinc-200 bg-zinc-900 px-1 rounded">P2 · R0 · V2C</code>. To handle foreground occluders, valid depths are min-pooled in each 8x8 stride window.
            </p>
            <div className="text-[10px] text-zinc-500 font-mono">Input: Raw points -&gt; [1, 1, 48, 156]</div>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded bg-black border border-zinc-800 space-y-2">
            <div className="text-white font-mono text-xs font-semibold">Step 2: Guided Categorical Depth</div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              The DepthNet head receives concatenated image features + normalized depth + binary validity mask. It predicts softmax distributions across <strong className="text-white">D=48 depth bins</strong> spanning 1m to 69m, supervised by an auxiliary CE loss (0.1 weight).
            </p>
            <div className="text-[10px] text-zinc-500 font-mono">Distribution: Softmax(logits) along D</div>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded bg-black border border-zinc-800 space-y-2">
            <div className="text-white font-mono text-xs font-semibold">Step 3: Frustum Lift &amp; Scatter Splat</div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              The outer product of context features and depth probabilities lifts 2D visual semantics into 3D camera rays. 3D frustum points are mapped to LiDAR coordinates and accumulated into the BEV plane via pure PyTorch <code className="text-zinc-200 bg-zinc-900 px-1 rounded">scatter_add</code>.
            </p>
            <div className="text-[10px] text-zinc-500 font-mono">Output: [B, 64, 248, 216] BEV tensor</div>
          </div>
        </div>
      </div>
    </div>
  );
}
