"use client";

import { Shield, ExternalLink, Database, Layers, CheckCircle } from "lucide-react";
import tpData from "../../public/data/third_party.json";

export default function ReferenceModelsPage() {
  const frameworks = [
    {
      name: "OpenPCDet",
      url: "https://github.com/open-mmlab/OpenPCDet",
      commit: "233f849829b6ac19afb8af8837a0246890908755",
      license: "Apache 2.0",
      role: "Base 3D perception library; LiDAR VFE (PillarVFE), PointPillarScatter, 2D BEV Backbone, AnchorHeadSingle, and standard KITTI evaluation protocol.",
      attribution: "OpenPCDet Development Team (OpenMMLab)"
    },
    {
      name: "Lift-Splat-Shoot (LSS)",
      url: "https://github.com/nv-tlabs/lift-splat-shoot",
      commit: "Paper: ECCV 2020",
      license: "Apache 2.0 / MIT",
      role: "Conceptual reference for frustum construction, camera-to-BEV pooling via outer product of camera features and depth distribution.",
      attribution: "Philion & Fidler (NVIDIA Research)"
    },
    {
      name: "BEVFusion (MIT)",
      url: "https://github.com/mit-han-lab/bevfusion",
      commit: "Paper: ICRA 2023",
      license: "MIT",
      role: "Architectural reference for unified dual-stream concatenation and LiDAR-depth guidance concepts.",
      attribution: "Liu et al. (MIT HAN Lab)"
    }
  ];

  const pretrainedWeights = [
    {
      model: "ResNet-18",
      source: "PyTorch / Torchvision Hub",
      checkpoint: "ResNet18_Weights.IMAGENET1K_V1",
      license: "BSD-3-Clause",
      role: "2D camera backbone initialization for BEVFusionLite light model and ablation studies."
    },
    {
      model: "ResNet-50",
      source: "PyTorch / Torchvision Hub",
      checkpoint: "ResNet50_Weights.IMAGENET1K_V2",
      license: "BSD-3-Clause",
      role: "2D camera backbone initialization for Heavy Baseline."
    },
    {
      model: "MobileNetV3-Large",
      source: "PyTorch / Torchvision Hub",
      checkpoint: "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
      license: "BSD-3-Clause",
      role: "2D camera backbone initialization for ultra-lightweight ablation study."
    },
    {
      model: "PointPillars KITTI Zoo Checkpoint",
      source: "OpenPCDet Model Zoo",
      checkpoint: "pointpillar_7728.pth",
      license: "Apache 2.0",
      role: "Published reference baseline comparison (strictly labelled as external published reference; never claimed as our training run)."
    }
  ];

  const datasets = [
    {
      name: "KITTI 3D Object Detection Benchmark",
      provider: "Karlsruhe Institute of Technology & Toyota Technological Institute at Chicago",
      size: "7,481 train/val samples, 7,518 test samples",
      license: "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)",
      usage: "Primary benchmark for 3D object detection across Car, Pedestrian, and Cyclist classes. Synchronized front-camera RGB images and Velodyne HDL-64E LiDAR point clouds."
    },
    {
      name: "nuScenes",
      provider: "Motional",
      size: "v1.0-mini (10 scenes, 6 cameras, 32-beam LiDAR)",
      license: "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)",
      usage: "Secondary benchmark for multi-camera surround-view BEV fusion and 360-degree object detection."
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Title */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Reference Models &amp; Third-Party Assets</h1>
        <p className="text-xs text-zinc-400 mt-1">
          Catalog of external codebases, pretrained checkpoints, and datasets. Source: <code className="text-zinc-300 font-mono text-[11px] bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">{tpData.source_file}</code>
        </p>
      </div>

      {/* Disclaimers & Integrity */}
      <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1.5">
        <div className="flex items-center gap-2 text-zinc-300 font-medium text-xs">
          <Shield className="w-3.5 h-3.5 text-zinc-400" />
          <span>Strict Scientific Integrity Notice</span>
        </div>
        <p className="text-xs text-zinc-400 leading-relaxed">
          None of the pretrained models or external checkpoints listed below were authored by our team. The PointPillars reference baseline is taken verbatim from the peer-reviewed, published OpenPCDet model zoo and is included exclusively for standardized benchmarking comparisons. All ImageNet backbone checkpoints were sourced from the official PyTorch Torchvision release under BSD-3-Clause terms.
        </p>
      </div>

      {/* Frameworks */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-zinc-400" />
          External Frameworks &amp; Libraries
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {frameworks.map((fw, idx) => (
            <div key={idx} className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-3 flex flex-col justify-between">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white text-xs">{fw.name}</h3>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-900 text-zinc-300 border border-zinc-800">
                    {fw.license}
                  </span>
                </div>
                <p className="text-xs text-zinc-400 leading-relaxed">{fw.role}</p>
              </div>
              <div className="pt-3 border-t border-zinc-900 space-y-1">
                <div className="text-[10px] text-zinc-500 font-mono">Ref: {fw.commit}</div>
                <a
                  href={fw.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors font-medium"
                >
                  <span>Repository</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pretrained Weights */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Database className="w-4 h-4 text-zinc-400" />
          Pretrained Model Checkpoints
        </h2>
        <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-900/90 text-zinc-400 border-b border-zinc-800 uppercase font-mono text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4 font-semibold">Model</th>
                <th className="py-3 px-3 font-semibold">Source Provider</th>
                <th className="py-3 px-3 font-semibold">Checkpoint Tag</th>
                <th className="py-3 px-3 font-semibold">License</th>
                <th className="py-3 px-4 font-semibold">Role in BEVFusionLite</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/80 text-zinc-300 font-mono text-xs">
              {pretrainedWeights.map((pw, idx) => (
                <tr key={idx} className="hover:bg-zinc-900/50 transition-colors">
                  <td className="py-3 px-4 font-sans font-medium text-white">{pw.model}</td>
                  <td className="py-3 px-3 font-sans text-zinc-400">{pw.source}</td>
                  <td className="py-3 px-3 text-zinc-300 text-[11px]">{pw.checkpoint}</td>
                  <td className="py-3 px-3 text-zinc-400">{pw.license}</td>
                  <td className="py-3 px-4 font-sans text-zinc-400 text-[11px]">{pw.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Datasets */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-zinc-400" />
          Dataset Attribution &amp; Terms
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {datasets.map((ds, idx) => (
            <div key={idx} className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white text-xs">{ds.name}</h3>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-900 text-zinc-300 border border-zinc-800">
                  Non-Commercial
                </span>
              </div>
              <p className="text-[11px] text-zinc-500">Provider: {ds.provider}</p>
              <p className="text-xs text-zinc-400 leading-relaxed">{ds.usage}</p>
              <div className="pt-2 border-t border-zinc-900 text-[11px] font-mono text-zinc-500">
                License: {ds.license}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
