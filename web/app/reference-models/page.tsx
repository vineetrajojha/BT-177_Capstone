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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Title */}
      <div>
        <h1 className="text-3xl font-black text-white tracking-tight">Reference Models & Third-Party Assets</h1>
        <p className="text-xs text-slate-400 mt-1">
          Catalog of external codebases, pretrained checkpoints, and datasets. Source: <code className="text-brand-400 font-mono">{tpData.source_file}</code>
        </p>
      </div>

      {/* Disclaimers & Integrity */}
      <div className="p-6 rounded-xl bg-amber-500/10 border border-amber-500/20 space-y-2">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
          <Shield className="w-4 h-4" />
          Strict Scientific Integrity Notice
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          None of the pretrained models or external checkpoints listed below were authored by our team. The PointPillars reference baseline is taken verbatim from the peer-reviewed, published OpenPCDet model zoo and is included exclusively for standardized benchmarking comparisons. All ImageNet backbone checkpoints were sourced from the official PyTorch Torchvision release under BSD-3-Clause terms.
        </p>
      </div>

      {/* Frameworks */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-brand-400" />
          External Frameworks & Libraries
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {frameworks.map((fw, idx) => (
            <div key={idx} className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-4 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-white text-sm">{fw.name}</h3>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-dark-700 text-slate-300 border border-dark-600">
                    {fw.license}
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{fw.role}</p>
              </div>
              <div className="pt-4 border-t border-dark-700/60 space-y-2">
                <div className="text-[10px] text-slate-500 font-mono">Ref: {fw.commit}</div>
                <a
                  href={fw.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 text-xs text-brand-400 hover:text-brand-300 transition-colors font-medium"
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
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-brand-400" />
          Pretrained Model Checkpoints
        </h2>
        <div className="overflow-x-auto rounded-xl border border-dark-700 bg-dark-800/40">
          <table className="w-full text-left text-xs">
            <thead className="bg-dark-800/90 text-slate-400 border-b border-dark-700 uppercase font-mono text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4 font-semibold">Model</th>
                <th className="py-3 px-3 font-semibold">Source Provider</th>
                <th className="py-3 px-3 font-semibold">Checkpoint Tag</th>
                <th className="py-3 px-3 font-semibold">License</th>
                <th className="py-3 px-4 font-semibold">Role in BEVFusionLite</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-700/60 text-slate-300 font-mono text-xs">
              {pretrainedWeights.map((pw, idx) => (
                <tr key={idx} className="hover:bg-dark-700/30 transition-colors">
                  <td className="py-3 px-4 font-sans font-bold text-white">{pw.model}</td>
                  <td className="py-3 px-3 font-sans text-slate-400">{pw.source}</td>
                  <td className="py-3 px-3 text-brand-400 text-[11px]">{pw.checkpoint}</td>
                  <td className="py-3 px-3 text-slate-400">{pw.license}</td>
                  <td className="py-3 px-4 font-sans text-slate-300 text-[11px]">{pw.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Datasets */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <CheckCircle className="w-5 h-5 text-brand-400" />
          Dataset Attribution & Terms
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {datasets.map((ds, idx) => (
            <div key={idx} className="p-6 rounded-xl bg-dark-800/80 border border-dark-700 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-white text-sm">{ds.name}</h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-dark-700 text-amber-400 border border-amber-500/20">
                  Non-Commercial
                </span>
              </div>
              <p className="text-xs text-slate-400">Provider: {ds.provider}</p>
              <p className="text-xs text-slate-300 leading-relaxed">{ds.usage}</p>
              <div className="pt-2 border-t border-dark-700/60 text-[11px] font-mono text-slate-500">
                License: {ds.license}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
