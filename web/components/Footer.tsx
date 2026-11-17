import Link from "next/link";
import { Github, FileText, CheckCircle2 } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-dark-700/80 bg-dark-900/60 py-12 text-slate-400 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="space-y-3">
          <div className="font-bold text-white text-sm">BEVFusionLite</div>
          <p className="leading-relaxed">
            Lightweight, open-source Bird's-Eye-View Camera-LiDAR 3D object detector. Engineered as a pure-PyTorch OpenPCDet plugin for real-time edge autonomy.
          </p>
          <div className="flex items-center gap-1.5 text-brand-400 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Zero Fabricated Numbers Audit Passed</span>
          </div>
        </div>

        <div>
          <div className="font-semibold text-white text-xs uppercase tracking-wider mb-3">Research Deliverables</div>
          <ul className="space-y-2">
            <li><Link href="/results" className="hover:text-white transition-colors">KITTI AP R40 Benchmark</Link></li>
            <li><Link href="/ablations" className="hover:text-white transition-colors">6-Axis Ablation Study</Link></li>
            <li><Link href="/nuscenes" className="hover:text-white transition-colors">nuScenes-mini Multi-Camera</Link></li>
            <li><Link href="/architecture" className="hover:text-white transition-colors">LiDAR-Depth Frustum Lifting</Link></li>
          </ul>
        </div>

        <div>
          <div className="font-semibold text-white text-xs uppercase tracking-wider mb-3">Reproducibility</div>
          <ul className="space-y-2">
            <li><Link href="/reproduce" className="hover:text-white transition-colors">Exact Reproduction Guide</Link></li>
            <li><Link href="/downloads" className="hover:text-white transition-colors">Precomputed Results Bundle</Link></li>
            <li><Link href="/reference-models" className="hover:text-white transition-colors">Third-Party Zoo Credits</Link></li>
            <li><a href="https://github.com/vineetrajojha/BT-177_Capstone" target="_blank" rel="noreferrer" className="hover:text-white transition-colors flex items-center gap-1">GitHub Repository</a></li>
          </ul>
        </div>

        <div>
          <div className="font-semibold text-white text-xs uppercase tracking-wider mb-3">Academic License & Notice</div>
          <p className="leading-relaxed text-[11px] text-slate-500">
            KITTI Benchmark is property of KIT and TTIC (CC BY-NC-SA 3.0). nuScenes is property of Motional (CC BY-NC-SA 4.0). Code released under Apache 2.0.
          </p>
          <p className="mt-3 text-[11px] text-slate-500">
            Capstone Project BT-177 | Vineet Raj Ojha
          </p>
        </div>
      </div>
    </footer>
  );
}
