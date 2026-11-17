"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  Box, 
  BarChart3, 
  Layers, 
  Cpu, 
  BookOpen, 
  RefreshCw, 
  Download, 
  Info, 
  Github, 
  Menu, 
  X,
  ExternalLink,
  ShieldCheck
} from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = [
  { name: "Home", href: "/", icon: Box },
  { name: "Demo", href: "/demo", icon: Box },
  { name: "Results", href: "/results", icon: BarChart3 },
  { name: "Ablations", href: "/ablations", icon: Layers },
  { name: "nuScenes", href: "/nuscenes", icon: Layers },
  { name: "Arch", href: "/architecture", icon: Cpu },
  { name: "Literature", href: "/literature", icon: BookOpen },
  { name: "Reproduce", href: "/reproduce", icon: RefreshCw },
  { name: "References", href: "/reference-models", icon: ShieldCheck },
  { name: "Downloads", href: "/downloads", icon: Download },
  { name: "About", href: "/about", icon: Info },
];

export default function Navigation() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-dark-900/90 backdrop-blur-md border-b border-dark-700/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-emerald-400 flex items-center justify-center font-black text-dark-900 shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
                B
              </div>
              <div>
                <span className="font-bold text-lg text-white tracking-tight">BEVFusion<span className="text-brand-500">Lite</span></span>
                <span className="hidden sm:inline-block ml-2 px-2 py-0.5 text-[10px] font-semibold bg-brand-500/10 text-brand-400 border border-brand-500/20 rounded-full">
                  Edge 3D Perception
                </span>
              </div>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    active
                      ? "bg-dark-700 text-brand-400 border border-brand-500/30"
                      : "text-slate-300 hover:text-white hover:bg-dark-800"
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>

          <div className="hidden sm:flex items-center gap-2">
            <a
              href="https://github.com/vineetrajojha/BT-177_Capstone"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-dark-800 hover:bg-dark-700 border border-dark-600 text-slate-200 transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              <span>GitHub</span>
              <ExternalLink className="w-3 h-3 text-slate-400" />
            </a>
          </div>

          {/* Mobile hamburger */}
          <div className="lg:hidden flex items-center">
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-2 rounded-md text-slate-400 hover:text-white hover:bg-dark-800"
            >
              {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="lg:hidden px-4 pt-2 pb-4 space-y-1 bg-dark-900 border-b border-dark-700">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium ${
                  active ? "bg-dark-700 text-brand-400" : "text-slate-300 hover:bg-dark-800"
                }`}
              >
                <item.icon className="w-4 h-4" />
                {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
