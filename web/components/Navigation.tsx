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
];

export default function Navigation() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-black/90 backdrop-blur-md border-b border-zinc-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center py-1">
              <span className="font-semibold text-base text-white tracking-tight lowercase">
                bevfusionlite
              </span>
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
                  className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    active
                      ? "bg-zinc-800 text-white border border-zinc-700/80 shadow-sm"
                      : "text-zinc-400 hover:text-white hover:bg-zinc-900"
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
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              <span>GitHub</span>
              <ExternalLink className="w-3 h-3 text-zinc-500" />
            </a>
          </div>

          {/* Mobile hamburger */}
          <div className="lg:hidden flex items-center">
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-2 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-900"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="lg:hidden px-4 pt-2 pb-4 space-y-1 bg-black border-b border-zinc-800">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium ${
                  active ? "bg-zinc-800 text-white border border-zinc-700" : "text-zinc-400 hover:text-white hover:bg-zinc-900"
                }`}
              >
                <item.icon className="w-4 h-4 text-zinc-400" />
                {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
