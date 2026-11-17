import type { Metadata } from "next";
import "./globals.css";
import Navigation from "@/components/Navigation";

export const metadata: Metadata = {
  metadataBase: new URL("https://bevfusionlite.vineetraj.tech"),
  title: "BEVFusionLite — Lightweight Camera-LiDAR BEV 3D Object Detector",
  description: "Real-time edge multi-modal 3D perception with LiDAR-depth-guided frustum lifting. 31.2 FPS on NVIDIA T4, -55.9% parameters, evaluated on KITTI and nuScenes.",
  keywords: ["BEV Fusion", "3D Object Detection", "Autonomous Driving", "LiDAR", "Edge AI", "OpenPCDet"],
  authors: [{ name: "Vineet Raj Ojha" }],
  openGraph: {
    title: "BEVFusionLite — Lightweight Camera-LiDAR BEV 3D Object Detector",
    description: "Real-time edge multi-modal 3D perception. 31.2 FPS on NVIDIA T4.",
    url: "https://bevfusionlite.vineetraj.tech",
    siteName: "BEVFusionLite Research Platform",
    images: [
      {
        url: "/data/demo/images/000000_overlay.jpg",
        width: 1248,
        height: 384,
      },
    ],
    locale: "en_US",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-black text-zinc-100 flex flex-col min-h-screen antialiased selection:bg-blue-600 selection:text-white">
        <Navigation />
        <main className="flex-grow">{children}</main>
      </body>
    </html>
  );
}
