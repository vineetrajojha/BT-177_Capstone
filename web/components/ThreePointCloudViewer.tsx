"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";

interface PointCloudViewerProps {
  points: number[]; // flat [x, y, z, ...]
  predictedBoxes: any[];
  gtBoxes: any[];
  showGT: boolean;
  scoreThreshold: number;
  classFilter: string[];
}

export default function ThreePointCloudViewer({
  points,
  predictedBoxes,
  gtBoxes,
  showGT,
  scoreThreshold,
  classFilter,
}: PointCloudViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const boxGroupRef = useRef<THREE.Group | null>(null);
  const pointsObjRef = useRef<THREE.Points | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 600;
    const height = mount.clientHeight || 400;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    sceneRef.current = scene;

    // 2. Camera: In LiDAR coords (x forward, y left, z up)
    // We position camera back and above looking forward
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
    camera.position.set(-20, 0, 15);
    camera.up.set(0, 0, 1);
    camera.lookAt(25, 0, 0);

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.innerHTML = "";
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Grid Helper in XY ground plane
    const gridHelper = new THREE.GridHelper(80, 40, 0x334155, 0x1e293b);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(30, 0, -1.7);
    scene.add(gridHelper);

    // Coordinate axis indicator at sensor origin
    const axes = new THREE.AxesHelper(3);
    scene.add(axes);

    // Group for 3D boxes
    const boxGroup = new THREE.Group();
    scene.add(boxGroup);
    boxGroupRef.current = boxGroup;

    // 5. Orbit Interaction via Mouse Drag
    let isDragging = false;
    let prevMousePos = { x: 0, y: 0 };
    let spherical = { radius: 35, theta: -Math.PI / 2, phi: Math.PI / 3 };
    const target = new THREE.Vector3(25, 0, 0);

    function updateCamera() {
      camera.position.x = target.x + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      camera.position.y = target.y + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      camera.position.z = target.z + spherical.radius * Math.cos(spherical.phi);
      camera.lookAt(target);
    }
    updateCamera();

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;

      spherical.theta -= deltaX * 0.008;
      spherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, spherical.phi - deltaY * 0.008));

      updateCamera();
      prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => { isDragging = false; };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      spherical.radius = Math.max(10, Math.min(100, spherical.radius + e.deltaY * 0.05));
      updateCamera();
    };

    const dom = renderer.domElement;
    dom.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    dom.addEventListener("wheel", onWheel, { passive: false });

    // Render loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!mount) return;
      const newW = mount.clientWidth;
      const newH = mount.clientHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      dom.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
    };
  }, []);

  // Update Points Geometry
  useEffect(() => {
    if (!sceneRef.current || !points || points.length === 0) return;

    if (pointsObjRef.current) {
      sceneRef.current.remove(pointsObjRef.current);
      pointsObjRef.current.geometry.dispose();
    }

    const count = points.length / 3;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points);
    const colors = new Float32Array(points.length);

    // Color by distance
    for (let i = 0; i < count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      const dist = Math.sqrt(x * x + y * y);

      // Gradient from cyan (near) to violet (far)
      const t = Math.min(dist / 50.0, 1.0);
      colors[i * 3] = 0.2 + 0.6 * t;      // R
      colors[i * 3 + 1] = 0.8 * (1.0 - t * 0.5); // G
      colors[i * 3 + 2] = 0.9 * (1.0 - t * 0.2); // B
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.18,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });

    const pointCloud = new THREE.Points(geometry, material);
    sceneRef.current.add(pointCloud);
    pointsObjRef.current = pointCloud;
  }, [points]);

  // Update 3D Wireframe Bounding Boxes
  useEffect(() => {
    if (!boxGroupRef.current) return;
    const group = boxGroupRef.current;

    // Clear existing boxes
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if ((obj as any).geometry) (obj as any).geometry.dispose();
      if ((obj as any).material) (obj as any).material.dispose();
    }

    // Helper to create wireframe 3D box
    const createBox = (box: number[], color: number, isDashed = false) => {
      // box is [x, y, z, l, w, h, yaw] in LiDAR coords
      const [x, y, z, l, w, h, yaw] = box;
      const geom = new THREE.BoxGeometry(l, w, h);
      const edges = new THREE.EdgesGeometry(geom);
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      const line = new THREE.LineSegments(edges, mat);

      line.position.set(x, y, z);
      line.rotation.z = yaw;
      return line;
    };

    // 1. Predicted boxes
    predictedBoxes.forEach((pred) => {
      if (pred.score < scoreThreshold) return;
      if (!classFilter.includes(pred.class)) return;

      const col = pred.class === "Car" ? 0x22c55e : (pred.class === "Pedestrian" ? 0x3b82f6 : 0xf97316);
      group.add(createBox(pred.box_3d, col));
    });

    // 2. GT boxes (dashed / gray or yellow)
    if (showGT && gtBoxes) {
      gtBoxes.forEach((gt) => {
        if (!classFilter.includes(gt.class)) return;
        group.add(createBox(gt.box_3d, 0xfacc15, true));
      });
    }
  }, [predictedBoxes, gtBoxes, showGT, scoreThreshold, classFilter]);

  return (
    <div className="relative w-full h-[450px] rounded-xl overflow-hidden border border-dark-700 bg-dark-900">
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-md bg-dark-900/80 backdrop-blur border border-dark-700 text-[11px] text-slate-400 pointer-events-none flex items-center gap-3">
        <span>Drag: Rotate</span>
        <span>Scroll: Zoom</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-brand-500 inline-block" /> Prediction</span>
        {showGT && <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block" /> Ground Truth</span>}
      </div>
    </div>
  );
}
