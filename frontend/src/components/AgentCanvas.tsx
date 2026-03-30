"use client";

import { useEffect, useRef } from "react";

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

interface Agent {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  label: string;
}

const LABELS: [string, string][] = [
  ["Cooperative", "#22c55e"],
  ["Aggressive", "#ef4444"],
  ["Robust", "#14b8a6"],
  ["Unstable", "#6b7280"],
  ["Developing", "#444444"],
];

const CONNECTION_DIST = 120;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function createAgents(w: number, h: number): Agent[] {
  const agents: Agent[] = [];
  const pad = 20;

  // Champion
  agents.push({
    id: 0,
    x: rand(pad, w - pad),
    y: rand(pad, h - pad),
    vx: rand(-0.25, 0.25),
    vy: rand(-0.25, 0.25),
    radius: 7,
    label: "Champion",
    color: "#14b8a6",
  });

  // Other agents
  for (let i = 1; i < 16; i++) {
    const [label, color] = LABELS[Math.floor(Math.random() * LABELS.length)];
    agents.push({
      id: i,
      x: rand(pad, w - pad),
      y: rand(pad, h - pad),
      vx: rand(-0.25, 0.25),
      vy: rand(-0.25, 0.25),
      radius: Math.floor(rand(3, 7)),
      label,
      color,
    });
  }

  return agents;
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AgentCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const agentsRef = useRef<Agent[]>([]);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Size canvas to window
    function resize() {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();

    // Create agents sized to current viewport
    agentsRef.current = createAgents(canvas.width, canvas.height);

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    // --- draw one frame ---
    function draw() {
      if (!canvas || !ctx) return;
      const agents = agentsRef.current;
      const w = canvas.width;
      const h = canvas.height;

      // Clear (fully transparent)
      ctx.clearRect(0, 0, w, h);

      // Connection lines
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const dx = agents[i].x - agents[j].x;
          const dy = agents[i].y - agents[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECTION_DIST) {
            const opacity = (1 - dist / CONNECTION_DIST) * 0.12;
            ctx.beginPath();
            ctx.moveTo(agents[i].x, agents[i].y);
            ctx.lineTo(agents[j].x, agents[j].y);
            ctx.strokeStyle = `rgba(20,184,166,${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Agents — simple filled circles only
      for (const agent of agents) {
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, agent.radius, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(agent.color, 0.4);
        ctx.fill();
      }
    }

    // --- animation loop ---
    function animate() {
      if (!canvas) return;
      const agents = agentsRef.current;
      const w = canvas.width;
      const h = canvas.height;

      for (const agent of agents) {
        agent.x += agent.vx;
        agent.y += agent.vy;

        if (agent.x - agent.radius < 0 || agent.x + agent.radius > w) {
          agent.vx = -agent.vx;
          agent.x = Math.max(agent.radius, Math.min(w - agent.radius, agent.x));
        }
        if (agent.y - agent.radius < 0 || agent.y + agent.radius > h) {
          agent.vy = -agent.vy;
          agent.y = Math.max(agent.radius, Math.min(h - agent.radius, agent.y));
        }
      }

      draw();
      rafRef.current = requestAnimationFrame(animate);
    }

    if (reducedMotion) {
      draw();
    } else {
      rafRef.current = requestAnimationFrame(animate);
    }

    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 0,
        pointerEvents: "none",
        opacity: 0.35,
        background: "transparent",
      }}
    />
  );
}
