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
  label: string;
  color: string;
  rating: number;
  isChampion: boolean;
}

const LABELS: [string, string][] = [
  ["Cooperative", "#22c55e"],
  ["Aggressive", "#ef4444"],
  ["Robust", "#14b8a6"],
  ["Unstable", "#6b7280"],
  ["Developing", "#555555"],
];

const WIDTH = 480;
const HEIGHT = 320;
const CONNECTION_DIST = 80;
const PADDING = 30;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function createAgents(): Agent[] {
  const agents: Agent[] = [];

  // Champion
  agents.push({
    id: 0,
    x: rand(PADDING, WIDTH - PADDING),
    y: rand(PADDING, HEIGHT - PADDING),
    vx: rand(-0.4, 0.4),
    vy: rand(-0.4, 0.4),
    radius: 11,
    label: "Champion",
    color: "#14b8a6",
    rating: Math.floor(rand(1000, 1200)),
    isChampion: true,
  });

  // Other agents
  for (let i = 1; i < 10; i++) {
    const [label, color] = LABELS[Math.floor(Math.random() * LABELS.length)];
    agents.push({
      id: i,
      x: rand(PADDING, WIDTH - PADDING),
      y: rand(PADDING, HEIGHT - PADDING),
      vx: rand(-0.4, 0.4),
      vy: rand(-0.4, 0.4),
      radius: Math.floor(rand(5, 10)),
      label,
      color,
      rating: Math.floor(rand(800, 1200)),
      isChampion: false,
    });
  }

  return agents;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AgentCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const agentsRef = useRef<Agent[]>(createAgents());
  const rafRef = useRef<number>(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    // --- draw one frame ---
    function draw() {
      const agents = agentsRef.current;
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      const c = ctx!;

      // Clear
      c.fillStyle = "#0d0d0d";
      c.fillRect(0, 0, WIDTH, HEIGHT);

      // Connection lines
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const dx = agents[i].x - agents[j].x;
          const dy = agents[i].y - agents[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECTION_DIST) {
            const opacity = (1 - dist / CONNECTION_DIST) * 0.25;
            c.beginPath();
            c.moveTo(agents[i].x, agents[i].y);
            c.lineTo(agents[j].x, agents[j].y);
            c.strokeStyle = `rgba(20,184,166,${opacity})`;
            c.lineWidth = 0.5;
            c.stroke();
          }
        }
      }

      // Agents
      for (const agent of agents) {
        // Champion glow ring
        if (agent.isChampion) {
          c.beginPath();
          c.arc(agent.x, agent.y, agent.radius + 3, 0, Math.PI * 2);
          c.strokeStyle = `rgba(20,184,166,0.15)`;
          c.lineWidth = 1;
          c.stroke();
        }

        // Main circle
        c.beginPath();
        c.arc(agent.x, agent.y, agent.radius, 0, Math.PI * 2);
        c.fillStyle = hexToRgba(agent.color, 0.85);
        c.fill();

        // Inner highlight
        c.beginPath();
        c.arc(
          agent.x - agent.radius * 0.25,
          agent.y - agent.radius * 0.25,
          agent.radius * 0.35,
          0,
          Math.PI * 2
        );
        c.fillStyle = "rgba(255,255,255,0.15)";
        c.fill();

        // Champion crown
        if (agent.isChampion) {
          c.font = "8px sans-serif";
          c.fillStyle = "#f59e0b";
          c.textAlign = "center";
          c.fillText("★", agent.x, agent.y - agent.radius - 18);
        }

        // Label
        c.font = "8px sans-serif";
        c.fillStyle = agent.color;
        c.textAlign = "center";
        c.fillText(agent.label, agent.x, agent.y - agent.radius - 10);

        // Rating
        c.font = "9px sans-serif";
        c.fillStyle = "#555555";
        c.textAlign = "center";
        c.fillText(
          String(Math.floor(agent.rating)),
          agent.x,
          agent.y + agent.radius + 12
        );
      }
    }

    // --- animation loop ---
    function animate() {
      const agents = agentsRef.current;

      // Move agents
      for (const agent of agents) {
        agent.x += agent.vx;
        agent.y += agent.vy;

        if (agent.x - agent.radius < 0 || agent.x + agent.radius > WIDTH) {
          agent.vx = -agent.vx;
          agent.x = Math.max(agent.radius, Math.min(WIDTH - agent.radius, agent.x));
        }
        if (agent.y - agent.radius < 0 || agent.y + agent.radius > HEIGHT) {
          agent.vy = -agent.vy;
          agent.y = Math.max(agent.radius, Math.min(HEIGHT - agent.radius, agent.y));
        }
      }

      draw();
      rafRef.current = requestAnimationFrame(animate);
    }

    if (reducedMotion) {
      // Static render only
      draw();
    } else {
      rafRef.current = requestAnimationFrame(animate);
    }

    // Rating bump every 3s
    intervalRef.current = setInterval(() => {
      const agents = agentsRef.current;
      const nonChampions = agents.filter((a) => !a.isChampion);
      const target = nonChampions[Math.floor(Math.random() * nonChampions.length)];
      if (target) {
        target.rating += Math.floor(rand(10, 31));
      }
    }, 3000);

    return () => {
      cancelAnimationFrame(rafRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={WIDTH}
      height={HEIGHT}
      style={{ display: "block", width: WIDTH, height: HEIGHT }}
    />
  );
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
