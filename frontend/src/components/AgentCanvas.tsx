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

interface AgentCanvasProps {
  obstacleZones?: DOMRect[];
}

const LABELS: [string, string][] = [
  ["Cooperative", "#22c55e"],
  ["Aggressive", "#ef4444"],
  ["Robust", "#14b8a6"],
  ["Unstable", "#6b7280"],
  ["Developing", "#444444"],
];

const CONNECTION_DIST = 120;
const REPULSION_RANGE = 60;
const REPULSION_STRENGTH = 0.8;
const MAX_SPEED = 0.5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rand(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function clamp(val: number, min: number, max: number) {
  return Math.max(min, Math.min(max, val));
}

function isInsideAnyZone(x: number, y: number, zones: DOMRect[]): boolean {
  for (const z of zones) {
    if (x >= z.left && x <= z.right && y >= z.top && y <= z.bottom) {
      return true;
    }
  }
  return false;
}

function createAgents(w: number, h: number, zones: DOMRect[]): Agent[] {
  const agents: Agent[] = [];
  const pad = 20;
  const corners = [
    { x: pad, y: pad },
    { x: w - pad, y: pad },
    { x: pad, y: h - pad },
    { x: w - pad, y: h - pad },
  ];

  function safePosition(): { x: number; y: number } {
    for (let attempt = 0; attempt < 50; attempt++) {
      const x = rand(pad, w - pad);
      const y = rand(pad, h - pad);
      if (!isInsideAnyZone(x, y, zones)) {
        return { x, y };
      }
    }
    // Fallback: pick a corner
    const c = corners[Math.floor(Math.random() * corners.length)];
    return { x: c.x, y: c.y };
  }

  // Champion
  const champPos = safePosition();
  agents.push({
    id: 0,
    x: champPos.x,
    y: champPos.y,
    vx: rand(-0.25, 0.25),
    vy: rand(-0.25, 0.25),
    radius: 7,
    label: "Champion",
    color: "#14b8a6",
  });

  // Other agents
  for (let i = 1; i < 16; i++) {
    const [label, color] = LABELS[Math.floor(Math.random() * LABELS.length)];
    const pos = safePosition();
    agents.push({
      id: i,
      x: pos.x,
      y: pos.y,
      vx: rand(-0.25, 0.25),
      vy: rand(-0.25, 0.25),
      radius: Math.floor(rand(3, 7)),
      label,
      color,
    });
  }

  return agents;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AgentCanvas({ obstacleZones }: AgentCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const agentsRef = useRef<Agent[]>([]);
  const rafRef = useRef<number>(0);
  const zonesRef = useRef<DOMRect[]>([]);

  // Keep zones ref in sync with prop
  useEffect(() => {
    zonesRef.current = obstacleZones ?? [];
  }, [obstacleZones]);

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
    agentsRef.current = createAgents(canvas.width, canvas.height, zonesRef.current);

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

      // Agents — styled with labels
      for (const agent of agents) {
        const isChampion = agent.id === 0;

        // Outer ring (champion only)
        if (isChampion) {
          ctx.beginPath();
          ctx.arc(agent.x, agent.y, agent.radius + 4, 0, Math.PI * 2);
          ctx.strokeStyle = agent.color + "40";
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Main circle
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, agent.radius, 0, Math.PI * 2);
        ctx.fillStyle = agent.color + "66";
        ctx.fill();

        // Strategy label
        ctx.font = "bold 8px 'JetBrains Mono', monospace";
        ctx.fillStyle = agent.color + "99";
        ctx.textAlign = "center";
        ctx.fillText(agent.label, agent.x, agent.y - agent.radius - 4);

        // Champion star
        if (isChampion) {
          ctx.font = "7px sans-serif";
          ctx.fillStyle = "#f59e0b99";
          ctx.textAlign = "center";
          ctx.fillText("★", agent.x, agent.y - agent.radius - 13);
        }
      }
    }

    // --- animation loop ---
    function animate() {
      if (!canvas) return;
      const agents = agentsRef.current;
      const w = canvas.width;
      const h = canvas.height;
      const zones = zonesRef.current;

      for (const agent of agents) {
        agent.x += agent.vx;
        agent.y += agent.vy;

        // Obstacle repulsion
        for (const zone of zones) {
          const clampedX = clamp(agent.x, zone.left, zone.right);
          const clampedY = clamp(agent.y, zone.top, zone.bottom);
          const dx = agent.x - clampedX;
          const dy = agent.y - clampedY;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < REPULSION_RANGE) {
            const len = dist || 1;
            const force = (REPULSION_RANGE - dist) / REPULSION_RANGE * REPULSION_STRENGTH;
            agent.vx += (dx / len) * force * 0.05;
            agent.vy += (dy / len) * force * 0.05;
          }
        }

        // Clamp velocity to max speed
        const speed = Math.sqrt(agent.vx * agent.vx + agent.vy * agent.vy);
        if (speed > MAX_SPEED) {
          agent.vx = (agent.vx / speed) * MAX_SPEED;
          agent.vy = (agent.vy / speed) * MAX_SPEED;
        }

        // Wall bounce
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
