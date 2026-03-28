"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navLinks = [
  { label: "Simulate", href: "/simulate" },
  { label: "League", href: "/league" },
  { label: "Research", href: "/research" },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/league") return pathname === "/league";
  return pathname === href || pathname.startsWith(href + "/");
}

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        height: 48,
        background: "var(--bg-surface)",
        borderBottom: "1px solid var(--bg-border)",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      {/* Left — wordmark */}
      <div style={{ paddingLeft: 24 }}>
        <Link
          href="/"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--text-primary)",
            letterSpacing: "0.08em",
            textDecoration: "none",
          }}
        >
          masp
        </Link>
      </div>

      {/* Center — nav links */}
      <div style={{ display: "flex", gap: 32 }}>
        {navLinks.map((link) => {
          const active = isActive(pathname, link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              style={{
                fontSize: 13,
                color: active
                  ? "var(--text-primary)"
                  : "var(--text-secondary)",
                fontWeight: active ? 500 : 400,
                textDecoration: "none",
                transition: "color 150ms",
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  e.currentTarget.style.color = "var(--text-primary)";
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  e.currentTarget.style.color = "var(--text-secondary)";
                }
              }}
            >
              {link.label}
            </Link>
          );
        })}
      </div>

      {/* Right — reserved */}
      <div style={{ paddingRight: 24, minWidth: 48 }} />
    </nav>
  );
}
