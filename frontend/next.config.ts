import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL ?? "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
  async redirects() {
    return [
      {
        source: "/runs",
        destination: "/league",
        permanent: false,
      },
      {
        source: "/replay/:path*",
        destination: "/league",
        permanent: false,
      },
      {
        source: "/reports",
        destination: "/research",
        permanent: false,
      },
      {
        source: "/reports/:path*",
        destination: "/research/:path*",
        permanent: false,
      },
      {
        source: "/competitive",
        destination: "/simulate/head-to-head",
        permanent: false,
      },
      {
        source: "/competitive/league",
        destination: "/league",
        permanent: false,
      },
      {
        source: "/competitive/reports",
        destination: "/research",
        permanent: false,
      },
      {
        source: "/competitive/reports/:path*",
        destination: "/research/:path*",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
