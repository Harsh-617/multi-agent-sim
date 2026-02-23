import Link from "next/link";
import ConfigList from "@/components/ConfigList";

export default function HomePage() {
  return (
    <main className="max-w-3xl mx-auto p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Multi-Agent Simulation</h1>
        <div className="flex gap-2">
          <Link
            href="/league"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            League
          </Link>
          <Link
            href="/runs"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            Run History
          </Link>
          <Link
            href="/reports"
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            Reports
          </Link>
        </div>
      </div>
      <ConfigList />
    </main>
  );
}
