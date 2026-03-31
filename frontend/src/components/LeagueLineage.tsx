"use client";

import LineageGraph from "./LineageGraph";
import { LineageMember } from "@/lib/api";

interface Props {
  members: LineageMember[];
}

export default function LeagueLineage({ members }: Props) {
  const nodes = members.map((m) => ({
    id: m.member_id,
    parent_id: m.parent_id,
    rating: m.rating,
    created_at: m.created_at,
    notes: m.notes,
    label: m.label,
  }));
  return (
    <LineageGraph
      nodes={nodes}
      emptyMessage="No members yet. Run the pipeline to build a lineage."
    />
  );
}
