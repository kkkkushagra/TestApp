// src/components/MiniSidebar.jsx
import React from "react";

export default function MiniSidebar({ summary }) {
  return (
    <div className="flex items-center gap-3 bg-transparent p-2 rounded">
      <div className="text-sm text-muted flex items-center gap-2"><span className="kv">📄</span> <span>{summary?.total_clauses || 0}</span></div>
      <div className="text-sm text-red-300 flex items-center gap-2"><span>🚨</span> <span>{summary?.high_risk || 0}</span></div>
      <div className="text-sm text-yellow-300 flex items-center gap-2"><span>⚠️</span> <span>{summary?.medium_risk || 0}</span></div>
      <div className="text-sm text-green-300 flex items-center gap-2"><span>✔️</span> <span>{summary?.low_risk || 0}</span></div>
    </div>
  );
}
